"""Sequential real HTTP extraction of the downloaded arXiv PDF corpus."""
import hashlib
import importlib
import json
import os
import shutil
import time
from pathlib import Path

import django
import requests
import pymupdf

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()
from django.conf import settings
from django.contrib.auth import get_user_model
from contracts.models import Contract
from ai_extract.models import LayoutExtractionRun
from ai_extract.repositories import layout_repository
from ai_extract.repositories import document_graph_repository
from django.utils import timezone

ROOT = Path('/app')
PDFS = ROOT / 'PDF_180_2_3_TRANG'
REPORT = ROOT / 'evaluation' / 'corpus180_short_main_pipeline.json'
LOG = ROOT / 'evaluation' / 'corpus180_short_main_pipeline.log'
ORIGIN = 'http://127.0.0.1:8000'


def save(state):
    temp = REPORT.with_suffix('.json.part')
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, REPORT)


def emit(event, **fields):
    with LOG.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(event=event, timestamp=time.time(), **fields), ensure_ascii=False, default=str) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    manifest = json.loads((PDFS / 'manifest.json').read_text(encoding='utf-8'))
    entries = sorted(manifest['items'], key=lambda item: item['number'])
    if len(entries) != 180 or any(row.get('status') != 'downloaded' for row in entries):
        raise ValueError('The input manifest does not contain 180 downloaded PDFs')
    state = json.loads(REPORT.read_text(encoding='utf-8')) if REPORT.exists() else {
        'status': 'starting', 'total': 180, 'force_ocr': True, 'defer_colpali': True, 'documents': {}}
    user = get_user_model().objects.get(username='corpus20_local', is_active=True)
    auth = importlib.import_module(settings.SESSION_ENGINE).SessionStore()
    auth['_auth_user_id'] = str(user.pk)
    auth['_auth_user_backend'] = settings.AUTHENTICATION_BACKENDS[0]
    auth['_auth_user_hash'] = user.get_session_auth_hash()
    auth.save()
    client = requests.Session()
    page = client.get(ORIGIN + '/admin/login/', timeout=30)
    page.raise_for_status()
    csrf = client.cookies.get(settings.CSRF_COOKIE_NAME)
    if not csrf:
        raise RuntimeError('Could not obtain CSRF cookie')
    client.cookies.set(settings.SESSION_COOKIE_NAME, auth.session_key)
    headers = {'X-CSRFToken': csrf, 'Referer': ORIGIN + '/admin/login/'}
    state['status'] = 'running'
    state.pop('reason', None)
    save(state)
    for entry in entries:
        key = str(entry['number'])
        row = state['documents'].setdefault(key, dict(number=entry['number'], file=entry['file'],
            sha256=entry['sha256'], status='pending'))
        if row.get('status') == 'completed':
            continue
        try:
            with document_graph_repository.connection() as graph:
                graph.run('RETURN 1 AS ok').consume()
            endpoint = os.environ.get('RAG_INFERENCE_URL', '').rstrip('/')
            if not endpoint:
                raise RuntimeError('RAG_INFERENCE_URL is not configured')
            requests.get(endpoint + '/rag/health', timeout=10).raise_for_status()
        except Exception as error:
            state.update(status='paused_infrastructure', reason=f'{type(error).__name__}: {error}'[:500])
            save(state); emit('paused_infrastructure', number=entry['number'], error_type=type(error).__name__)
            return
        if min(shutil.disk_usage(ROOT).free, shutil.disk_usage('/').free) < 3 * 1024**3:
            state.update(status='paused_low_disk', reason='Less than 3 GiB free on a required filesystem')
            save(state); emit('paused_low_disk', number=entry['number']); return
        source = PDFS / entry['file']
        if not source.is_file() or sha256(source) != entry['sha256']:
            row.update(status='failed', error='Input file missing or SHA-256 mismatch')
            save(state); emit('input_failed', number=entry['number']); continue
        try:
            with pymupdf.open(source) as pdf:
                pages = len(pdf)
            if pages not in (2, 3):
                raise ValueError(f'Expected 2 or 3 pages, found {pages}')
        except Exception as error:
            row.update(status='failed', error=f'Invalid PDF: {error}')
            save(state); emit('input_failed', number=entry['number']); continue
        code = 'ARXIV180-' + entry['sha256'][:24]
        contract = Contract.objects.filter(contract_code=code).first()
        if contract and contract.company_id != user.company_id:
            row.update(status='failed', error='Existing contract code belongs to a different company')
            save(state); continue
        try:
            if not contract:
                row['status'] = 'uploading'; save(state)
                with source.open('rb') as file:
                    response = client.post(ORIGIN + '/api/contracts/create/',
                        data={'contract_code': code, 'title': entry['title'][:255], 'contract_type': entry['category'][:100]},
                        files={'file': (entry['file'], file, 'application/pdf')}, headers=headers, timeout=(10, 300))
                response.raise_for_status()
                body = response.json()
                if not body.get('success'):
                    raise RuntimeError('Project upload failed: ' + str(body)[:500])
                contract = Contract.objects.get(pk=body['contract_id'])
            version = contract.latest_version
            if not version or not version.files.exists():
                raise RuntimeError('Created contract is missing its source version/file')
            row.update(contract_id=contract.pk, version_id=version.pk)
            status_url = ORIGIN + f'/api/ai/contracts/{contract.pk}/extraction-status/'
            if row.get('status') in {'extracting', 'failed'}:
                status = client.get(status_url, timeout=30).json()
                while status.get('status') == 'RUNNING':
                    # A web restart can leave a RUNNING row after its workflow
                    # lease was released. Only retire it while holding that
                    # same lock, so a live extraction is never interrupted.
                    with layout_repository.workflow_slot() as available:
                        if available:
                            changed = LayoutExtractionRun.objects.filter(
                                pk=status.get('run_id'), version_id=version.pk, status='RUNNING'
                            ).update(status='FAILED', completed_at=timezone.now(),
                                     error_message='Interrupted when the extraction process stopped; safe to retry.')
                            if changed:
                                emit('interrupted_run_recovered', number=entry['number'], run_id=status.get('run_id'))
                                status = client.get(status_url, timeout=30).json()
                                continue
                    time.sleep(30)
                    status = client.get(status_url, timeout=30).json()
                if (status.get('status') == 'COMPLETED'
                        and status.get('processing', {}).get('force_ocr') is True
                        and status.get('processing', {}).get('defer_colpali') is True):
                    row.update(status='completed', recovered_status=status, run_id=status.get('run_id'))
                    save(state); emit('recovered', number=entry['number'], run_id=row['run_id']); continue
            row.update(status='extracting', started_at=time.time()); save(state)
            emit('extract_started', number=entry['number'], contract_id=contract.pk)
            response = client.post(ORIGIN + f'/api/ai/contracts/{contract.pk}/layout-extract/',
                json={'force_ocr': True, 'defer_colpali': True}, headers=headers, timeout=(10, None))
            body = response.json()
            row.update(http_status=response.status_code, result=body, finished_at=time.time(), run_id=body.get('run_id'))
            row['status'] = 'completed' if response.status_code == 200 and body.get('success') is True else 'failed'
            save(state); emit('extract_finished', number=entry['number'], status=row['status'], run_id=row.get('run_id'))
            if response.status_code >= 500:
                state.update(status='paused_pipeline_error', reason=str(body.get('error', 'Server error'))[:500])
                save(state); emit('paused_pipeline_error', number=entry['number'], http_status=response.status_code)
                return
            if row['status'] == 'failed' and body.get('graph_index', {}).get('status') == 'unavailable':
                state.update(status='paused_infrastructure', reason='Neo4j graph publication unavailable')
                save(state); emit('paused_infrastructure', number=entry['number'], service='neo4j'); return
            failed_pages = body.get('failed_pages') or []
            if (row['status'] == 'failed' and failed_pages and all(
                    page.get('stage') == 'surya' and page.get('error_type') == 'ConnectionError'
                    for page in failed_pages)):
                state.update(status='paused_inference_unavailable', reason='Surya connection failed; retry after recovery')
                save(state); emit('paused_inference_unavailable', number=entry['number']); return
        except (requests.RequestException, ValueError, RuntimeError, OSError) as error:
            row.update(status='failed', error_type=type(error).__name__, error=str(error)[:600]); save(state)
            emit('document_failed', number=entry['number'], error_type=type(error).__name__, error=str(error)[:300])
            state.update(status='paused_pipeline_error', reason=f'{type(error).__name__}: {error}'[:500])
            save(state); return
    state['status'] = 'finished'
    save(state)
    emit('finished', completed=sum(r.get('status') == 'completed' for r in state['documents'].values()),
         failed=sum(r.get('status') == 'failed' for r in state['documents'].values()))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        emit('fatal', error_type=type(error).__name__, error=str(error)[:600])
        raise

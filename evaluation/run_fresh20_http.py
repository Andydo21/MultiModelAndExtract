"""HTTP-only stability client for the 20 real PDFs; no pipeline/graph calls.

Run via manage.py shell for a temporary authenticated local session. All source
processing and identity work must be performed by project endpoints/workers.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import time
import requests
from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils import timezone
from evaluation.batch_io import save_state


def main():
    root=Path('/app')
    base=root/'evaluation/benchmarks/corpus20'
    selection=json.loads((root/'evaluation/benchmarks/corpus20_short_selection.json').read_text())
    previous=json.loads((base/'a3_ingestion.json').read_text())
    by_alias={d['alias']:d for d in previous['documents']}
    plan=[by_alias[alias] for alias in selection['documents']]
    if len(plan)!=20 or len({d['version_id'] for d in plan})!=20:
        raise ValueError('Expected the existing 20 distinct real source versions')
    limit=int(os.getenv('FRESH20_LIMIT','20'))
    if not 1<=limit<=20:raise ValueError('Document limit must be between 1 and 20')
    plan=plan[:limit]
    only=os.getenv('FRESH20_ONLY')
    if only:
        requested=only.split(',')
        if len(set(requested))!=len(requested) or not set(requested)<={d['alias'] for d in plan}:
            raise ValueError('Requested aliases must be distinct members of the selected corpus')
        plan=[d for d in plan if d['alias'] in requested]
        limit=len(plan)
    for item in plan:
        if hashlib.sha256((root/item['source_path']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Source PDF changed: '+item['alias'])
    if os.getenv('FRESH20_RESUME')=='1':
        location=json.loads((base/'fresh20_http_latest.json').read_text())
        native,host=Path(location['native']),Path(location['host'])
        if native.parent!=Path('/tmp') or host.parent!=base:raise ValueError('Invalid resume artifact path')
        report=json.loads((native/'manifest.json').read_text())
        if [d['version_id'] for d in report['planned_documents']]!=[d['version_id'] for d in plan]:
            raise ValueError('Resume must use the same document selection and limit')
        report.setdefault('client_interruptions',[]).append(dict(at=timezone.now().isoformat(),
            reason='HTTP client restarted to wait for identity after each document; source processing remains in project services'))
    else:
        stamp=timezone.now().strftime('%Y%m%dT%H%M%SZ')
        native=Path('/tmp')/('fresh20_http_'+stamp)
        native.mkdir()
        host=base/('fresh20_http_'+stamp)
        host.mkdir()
        report=dict(started_at=timezone.now().isoformat(),transport='real project HTTP API',
            pipeline_schema='generic-entity-relations-compacted-v3',
            force_ocr=True,defer_colpali=True,expected_identity_model='fastino/gliner2-base-v1',
            status='STARTING',requested_document_limit=limit,documents=[],planned_documents=plan)
    def save():
        save_state(native/'manifest.json',report)
        try:save_state(host/'manifest.json',report)
        except OSError as error:
            print(json.dumps(dict(event='artifact_mirror_failed',error_type=type(error).__name__,
                authoritative_path=str(native/'manifest.json'))),flush=True)
    save_state(base/'fresh20_http_latest.json',dict(host=str(host),native=str(native),
        log_container='riskdl_web',client_log='/tmp/fresh20_http_client.log'))
    user=get_user_model().objects.get(username='admin',is_active=True,company_id=5)
    auth=importlib.import_module(settings.SESSION_ENGINE).SessionStore()
    auth['_auth_user_id']=str(user.pk)
    auth['_auth_user_backend']=settings.AUTHENTICATION_BACKENDS[0]
    auth['_auth_user_hash']=user.get_session_auth_hash()
    auth.save()
    origin='http://127.0.0.1:8000'
    try:
        with requests.Session() as client:
            page=client.get(origin+'/admin/login/',timeout=20)
            page.raise_for_status()
            csrf=client.cookies.get(settings.CSRF_COOKIE_NAME)
            if not csrf:raise ValueError('Project CSRF token unavailable')
            client.cookies.set(settings.SESSION_COOKIE_NAME,auth.session_key)
            headers={'X-CSRFToken':csrf,'Referer':origin+'/admin/login/'}
            if os.getenv('FRESH20_RESET_GRAPHS')=='1':
                if report['documents']:
                    raise ValueError('Graph reset requires a new cohort, never an interrupted extraction')
                report['status']='RESETTING_OLD_GRAPHS'
                report['graph_resets']=[]
                save()
                for item in plan:
                    reset=client.post(origin+f"/api/ai/contracts/{item['contract_id']}/reset-graph/",
                        json={},headers=headers,timeout=120)
                    record=dict(alias=item['alias'],contract_id=item['contract_id'],http_status=reset.status_code)
                    try:record['response']=reset.json()
                    except ValueError:record['response_text']=reset.text
                    report['graph_resets'].append(record)
                    save()
                    reset.raise_for_status()
                    result=record.get('response',{})
                    if (result.get('success') is not True or result.get('remaining_nodes')!=0
                            or result.get('remaining_snapshots')!=0
                            or result.get('unrelated_nodes_before')!=result.get('unrelated_nodes_after')):
                        raise RuntimeError('Project graph reset verification failed')
                    print(json.dumps(dict(event='fresh20_graph_reset',**record)),flush=True)
            def wait_identity(row):
                expected=row['extraction_run_id']
                status_url=origin+f"/api/ai/contracts/{row['contract_id']}/extraction-status/"
                row['status']='VERIFYING_PROJECT_PREGRAPH_IDENTITY'
                submitted=False
                read_failures=0
                while True:
                    try:
                        response=client.get(status_url,timeout=(5,60))
                    except (requests.Timeout,requests.ConnectionError) as error:
                        read_failures+=1
                        row.setdefault('status_poll_errors',[]).append(dict(at=timezone.now().isoformat(),
                            error_type=type(error).__name__,message=str(error),attempt=read_failures))
                        save()
                        print(json.dumps(dict(event='fresh20_status_transport_error',alias=row['alias'],
                            attempt=read_failures,error_type=type(error).__name__)),flush=True)
                        if read_failures>=3:raise
                        time.sleep(3)
                        continue
                    read_failures=0
                    response.raise_for_status()
                    actual=response.json()
                    row['latest_project_status']=actual
                    if actual.get('run_id')!=expected:raise RuntimeError('Source run changed during identity test')
                    identity=actual.get('identity',{})
                    result=identity.get('result') or {}
                    if identity.get('state')=='complete':
                        if result.get('model')!=report['expected_identity_model']:
                            raise RuntimeError('Unexpected identity model; no fallback accepted')
                        row['status']='IDENTITY_PROCESSED_PENDING_REVIEW'
                        save()
                        print(json.dumps(dict(event='fresh20_identity_finished',alias=row['alias'],
                            run_id=expected,result=result)),flush=True)
                        return
                    if identity.get('state')=='failed':
                        row['status']='IDENTITY_FAILED';save()
                        raise RuntimeError('Project identity worker failed; test stopped at this document')
                    if not submitted and identity.get('state')!='running':
                        queued=client.post(origin+f"/api/ai/contracts/{row['contract_id']}/resolve-entities/",
                            json={},headers=headers,timeout=30)
                        row['identity_enqueue']=dict(http_status=queued.status_code,response=queued.json())
                        queued.raise_for_status();submitted=True
                        print(json.dumps(dict(event='fresh20_identity_requested',alias=row['alias'],
                            run_id=expected,response=row['identity_enqueue'])),flush=True)
                    save()
                    time.sleep(5)
            report['status']='EXTRACTING_FROM_DOCUMENT_1'
            for ordinal,item in enumerate(plan,1):
                url=origin+f"/api/ai/contracts/{item['contract_id']}/layout-extract/"
                existing=next((r for r in report['documents'] if r['alias']==item['alias']),None)
                if (existing and existing['status'] in {'FAILED','TRANSPORT_FAILED'}
                        and os.getenv('FRESH20_RETRY_FAILED')=='1'):
                    report.setdefault('failed_document_attempts',[]).append(dict(existing,
                        archived_for_retry_at=timezone.now().isoformat()))
                    report['documents'].remove(existing)
                    existing=None
                    save()
                if existing is not None:
                    prior=existing.get('response')
                    if isinstance(prior,dict) and prior.get('success') and prior.get('force_ocr'):
                        existing['extraction_run_id']=prior['run_id']
                        wait_identity(existing)
                        continue
                    if existing['status']=='REQUESTING':
                        # A restarted HTTP client may miss the original POST
                        # response. Recover only facts returned by the project
                        # status API; retain that the POST response was lost.
                        status=client.get(origin+f"/api/ai/contracts/{item['contract_id']}/extraction-status/",timeout=30)
                        status.raise_for_status();actual=status.json()
                        if actual.get('status')=='COMPLETED' and actual.get('processing',{}).get('force_ocr') is True:
                            existing.update(extraction_run_id=actual['run_id'],
                                recovered_project_status=actual,original_post_response_not_observed=True)
                            wait_identity(existing)
                            continue
                        raise RuntimeError('Interrupted document has not completed; no result fabricated')
                    raise RuntimeError('Prior document failed; inspect the saved real response before resuming')
                row=dict(ordinal=ordinal,alias=item['alias'],contract_id=item['contract_id'],
                    version_id=item['version_id'],url=url,started_at=timezone.now().isoformat(),
                    request=dict(force_ocr=True,defer_colpali=True),status='REQUESTING')
                report['documents'].append(row)
                save()
                print(json.dumps(dict(event='fresh20_http_start',**row)),flush=True)
                started=time.monotonic()
                try:
                    response=client.post(url,json=row['request'],headers=headers,
                        timeout=(10,None),allow_redirects=False)
                    row['http_status']=response.status_code
                    try:row['response']=response.json()
                    except ValueError:row['response']=response.text
                    body=row['response']
                    row['status']=('EXTRACTED_PENDING_IDENTITY' if response.status_code==200
                        and isinstance(body,dict) and body.get('success') is True
                        and body.get('force_ocr') is True else 'FAILED')
                except requests.RequestException as error:
                    row.update(status='TRANSPORT_FAILED',error_type=type(error).__name__,error=str(error))
                row['elapsed_s']=round(time.monotonic()-started,3)
                save()
                print(json.dumps(dict(event='fresh20_http_finish',ordinal=ordinal,alias=row['alias'],
                    status=row['status'],http_status=row.get('http_status'),
                    run_id=row.get('response',{}).get('run_id') if isinstance(row.get('response'),dict) else None)),flush=True)
                if row['status']!='EXTRACTED_PENDING_IDENTITY':
                    raise RuntimeError('Extraction failed; test stopped at this document')
                row['extraction_run_id']=row['response']['run_id']
                wait_identity(row)
            report['status']='EXTRACTION_REQUESTS_FINISHED_CHECKING_IDENTITY'
            save()
            # Observe the ordinary worker through project GET endpoints. Do not
            # claim jobs, modify receipts, or manufacture a completed result.
            deadline=time.monotonic()+24*3600
            while time.monotonic()<deadline:
                pending=0
                for row in report['documents']:
                    if row['status'] in {'FAILED','TRANSPORT_FAILED'}:continue
                    url=origin+f"/api/ai/contracts/{row['contract_id']}/extraction-status/"
                    try:
                        response=client.get(url,timeout=30)
                        response.raise_for_status()
                        status=response.json()
                        row['latest_project_status']=status
                        identity=status.get('identity',{})
                        result=identity.get('result') or {}
                        if status.get('run_id')!=row['response'].get('run_id'):
                            row['status']='SUPERSEDED_DURING_TEST'
                        elif identity.get('state')=='complete':
                            row['status']=('IDENTITY_PROCESSED_PENDING_REVIEW' if
                                result.get('model')==report['expected_identity_model'] else 'WRONG_IDENTITY_MODEL')
                        elif identity.get('state')=='failed':row['status']='IDENTITY_FAILED'
                        else:pending+=1
                    except (requests.RequestException,ValueError) as error:
                        row['status_poll_error']=dict(type=type(error).__name__,message=str(error))
                        pending+=1
                save()
                if not pending:break
                time.sleep(20)
            report['status']='FINISHED_WITH_RECORDED_RESULTS'
            report['finished_at']=timezone.now().isoformat()
            save()
    finally:
        auth.delete()
        save()


if __name__=='__main__':
    import contextlib
    import traceback
    with open('/tmp/fresh20_http_client.log','a',encoding='utf-8',buffering=1) as log, \
            contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:main()
        except BaseException:
            traceback.print_exc()
            raise

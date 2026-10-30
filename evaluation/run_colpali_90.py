"""Backfill ColPali for a selected range of completed short-corpus documents."""
import json
import os
import time
from pathlib import Path

import django
import requests

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()
from ai_extract.models import LayoutExtractionRun
from ai_extract.services.layout_service import backfill_colpali
from ai_extract.services.contract_pipeline_service import WorkflowBusyError

ROOT = Path('/app/evaluation')
SOURCE = ROOT / 'corpus180_short_main_pipeline.json'
START = int(os.environ.get('COLPALI_CORPUS_START', '1'))
END = int(os.environ.get('COLPALI_CORPUS_END', '90'))
if not 1 <= START <= END <= 180:
    raise ValueError('Invalid short-corpus document range')
REPORT = ROOT / ('colpali90_backfill.json' if (START, END) == (1, 90)
                 else f'colpali_{START}_{END}_backfill.json')


def save(state):
    tmp = REPORT.with_suffix('.json.part')
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, REPORT)


def main():
    source = json.loads(SOURCE.read_text(encoding='utf-8'))['documents']
    selected = [source[str(i)] for i in range(START, END + 1)]
    if any(item.get('status') != 'completed' or not item.get('run_id') for item in selected):
        raise ValueError('All selected documents must have completed source runs')
    base = os.environ.get('COLPALI_BASE_URL', '').rstrip('/')
    if not base:
        raise ValueError('COLPALI_BASE_URL is required')
    requests.get(base + '/health', timeout=20).raise_for_status()
    state = json.loads(REPORT.read_text(encoding='utf-8')) if REPORT.exists() else {
        'total': END - START + 1, 'selection': f'short corpus documents {START}-{END}', 'documents': {}}
    state.update(status='running', started_at=state.get('started_at', time.time()))
    state.pop('reason', None)
    save(state)
    for item in selected:
        key = str(item['number'])
        row = state['documents'].setdefault(key, {'number': item['number'], 'run_id': item['run_id']})
        if row.get('status') in ('completed', 'no_figures') and row.get('run_id') == item['run_id']:
            continue
        row.update(status='processing', run_id=item['run_id'], started_at=time.time())
        save(state)
        print('starting document', key, 'run', item['run_id'], flush=True)
        try:
            run = LayoutExtractionRun.objects.select_related('version__contract').get(pk=item['run_id'])
            for attempt in range(12):
                try:
                    result = backfill_colpali(run, only_missing=True)
                    break
                except WorkflowBusyError:
                    if attempt == 11:
                        raise
                    time.sleep(10)
            status = result.get('status')
            if status == 'SKIPPED' and result.get('images') == 0:
                row_status = 'no_figures'
            elif status == 'COMPLETED' and result.get('graph', {}).get('status') == 'ready':
                row_status = 'completed'
            else:
                row_status = 'failed'
            row.update(status=row_status, result=result, finished_at=time.time())
            save(state)
            print('finished document', key, 'status', row_status, 'images', result.get('images'),
                  'points', result.get('points'), 'error', result.get('error'), flush=True)
            if row_status == 'failed':
                try:
                    requests.get(base + '/health', timeout=20).raise_for_status()
                except requests.RequestException as error:
                    state.update(status='paused_gateway_unavailable', reason=type(error).__name__)
                    save(state)
                    return
        except Exception as error:
            row.update(status='failed', error_type=type(error).__name__, error=str(error)[:500], finished_at=time.time())
            state.update(status='paused_error', reason=f'{type(error).__name__}: {error}'[:500])
            save(state)
            print('paused document', key, 'error', type(error).__name__, str(error)[:300], flush=True)
            return
    state.update(status='finished', finished_at=time.time())
    save(state)
    print('finished batch', flush=True)


if __name__ == '__main__':
    main()

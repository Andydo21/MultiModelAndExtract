"""Run the visual question set through the production NewIdea search path."""
import json
import os
import contextlib
import re
import time
from pathlib import Path

import django
django.setup()

from django.contrib.auth import get_user_model
from NewIdea.services.knowledge_search_service import search_knowledge

root = Path('/app')
questions = re.findall(r'(?m)^(\d+)\. (.+)$', (root / 'CAU_HOI_TEST_COLPALI_20_PDF.md').read_text(encoding='utf-8'))
if len(questions) != 50:
    raise RuntimeError('Expected exactly 50 questions')
user = get_user_model().objects.get(username='corpus20_local')
# Keep runs after the figure-context change separate from the earlier baseline.
output = root / 'evaluation' / os.getenv('COLPALI_EVAL_OUTPUT', 'colpali50_results_after_visual_context.jsonl')
max_questions = int(os.getenv('COLPALI_EVAL_COUNT', '50'))
if not 1 <= max_questions <= len(questions):
    raise ValueError('Invalid question count')
previous = {}
if output.exists():
    for line in output.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        previous[row['number']] = row
with output.open('a', encoding='utf-8') as stream:
    for number, question in questions[:max_questions]:
        prior = previous.get(int(number), {})
        if prior.get('status') in {'generated', 'abstained'} and (prior.get('streams') or {}).get('colpali', {}).get('state') in {'ready','empty'}:
            continue
        started = time.monotonic()
        try:
            with (root / 'evaluation' / 'colpali50_model.log').open('a', encoding='utf-8') as model_log:
                with contextlib.redirect_stdout(model_log):
                    result = search_knowledge(user, question, generate=True)
            coverage = result.get('context_coverage') or {}
            sources = result.get('citations') or []
            record = dict(number=int(number), question=question, search_id=result.get('search_id'),
                status=result.get('generation_status'), answer=result.get('answer_text'),
                answer_quotes=result.get('answer_quotes', []), diagnostics=result.get('generation_diagnostics'),
                versions_scanned=coverage.get('versions_scanned'), streams=coverage.get('retrieval_streams'),
                reranker_state=coverage.get('reranker_state'), selection=coverage.get('source_selection'),
                generation_calls=result.get('generation_calls', []),
                sources=[{k:s.get(k) for k in ('evidence_id','original_name','page_number','layout_type',
                    'source_url','text','image_path','visual_block_id','colpali_point_id','colpali_page_rank',
                    'retrieval_streams','expansion_reason')} for s in sources])
        except Exception as error:
            record = dict(number=int(number), question=question, status='error',
                          error_type=type(error).__name__, error=str(error)[:500])
        record['elapsed_seconds'] = round(time.monotonic() - started, 2)
        stream.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
        stream.flush()
        print(f"COLPALI50 {number}/50 status={record['status']} elapsed={record['elapsed_seconds']}s", flush=True)
        if ((record.get('diagnostics') or {}).get('state') == 'gateway_unavailable'
                or ((record.get('streams') or {}).get('colpali') or {}).get('state') == 'unavailable'):
            print('COLPALI50 stopped: gateway unavailable; resume this command after restoring it.', flush=True)
            break

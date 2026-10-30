"""Run the 50 pilot questions once through the production flow and store the answers.

  docker exec -e RAG_INFERENCE_URL=http://host.docker.internal:8010 riskdl_web \
      python -X utf8 -m evaluation.runners.pilot_mainflow FROZEN OUT

Each question goes through search_knowledge(generate=True): planner, BM25/BGE-M3/ColPali retrieval,
RRF, reranker and graph expansion. The local GPU runtime keeps one model resident, so retrieval runs
one question at a time. The production answer step (answer_from_sources: evidence pods, checked
generation) then runs twice on the same retrieval, images on and images off, in parallel against the
remote generator. Production code is unchanged: images off replaces needs_source_images with False
for that call only. The per-user web quota is an anti-abuse limit, so receipts are created directly.
"""
import argparse
import concurrent.futures as cf
import json
import os
import threading
import time
from pathlib import Path

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
import django
django.setup()

from django.contrib.auth import get_user_model
from django.db import close_old_connections

from NewIdea.repositories import knowledge_repository
from NewIdea.services import knowledge_answer_service
from NewIdea.services.knowledge_search_service import search_knowledge

USER_ID = 1000000
local = threading.local()
write_lock = threading.Lock()
original_answer = knowledge_answer_service.answer_from_sources
original_images = knowledge_answer_service.needs_source_images
original_request = knowledge_answer_service.request_answer_with_context_retry


def needs_images(*args, **kwargs):
    wanted = original_images(*args, **kwargs)
    local.images_wanted = wanted
    return wanted if local.images == 'on' else False


def request(*args, **kwargs):
    response = original_request(*args, **kwargs)
    local.responses.append(response)
    return response


def capture(query, citations, coverage, progress=None):
    """Stand-in during retrieval: keep the exact arguments the production flow passes to generation."""
    local.captured = (query, citations, coverage)
    return {}


knowledge_answer_service.needs_source_images = needs_images
knowledge_answer_service.request_answer_with_context_retry = request


def retrieve(user, question):
    local.captured = None
    receipt = knowledge_repository.create_search(user, question['question'].strip(), None, True, 'knowledge')
    started = time.perf_counter()
    knowledge_answer_service.answer_from_sources = capture
    try:
        result = search_knowledge(user, question['question'], generate=True, existing_receipt=receipt)
    finally:
        knowledge_answer_service.answer_from_sources = original_answer
    return receipt.pk, result, local.captured, (time.perf_counter()-started)*1000


def generate(question, retrieval, images):
    close_old_connections()
    log_id, result, captured, retrieval_ms = retrieval
    local.images, local.responses, local.images_wanted = images, [], None
    started = time.perf_counter()
    answer, status = {}, 'completed'
    if captured is None:
        status = 'no_generation_call'  # production stopped before generation (e.g. retrieval unavailable)
        answer = {k: result.get(k) for k in ('answer', 'generation_status', 'generation_diagnostics')}
    else:
        try:
            answer = original_answer(*captured)
        except Exception as error:
            status, answer = 'failed', dict(error_type=type(error).__name__, error=str(error)[:500])
    claims = [c for response in local.responses for c in response.get('citation_claims', [])]
    usage = [o.get('usage') for o in answer.get('qwen_outputs', []) if o.get('usage')]
    coverage = result.get('context_coverage') or {}
    return dict(question_id=question['question_id'], question=question['question'], images=images,
        images_requested_by_policy=local.images_wanted, images_sent=bool(local.images_wanted) and images == 'on',
        status=status, search_log_id=log_id, retrieval_ms=retrieval_ms, generation_ms=(time.perf_counter()-started)*1000,
        generation_status=answer.get('generation_status'), generation_diagnostics=answer.get('generation_diagnostics'),
        answer=answer.get('answer'), answer_text=answer.get('answer_text'),
        claims=[dict(text=c.get('text', ''), evidence_ids=c.get('evidence_ids', []), part_ids=c.get('part_ids', []),
                     quotes=c.get('quotes', [])) for c in claims],
        answer_source_ids=answer.get('answer_source_ids', []), answer_notes=answer.get('answer_notes', []),
        citation_check=answer.get('citation_check'),
        sources=[dict(evidence_id=c.get('evidence_id'), version_id=c.get('version_id'), chunk_id=c.get('chunk_id'),
                      expansion_reason=c.get('expansion_reason'), layout_type=c.get('layout_type'),
                      content_hash=c.get('content_hash'), text=c.get('text', '')) for c in result.get('citations', [])],
        stage_timings_ms=result.get('stage_timings_ms'), query_planning=coverage.get('query_planning'),
        retrieval_streams=coverage.get('retrieval_streams'),
        prompt_tokens=sum(u.get('prompt_tokens') or 0 for u in usage) or None,
        completion_tokens=sum(u.get('completion_tokens') or 0 for u in usage) or None,
        generation_calls=len(answer.get('qwen_outputs', [])), qwen_errors=[o.get('outputs', {}).get('error')
            for o in answer.get('qwen_outputs', []) if isinstance(o.get('outputs'), dict) and o['outputs'].get('error')],
        error=answer.get('error'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('frozen')
    parser.add_argument('out')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    targets = {m: out/f'answers_images_{m}.jsonl' for m in ('on', 'off')}
    done = {m: {r['question_id'] for r in map(json.loads, t.open(encoding='utf-8')) if r['status'] == 'completed'
                and r['generation_status'] != 'retrieval_unavailable'
                and (r.get('generation_diagnostics') or {}).get('state') != 'gateway_unavailable'}
            if t.exists() else set() for m, t in targets.items()}
    questions = [json.loads(l) for l in (Path(args.frozen)/'gold_questions.jsonl').open(encoding='utf-8')]
    todo = [q for q in questions if any(q['question_id'] not in d for d in done.values())]
    user = get_user_model().objects.get(pk=USER_ID)
    print(f'{len(todo)} of {len(questions)} questions remain', flush=True)

    def finish(future, mode):
        row = future.result()
        with write_lock, targets[mode].open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
        print(row['question_id'], 'images', mode, row['status'], row['generation_status'],
              'sent' if row['images_sent'] else '', round(row['retrieval_ms']/1000), round(row['generation_ms']/1000), flush=True)

    with cf.ThreadPoolExecutor(4) as pool:
        for question in todo:
            retrieval = retrieve(user, question)  # sequential: one GPU model resident at a time
            for mode in ('on', 'off'):
                if question['question_id'] not in done[mode]:
                    future = pool.submit(generate, question, retrieval, mode)
                    future.add_done_callback(lambda f, m=mode: finish(f, m))
        pool.shutdown(wait=True)


if __name__ == '__main__':
    main()

"""Search quotas, evidence receipts and durable Qwen job submission."""
from django.conf import settings
from django.db import transaction
from django.utils import timezone
from NewIdea.repositories import knowledge_repository, contract_repository
from .evidence_service import digest
from .search_evidence_service import visible_text_records
from .errors import KnowledgeError
import time


def update_progress(receipt, stage, **details):
    """Persist a small phase marker; final results include advisory review notes."""
    if not hasattr(receipt, '_progress_started'):
        receipt._progress_started = time.perf_counter()
    receipt.result['progress'] = dict(
        stage=stage, elapsed_ms=round((time.perf_counter() - receipt._progress_started) * 1000),
        updated_at=timezone.now().isoformat(), **details,
    )
    knowledge_repository.save_search(receipt, 'result')

def access_snapshot(user):
    return dict(company_id=user.company_id, is_superuser=user.is_superuser)

def reserve_search(user, query, version=None, generated=False, scope='version'):
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 2000:
        raise ValueError('Câu hỏi cần từ 1 đến 2000 ký tự.')
    # PostgreSQL row lock makes concurrent requests share the same quota.
    with transaction.atomic():
        knowledge_repository.lock_user(user)
        recent, generated_recent, pending = knowledge_repository.search_quota(user)
        if recent >= 30 or (generated and (generated_recent >= 6 or pending >= 3)):
            return None
        return knowledge_repository.create_search(user, query.strip(), version, generated, scope)



def finish_search(log, data):
    citations = data.get('citations', data.get('results', []))
    log.result = dict(progress=log.result.get('progress', {}), access=access_snapshot(log.user), answer=data.get('answer', ''), generation_status=data.get('generation_status'),
        quality_gate=data.get('quality_gate'), coverage=data.get('context_coverage'),
        qwen_outputs=data.get('qwen_outputs', []),
        answer_text=data.get('answer_text'), answer_source_ids=data.get('answer_source_ids', []),answer_quotes=data.get('answer_quotes', []),answer_notes=data.get('answer_notes', []),
        citation_check=data.get('citation_check'), generation_diagnostics=data.get('generation_diagnostics'),
        generation_calls=data.get('retrieval_trace', {}).get('generation_calls', data.get('generation_calls', [])),
        sources=[dict(c, sha256=digest(c.get('text', ''))) for c in citations],
        model=getattr(settings, 'QWEN_MODEL_ID', 'configured-Qwen') if log.generated else None,
        prompt_policy='contract-evidence-v1', embedding_model='BAAI/bge-m3', reranker_model='BAAI/bge-reranker-v2-m3',
        retrieval_policy='equal-baseline-v2', recorded_at=timezone.now().isoformat(),
        model_revision_verified=False)
    log.status = 'FINALIZING' if getattr(log, '_defer_completion', False) else 'COMPLETED'
    knowledge_repository.save_search(log, 'result', 'status')




def read_job(user, search_id):
    from .case_rules import can_review
    receipt = knowledge_repository.get_visible_search(user, search_id)
    if receipt.user_id != user.pk and not can_review(user):
        raise KnowledgeError('Không có quyền xem lịch sử này.', 403)
    return dict(success=True, search_id=receipt.pk, status=receipt.status,
                result=receipt.result if receipt.status != 'QUEUED' else {})


def enqueue_search(user, data):
    if data.get('scope') in {'knowledge', 'company'}:
        # Accept the old client value while giving new jobs an explicit corpus scope.
        receipt = reserve_search(user, data.get('query'), generated=True, scope='knowledge')
        if receipt is None:
            raise KnowledgeError('Đã đạt giới hạn; thử lại sau một phút.', 429)
        receipt.status = 'QUEUED'
        receipt.result = {'access': access_snapshot(user), 'request': dict(query=receipt.query, generate=True, scope='knowledge')}
        knowledge_repository.save_search(receipt, 'status', 'result')
        return dict(success=True, search_id=receipt.pk, status='QUEUED')
    version = contract_repository.get_visible_version(user, data.get('version_id'))
    mode = data.get('mode', 'structural')
    if mode not in {'structural', 'whole_contract'}:
        raise KnowledgeError('Phạm vi không hợp lệ.')
    receipt = reserve_search(user, data.get('query'), version, generated=True)
    if receipt is None:
        raise KnowledgeError('Đã đạt giới hạn; thử lại sau một phút.', 429)
    receipt.status = 'QUEUED'
    receipt.result = {'request': dict(query=receipt.query, version_id=version.pk, generate=True, mode=mode)}
    knowledge_repository.save_search(receipt, 'status', 'result')
    return dict(success=True, search_id=receipt.pk, status='QUEUED')


def available_sources(user, version_id):
    version = contract_repository.get_visible_version(user, version_id)
    return dict(success=True, sources=[dict(chunk_id=r['chunk_id'], text=r['text'][:180],
        source_location=r['source_location']) for r in contract_repository.version_sources(version)])


def process_next_job(search):
    with transaction.atomic():
        receipt = knowledge_repository.next_queued_search()
        if not receipt:
            return False
        receipt.status = 'RUNNING'
        knowledge_repository.save_search(receipt, 'status')
    try:
        data = receipt.result['request']
        if not receipt.user.is_active:
            raise RuntimeError('Requester account disabled')
        receipt._defer_completion = True
        if data.get('scope') == 'knowledge_index':
            from ai_extract.services.contract_pipeline_service import PlanACollectionIndexer
            records = [r for r in visible_text_records(receipt.user)
                       if r['knowledge_key'] == data['knowledge_key']]
            indexed = PlanACollectionIndexer().index_text_records(records)
            response = dict(success=indexed.status == 'COMPLETED', index_status=indexed.status,
                            points=indexed.indexed_count, error=indexed.error)
        elif data.get('scope') == 'expert':
            from .expert_submission_service import process
            response = process(receipt)
        elif data.get('scope') in {'knowledge', 'company'}:
            if data.get('scope') == 'knowledge':
                if receipt.result.get('access') != access_snapshot(receipt.user):
                    raise RuntimeError('Requester access changed')
            elif receipt.user.company_id != data.get('company_id'):
                raise RuntimeError('Requester company changed')
            from .knowledge_search_service import search_knowledge
            response = search_knowledge(receipt.user, data['query'], generate=True, existing_receipt=receipt)
        else:
            response = search(receipt.user, receipt.version.contract_id, data, existing_receipt=receipt)
        receipt.result['response'] = response
        receipt.status = 'COMPLETED'
        receipt.result['progress'] = dict(receipt.result.get('progress', {}), stage='completed')
        knowledge_repository.save_search(receipt, 'result', 'status')
    except Exception:
        receipt.status = 'FAILED'
        receipt.result['error'] = 'Không hoàn tất xử lý. Kiểm tra A3, Qwen và quyền truy cập; tài liệu đã tải vẫn được giữ.'
        knowledge_repository.save_search(receipt, 'status', 'result')
    return True


def recover_stale_jobs():
    return knowledge_repository.fail_stale_searches()

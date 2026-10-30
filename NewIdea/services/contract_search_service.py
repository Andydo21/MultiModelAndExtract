"""Retrieve and answer against one authorized contract version using A3."""
from ai_extract.services.document_structure_service import structure_for_version
import time
from shared.retrieval import source_excerpt
from ai_extract.services.contract_pipeline_service import retrieve_hybrid, review_contract
from NewIdea.repositories import contract_repository, knowledge_repository
from .search_job_service import reserve_search, finish_search, update_progress
from .errors import KnowledgeError
from .search_evidence_service import with_expert_evidence, matched_structure_nodes, EXPERT_INSTRUCTION, find_expert_cases
from .knowledge_answer_service import generate_checked_answer, needs_source_images


def search_contract(user, contract_id, data, existing_receipt=None):
    contract = contract_repository.get_visible_contract(user, contract_id)
    started = time.perf_counter()
    receipt = None
    try:
        if not isinstance(data.get('query'), str) or not data['query'].strip():
            raise ValueError('Cần câu hỏi dạng chuỗi.')
        query = data['query'].strip()
        if data.get('mode') == 'whole_contract' and data.get('generate') is not True:
            raise ValueError('Rà soát toàn bộ cần bật tổng hợp Qwen; tra cứu thường không đọc hết nguồn.')
        version = contract_repository.contract_version(contract, data.get('version_id'))
        if version is None:
            raise ValueError('Hợp đồng chưa có phiên bản.')
        receipt = existing_receipt
        if receipt is not None:
            if receipt.user_id != user.pk or receipt.version_id != version.pk or receipt.query != query:
                raise ValueError('Search job scope mismatch.')
        else:
            receipt = reserve_search(user, query, version, data.get('generate') is True)
        if receipt is None:
            raise KnowledgeError('Đã đạt giới hạn tra cứu; thử lại sau một phút.', 429)
        update_progress(receipt, 'retrieving', documents_total=1)
        tree = structure_for_version(version, contract_repository.version_sources(version), contract.title)
        questions=[]
        planning={}
        if data.get('generate') is True:
            from ai_extract.services.contract_pipeline_service import plan_retrieval_questions
            planning=plan_retrieval_questions(query)
            questions=planning['queries']
        citations, coverage = (retrieve_hybrid(query,tree,version,queries=questions) if questions
                               else retrieve_hybrid(query,tree,version))
        if planning:
            coverage['query_planning']=planning
            coverage['answer_parts']=planning.get('answer_parts',[])
            coverage['requested_parts']=[p['question'] for p in coverage['answer_parts']] or questions
        for index, citation in enumerate(citations, 1):
            citation['text_is_excerpt'] = False
            citation['evidence_id'] = f'E{index}'
        whole_document = data.get('mode') == 'whole_contract' or planning.get('intent',{}).get('scope') == 'whole_document'
        case_search = dict(state='ready')
        try:
            similar = find_expert_cases(query, contract_repository.approved_related_cases(user, contract))
        except Exception as error:
            similar = []
            case_search = dict(state='unavailable', error_type=type(error).__name__)
        matched_nodes = matched_structure_nodes(tree, citations, version, contract_repository.completed_extraction_run(version))
        if not whole_document:
            citations = with_expert_evidence(citations, similar)
        quality = dict(accepted=bool(citations), scope='source_availability_not_semantic_verification')
        retrieval_ms = round((time.perf_counter() - started) * 1000, 2)
        answer = source_excerpt(citations) if citations else 'Chưa tìm thấy bằng chứng trong phiên bản hợp đồng này. Hãy chạy trích xuất tài liệu hoặc bổ sung câu hỏi.'
        generation_status = 'source_excerpt' if citations else 'abstained'
        generation_calls = []
        answer_presentation = {}
        qwen_outputs = []
        citation_check = {'valid': False, 'scope': 'identifier_validation_only'}
        generation_diagnostics = {'state': 'insufficient_evidence' if data.get('generate') is True else 'not_requested'}
        # Generation is opt-in so missing gateway config cannot trigger a legacy external call.
        if tree['leaves'] and data.get('generate') is True and (citations or whole_document):
            generation_diagnostics = {'state': 'requested'}
            from ai_extract.services.contract_pipeline_service import request_qwen, QwenGatewayError
            try:
                if whole_document:
                    update_progress(receipt, 'reviewing_document')
                    reviewed, generation_status, citation_check, citations, coverage = review_contract(query, tree, request_qwen)
                    answer = reviewed or source_excerpt(citations)
                    generation_calls = coverage.get('generation_calls', [])
                    qwen_outputs = coverage.get('qwen_outputs', [])
                    generation_diagnostics = {'state': generation_status}
                else:
                    payload = {'query': query, 'evidence': citations,
                        'outline': [{'kind': n['kind'], 'title': n['title'], 'parent_id': n['parent_id']}
                                    for n in tree['nodes'][:40]], 'coverage': coverage,
                        'expert_policy': EXPERT_INSTRUCTION if any(c.get('evidence_kind')=='expert_opinion' for c in citations) else '',
                        'use_images': needs_source_images(planning,citations)}
                    generated = generate_checked_answer(payload, request_model=request_qwen,
                        progress=lambda stage: update_progress(receipt, stage))
                    answer = generated['answer']
                    answer_presentation = {key:generated[key] for key in ('answer_text','answer_source_ids','answer_quotes','answer_notes') if key in generated}
                    generation_status = generated['generation_status']
                    citation_check = generated['citation_check']
                    generation_calls = generated['generation_calls']
                    qwen_outputs = generated.get('qwen_outputs', [])
                    generation_diagnostics = generated['generation_diagnostics']
            except (QwenGatewayError, KeyError, TypeError) as error:
                generation_diagnostics = {'state': 'gateway_unavailable', 'error_type': type(error).__name__}
        from ai_extract.services.graph_visualization_service import tree_payload
        result = {'structure_visualization': tree_payload(tree, contract.pk, version.pk),
            'retrieval_trace': dict(streams=coverage.get('retrieval_streams', {}), selected=[dict(evidence_id=c['evidence_id'], chunk_id=c['chunk_id'], source_id=c.get('source_chunk_id'), reason=c.get('expansion_reason', 'review_batch'), heading_path=c.get('heading_path', [])) for c in citations], generation_calls=generation_calls, generation_status=generation_status),
            'success': True, 'answer': answer, 'citations': citations,
            **answer_presentation,
            'context_coverage': coverage,
            'quality_gate': quality,
            'structure_outline': [dict(id=n['id'], parent_id=n['parent_id'], kind=n['kind'], title=n['title']) for n in tree['nodes']],
            'generation_status': generation_status, 'citation_check': citation_check,
            'generation_diagnostics': generation_diagnostics,
            'qwen_outputs': qwen_outputs,
            'matched_nodes': matched_nodes, 'similar_cases': similar[:3], 'expert_case_search': case_search, 'contract_id': contract.pk, 'version_id': version.pk,
            'retrieval_latency_ms': retrieval_ms, 'latency_ms': round((time.perf_counter() - started) * 1000, 2),
            'score_kind': 'retrieval_ranking_not_accuracy', 'search_id': receipt.pk,
            'expert_handoff_recommended': not quality['accepted'],
            'evidence_sources': [dict(chunk_id=r['chunk_id'], source_location=r['source_location'], text=r['text'][:180])
                                 for r in contract_repository.version_sources(version)]}
        finish_search(receipt, result)
        return result
    finally:
        if receipt and receipt.status == 'RUNNING':
            receipt.status = 'FAILED'
            knowledge_repository.save_search(receipt, 'status')


def workspace_context(user, contract_id=None, version_id=None):
    from .case_rules import can_review
    contracts = contract_repository.workspace_contracts(user)
    selected = contract_repository.get_visible_contract(user, contract_id) if contract_id else None
    version = contract_repository.contract_version(selected) if selected else None
    if selected and version_id:
        if not str(version_id).isdecimal():
            raise KnowledgeError('Version ID không hợp lệ.')
        version = contract_repository.contract_version(selected, version_id)
    return dict(available_contracts=contracts, selected_contract=selected, selected_version=version,
        can_review=can_review(user))

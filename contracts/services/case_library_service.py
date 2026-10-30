from contracts.repositories import case_library_repository as workflow_repository
import json
from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import Q
from django.urls import reverse
from django.utils import timezone
from contracts.models import ContractVersion, ContractKnowledgeCase
from NewIdea.services.case_rules import CASE_FIELDS, approval_gaps, can_review, serialize_case
from NewIdea.repositories.contract_repository import visible_contracts, case_queryset
from NewIdea.services.evidence_service import bind_evidence
from contracts.services.results import ServiceResult
from contracts.repositories.knowledge_data_repository import get_record

def workspace(contract_id, user, query_params, *, newidea=False):
    contracts = visible_contracts(user).order_by('-updated_at')
    selected = get_record(contracts, pk=contract_id) if contract_id else None
    version = selected.latest_version if selected else None
    if selected and query_params.get('version_id'):
        if not query_params['version_id'].isdecimal():
            return ServiceResult({'success': False, 'error': 'Version ID không hợp lệ.'}, status=400)
        version = get_record(selected.versions, pk=query_params['version_id'])
    return ServiceResult({'available_contracts': contracts, 'selected_contract': selected, 'selected_version': version, 'can_review': can_review(user), 'newidea_mode': newidea, 'contract_workspace_url_name': 'newidea_contract' if newidea else 'contract_intelligence_detail', 'search_endpoint': reverse('newidea_search' if newidea else 'contract_intelligence_search', args=[selected.pk]) if selected else ''}, template='contracts/intelligence.html')

def manage_cases(user, method, data, query_params):
    if not user.is_authenticated:
        return ServiceResult({'success': False, 'error': 'Authentication required.'}, status=401)
    cases = case_queryset(user)
    if method == 'GET':
        if not can_review(user):
            cases = cases.filter(Q(status='approved') | Q(author=user))
        query = query_params.get('q', '').strip()
        if query:
            cases = cases.filter(status='approved')
        contract_type = query_params.get('contract_type', '').strip()
        if contract_type:
            cases = cases.filter(version__contract__contract_type=contract_type)
        if query:
            from NewIdea.services.search_evidence_service import find_expert_cases
            try:
                results = find_expert_cases(query, cases, limit=100)
            except Exception:
                return ServiceResult({'success': False, 'error': 'Dịch vụ xếp hạng AI chưa khả dụng; hãy thử lại sau.'}, status=503)
        else:
            results = [dict(serialize_case(case), requires_applicability_review=True) for case in cases[:100]]
        return ServiceResult({'success': True, 'cases': results[:100]})
    try:
        data = data
        if not isinstance(data, dict):
            raise ValueError('JSON phải là object.')
        action = data.get('action', 'create')
        if action not in {'create', 'revise', 'approve', 'deprecate'}:
            raise ValueError('Thao tác không hợp lệ.')
        with transaction.atomic():
            if action == 'create':
                version = get_record(workflow_repository.filter_contract_version_for_knowledge_api(user, visible_contracts), pk=data.get('version_id'))
                values = {f: data.get(f, '') for f in CASE_FIELDS}
                case = ContractKnowledgeCase(version=version, author=user)
            else:
                # Only lock the case. Reviewer is nullable, and PostgreSQL cannot
                # lock the nullable side of the select_related outer join.
                case = get_record(cases.select_for_update(of=('self',)), pk=data.get('id'))
                if action in {'approve', 'deprecate'} and (not can_review(user)):
                    return ServiceResult({'success': False, 'error': 'Cần vai trò MANAGER hoặc ADMIN.'}, status=403)
                if action == 'revise':
                    if case.author_id != user.id and (not can_review(user)):
                        return ServiceResult({'success': False, 'error': 'Không có quyền sửa case.'}, status=403)
                    latest = workflow_repository.first_contract_knowledge_case_for_knowledge_api(case)
                    values = {f: data.get(f, getattr(case, f)) for f in CASE_FIELDS}
                    case = ContractKnowledgeCase(case_id=case.case_id, revision=latest.revision + 1, version=case.version, author=user)
                elif action == 'approve':
                    if case.author_id == user.pk:
                        return ServiceResult({'success': False, 'error': 'Reviewer phải khác người tạo revision.'}, status=403)
                    if case.status != 'draft' or approval_gaps(case):
                        raise ValueError('Case nháp cần vấn đề, cách xử lý, xác minh, điều kiện áp dụng và nguồn để duyệt.')
                    siblings = workflow_repository.filter_contract_knowledge_case_for_knowledge_api(case)
                    if siblings.filter(revision__gt=case.revision).exists():
                        raise ValueError('Chỉ duyệt revision mới nhất.')
                    for old in siblings.filter(status='approved'):
                        old.status = 'deprecated'
                        workflow_repository.save_record_for_knowledge_api(old)
                        workflow_repository.create_audit_log_for_knowledge_api(json, old, user)
                    case.status = 'approved'
                    case.reviewer = user
                    case.approved_at = timezone.now()
                    case.evidence_snapshot = bind_evidence(case)
                else:
                    if case.status != 'approved':
                        raise ValueError('Chỉ ngừng sử dụng case đã duyệt.')
                    case.status = 'deprecated'
            if action in {'create', 'revise'}:
                if any((not isinstance(v, str) for v in values.values())):
                    raise ValueError('Các trường tri thức phải là chuỗi.')
                for f, value in values.items():
                    setattr(case, f, value.strip())
                if not case.title or not case.issue:
                    raise ValueError('Cần tiêu đề và vấn đề của điều khoản.')
            case.full_clean()
            workflow_repository.save_case_status(case)
            if action == 'approve':
                from contracts.models import ContractKnowledgeInquiry
                workflow_repository.update_contract_knowledge_inquiry_for_knowledge_api(case, timezone)
            workflow_repository.create_audit_log_for_knowledge_api_by_user_and_contract_and_action_and_payload(action, case, json, user)
            if action == 'approve':
                from NewIdea.repositories.knowledge_repository import enqueue_case_index
                enqueue_case_index(case, user)
        return ServiceResult({'success': True, 'case': serialize_case(case)})
    except (ValueError, TypeError, ValidationError) as exc:
        return ServiceResult({'success': False, 'error': str(exc)}, status=400)

def draft_expert_case(user, data):
    if not user.is_authenticated:
        return ServiceResult({'success': False, 'error': 'Authentication required.'}, status=401)
    try:
        data = data
        version = get_record(ContractVersion, pk=data.get('version_id'), contract__in=visible_contracts(user))
        note = data.get('note', '')
        if not isinstance(note, str) or not 20 <= len(note) <= 12000:
            raise ValueError('Ghi chú chuyên gia cần từ 20 đến 12000 ký tự.')
        from ai_extract.services.contract_pipeline_service import request_qwen, QwenGatewayError
        try:
            proposal = request_qwen('expert_case_draft', {'note': note, 'fields': CASE_FIELDS})
        except QwenGatewayError:
            return ServiceResult({'success': False, 'error': 'Qwen chưa khả dụng; hãy nhập case thủ công.'}, status=503)
        if not isinstance(proposal, dict):
            raise ValueError('Qwen không trả được bản nháp có cấu trúc.')
        draft = {f: str(proposal.get(f, '')).strip() for f in CASE_FIELDS}
        return ServiceResult({'success': True, 'draft': draft, 'version_id': version.pk, 'requires_expert_confirmation': True, 'saved': False})
    except (ValueError, TypeError) as error:
        return ServiceResult({'success': False, 'error': str(error)}, status=400)

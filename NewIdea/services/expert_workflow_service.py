"""Expert assignment, evidence-bound answers and inquiry state transitions."""
import json
from django.db import transaction
from .case_rules import can_review, serialize_case
from NewIdea.repositories import knowledge_repository
from .case_rules import CASE_FIELDS
from .evidence_service import bind_evidence
from .errors import KnowledgeError


def serialize_inquiry(item):
    return dict(id=item.pk, question=item.question, status=item.status, version_id=item.version_id,
        contract_id=item.version.contract_id, requester=item.requester.username,
        assignee=item.assignee.username if item.assignee else None,
        case_id=item.case_id, search_id=item.search_id)


def list_inquiries(user):
    items = knowledge_repository.inquiry_list(user, own_only=not can_review(user))
    reviewers = knowledge_repository.reviewer_list(user.company_id)
    return dict(success=True, inquiries=[serialize_inquiry(i) for i in items],
                reviewers=reviewers)


def inquiry_detail(user, inquiry_id):
    if not str(inquiry_id).isdecimal():
        raise KnowledgeError('ID yêu cầu không hợp lệ.')
    item = knowledge_repository.get_visible_inquiry(user, inquiry_id, own_only=not can_review(user))
    return dict(success=True, inquiry=serialize_inquiry(item),
                case=serialize_case(item.case) if item.case else None, search=item.search.result)


def update_inquiry(user, data):
    action = data.get('action', 'create')
    with transaction.atomic():
        if action == 'create':
            receipt = knowledge_repository.get_visible_search(user, data.get('search_id'), completed_own=True)
            if receipt.version_id is None:
                raise KnowledgeError('Chọn một tài liệu liên quan để gửi yêu cầu chuyên gia.')
            item, created = knowledge_repository.create_inquiry(receipt, user)
            if not created:
                return dict(success=True, inquiry=serialize_inquiry(item))
        else:
            item = knowledge_repository.get_visible_inquiry(user, data.get('id'), lock=True)
            if action == 'cancel':
                if item.requester_id != user.pk or item.status not in {'OPEN', 'ASSIGNED'}:
                    raise KnowledgeError('Không thể hủy yêu cầu này.', 403)
                item.status = 'CANCELLED'
            elif action == 'assign':
                if not can_review(user):
                    raise KnowledgeError('Cần quản lý phân công.', 403)
                if item.status not in {'OPEN', 'ASSIGNED'}:
                    raise KnowledgeError('Yêu cầu không còn chờ trả lời.')
                assignee = knowledge_repository.get_active_assignee(item.version.contract.company_id, data.get('assignee_id'))
                if not can_review(assignee):
                    raise KnowledgeError('Người nhận cần vai trò MANAGER hoặc ADMIN.')
                item.assignee, item.status = assignee, 'ASSIGNED'
            elif action == 'answer':
                if item.assignee_id != user.pk or item.status not in {'ASSIGNED', 'ANSWERED'}:
                    raise KnowledgeError('Chỉ chuyên gia được phân công có thể trả lời.', 403)
                if item.case and item.case.status != 'draft':
                    raise KnowledgeError('Case đã duyệt; không ghi đè.')
                values = {field: data.get(field, '') for field in CASE_FIELDS}
                if any(not isinstance(value, str) for value in values.values()):
                    raise KnowledgeError('Các trường case phải là chuỗi.')
                case = item.case or knowledge_repository.new_case(item.version, user)
                for field, value in values.items():
                    setattr(case, field, value.strip())
                case.evidence_snapshot = bind_evidence(case)
                case.full_clean()
                knowledge_repository.save_case(case)
                item.case, item.status = case, 'ANSWERED'
            else:
                raise KnowledgeError('Thao tác không hợp lệ.')
            knowledge_repository.save_inquiry(item)
        knowledge_repository.record_inquiry_action(user, item, action,
            json.dumps({'inquiry_id': item.pk, 'case_id': item.case_id}))
    result = dict(success=True, inquiry=serialize_inquiry(item))
    if item.case:
        result['case'] = serialize_case(item.case)
    return result

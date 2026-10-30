"""Database access for search receipts, expert inquiries, cases and audit records."""
from datetime import timedelta
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.db.models import Q
from contracts.models import ContractKnowledgeCase, ContractKnowledgeInquiry, KnowledgeSearchLog, AuditLog


def lock_user(user):
    return get_user_model().objects.select_for_update().get(pk=user.pk)


def search_quota(user):
    recent = KnowledgeSearchLog.objects.filter(user=user, created_at__gte=timezone.now()-timedelta(minutes=1))
    pending = KnowledgeSearchLog.objects.filter(user=user, generated=True, status__in=['QUEUED', 'RUNNING', 'FINALIZING']).count()
    return recent.count(), recent.filter(generated=True).count(), pending


def create_search(user, query, version, generated, scope):
    return KnowledgeSearchLog.objects.create(user=user, version=version, query=query, generated=generated, scope=scope)


def save_search(log, *fields):
    log.save(update_fields=list(fields))


def get_visible_search(user, search_id, completed_own=False):
    from .contract_repository import visible_contracts
    company_scope = Q(user=user, scope='company', version__isnull=True) & (
        Q(result__coverage__company_id=user.company_id) | Q(result__request__company_id=user.company_id))
    company_scope |= Q(user=user, scope='expert', result__request__company_id=user.company_id)
    if not user.company_id:
        company_scope = Q(pk__in=[])
    knowledge_scope = Q(user=user, scope='knowledge', version__isnull=True,
        result__access__company_id=user.company_id, result__access__is_superuser=user.is_superuser)
    records = KnowledgeSearchLog.objects.filter(Q(version__contract__in=visible_contracts(user)) | company_scope | knowledge_scope)
    if completed_own:
        records = records.filter(user=user, status='COMPLETED')
    return records.get(pk=search_id)


def visible_inquiries(user, own_only=False):
    from .contract_repository import visible_contracts
    records = ContractKnowledgeInquiry.objects.filter(version__contract__in=visible_contracts(user)).select_related('version__contract', 'requester', 'assignee', 'case', 'search')
    return records.filter(requester=user) if own_only else records


def get_visible_inquiry(user, inquiry_id, own_only=False, lock=False):
    records = visible_inquiries(user, own_only)
    if lock:
        records = records.select_for_update(of=('self',))
    return records.get(pk=inquiry_id)


def active_reviewers(company_id):
    return get_user_model().objects.filter(is_active=True, company_id=company_id, role__role_name__in=['MANAGER', 'ADMIN']) if company_id else get_user_model().objects.none()


def get_active_assignee(company_id, user_id):
    return get_user_model().objects.get(pk=user_id, is_active=True, company_id=company_id)


def create_inquiry(receipt, user):
    return ContractKnowledgeInquiry.objects.get_or_create(search=receipt,
        defaults=dict(version=receipt.version, requester=user, question=receipt.query))


def new_case(version, author):
    return ContractKnowledgeCase(version=version, author=author)


def save_case(case):
    case.save()


def enqueue_case_index(case, user):
    """Index after commit in the existing worker, keeping HTTP writes fast."""
    from django.db import transaction
    if case.status == 'approved':
        transaction.on_commit(lambda: KnowledgeSearchLog.objects.create(user=user, scope='knowledge_index',
            query='Index expert case', status='QUEUED', generated=False,
            result=dict(request=dict(scope='knowledge_index', knowledge_key=f'A2:{case.case_id}:r{case.revision}'))))


def save_inquiry(inquiry):
    inquiry.save()


def record_inquiry_action(user, inquiry, action, payload):
    AuditLog.objects.create(user=user, contract=inquiry.version.contract,
        action='A2_INQUIRY_'+action.upper(), payload=payload)


def inquiry_list(user, own_only=False):
    return list(visible_inquiries(user, own_only).order_by('-updated_at')[:100])


def reviewer_list(company_id):
    return list(active_reviewers(company_id).values('id', 'username'))


def next_queued_search():
    from django.db import connection
    jobs = KnowledgeSearchLog.objects.filter(status='QUEUED').order_by('created_at')
    jobs = jobs.select_for_update(skip_locked=True) if connection.features.has_select_for_update_skip_locked else jobs.select_for_update()
    return jobs.first()


def fail_stale_searches():
    return KnowledgeSearchLog.objects.filter(status__in=['RUNNING','FINALIZING'], generated=True,
        created_at__lt=timezone.now()-timedelta(hours=2)).update(status='FAILED')

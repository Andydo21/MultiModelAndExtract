"""Database operations for case library."""
from contracts.models import Contract, ContractVersion, ContractKnowledgeCase, AuditLog
from contracts.models import ContractKnowledgeInquiry

def filter_contract_version_for_knowledge_api(user, visible_contracts):
    return ContractVersion.objects.filter(contract__in=visible_contracts(user))


def first_contract_knowledge_case_for_knowledge_api(case):
    return ContractKnowledgeCase.objects.select_for_update().filter(case_id=case.case_id).order_by('-revision').first()


def filter_contract_knowledge_case_for_knowledge_api(case):
    return ContractKnowledgeCase.objects.select_for_update().filter(case_id=case.case_id)


def save_record_for_knowledge_api(old):
    return old.save(update_fields=['status', 'updated_at'])


def create_audit_log_for_knowledge_api(json, old, user):
    return AuditLog.objects.create(user=user, contract=old.version.contract, action='A2_CASE_SUPERSEDED', payload=json.dumps({'case_id': str(old.case_id), 'revision': old.revision}))


def save_case_status(case):
    return case.save()


def update_contract_knowledge_inquiry_for_knowledge_api(case, timezone):
    return ContractKnowledgeInquiry.objects.filter(case__case_id=case.case_id, status='ANSWERED').update(status='RESOLVED', case=case, updated_at=timezone.now())


def create_audit_log_for_knowledge_api_by_user_and_contract_and_action_and_payload(action, case, json, user):
    return AuditLog.objects.create(user=user, contract=case.version.contract, action=f'A2_CASE_{action.upper()}', payload=json.dumps({'case_id': str(case.case_id), 'revision': case.revision, 'status': case.status}))


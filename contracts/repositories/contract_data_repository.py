"""Database operations for contract workflows."""
from contracts.models import ContractVersion
from contracts.models import ContractContext
from ai_extract.models import ContractSummary
from ai_extract.models import Clause
from ai_extract.models import ExtractedEntity
from contracts.models import User

def first_ai_analyses_for_list_all_contracts(c):
    return c.ai_analyses.order_by('-created_at').first()


def get_versions_for_get_contract_details(c, version_id):
    return c.versions.get(id=version_id)


def create_contract_version_for_get_contract_details(c):
    return ContractVersion.objects.create(contract=c, version_number=1)


def first_files_for_get_contract_details(version):
    return version.files.first()


def first_ai_analyses_for_get_contract_details(version):
    return version.ai_analyses.order_by('-created_at').first()


def all_ai_extract_clauses_for_get_contract_details(version):
    return version.ai_extract_clauses.all()


def all_ai_extract_entities_for_get_contract_details(cl):
    return cl.ai_extract_entities.all()


def all_ai_analyses_for_get_contract_details(version):
    return version.ai_analyses.all()


def all_reviews_for_get_contract_details(analysis):
    return analysis.reviews.all()


def all_findings_for_get_contract_details(latest_analysis):
    return latest_analysis.findings.all()


def order_by_contexts_for_get_contract_details(version):
    return version.contexts.filter(context_type='raw_text').order_by('id')


def order_by_versions_for_get_contract_details(c):
    return c.versions.all().order_by('-version_number')


def latest_analysis_for_listed_version(v):
    return v.ai_analyses.first()


def first_contract_summary_for_get_contract_details(version):
    return ContractSummary.objects.filter(version=version).first()


def create_contract_version_for_create_and_analyze_contract(contract):
    return ContractVersion.objects.create(contract=contract, version_number=1, change_summary='Initial version')


def create_contract_version_for_create_new_version(change_summary, contract, next_num):
    return ContractVersion.objects.create(contract=contract, version_number=next_num, change_summary=change_summary)


def save_contract_for_create_new_version(contract):
    return contract.save()


def get_versions_for_analyze_contract(contract, version_id):
    return contract.versions.get(id=version_id)


def delete_ai_analyses_for_analyze_contract(version):
    return version.ai_analyses.all().delete()


def delete_clause_for_manual_extract_contract(version):
    return Clause.objects.filter(version=version).delete()


def filter_clause_for_manual_extract_contract(version):
    return Clause.objects.filter(version=version)


def first_user_for_submit_expert_review():
    return User.objects.filter(is_superuser=True).first()


def exists_ai_extract_clauses_for_run_ai_analysis_via_api(version):
    return version.ai_extract_clauses.exists()


def exists_ai_extract_entities_for_run_ai_analysis_via_api(cl):
    return cl.ai_extract_entities.exists()


def filter_extracted_entity_for_run_ai_analysis_via_api(clauses):
    return ExtractedEntity.objects.filter(clause__in=clauses)


def get_or_create_extracted_entity_for_run_ai_analysis_via_api(Decimal, cl, entity):
    return ExtractedEntity.objects.get_or_create(clause=cl, entity_type=entity.get('entity_type'), entity_value=entity.get('entity_value'), defaults={'normalized_value': entity.get('normalized_value'), 'confidence_score': Decimal(str(entity.get('confidence_score', 1.0)))})


def get_or_create_extracted_entity_for_extract_and_save_basic_entities(Decimal, clause, p):
    return ExtractedEntity.objects.get_or_create(clause=clause, entity_type='PARTY', entity_value=p, defaults={'normalized_value': p.upper(), 'confidence_score': Decimal('0.95')})


def get_or_create_extracted_entity_for_extract_and_save_basic_entities_by_clause_and_entity_type_and_entity_value(Decimal, act_norm, act_val, clause):
    return ExtractedEntity.objects.get_or_create(clause=clause, entity_type='ACTION', entity_value=act_val, defaults={'normalized_value': act_norm, 'confidence_score': Decimal('0.90')})

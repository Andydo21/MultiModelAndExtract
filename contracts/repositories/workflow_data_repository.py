"""Database operations for workflow workflows."""
from contracts.models import ContractVersion, ContractContext
from contracts.models import ContractVersion
from contracts.models import User

def get_versions_for_push_to_workflow(contract, version_id):
    return contract.versions.get(id=version_id)


def order_by_contexts_for_push_to_workflow(version):
    return version.contexts.filter(context_type='raw_text').order_by('id')


def first_files_for_push_to_workflow(version):
    return version.files.first()


def all_ai_extract_clauses_for_push_to_workflow(version):
    return version.ai_extract_clauses.all()


def distinct_ai_extract_clauses_for_push_to_workflow(version):
    return version.ai_extract_clauses.values_list('clause_type', flat=True).exclude(clause_type__isnull=True).distinct()


def first_user_for_push_to_workflow():
    return User.objects.filter(is_superuser=True).first()

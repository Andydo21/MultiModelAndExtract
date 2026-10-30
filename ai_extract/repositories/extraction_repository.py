"""Contract-version selection and clause access for extraction workflows."""
from contracts.models import Contract, ContractVersion
from ai_extract.models import Clause
from NewIdea.repositories.contract_repository import visible_contracts


def contract_for(user, contract_id):
    return visible_contracts(user).prefetch_related('versions__ai_extract_clauses').get(pk=contract_id)


def version_for(contract, version_id=None):
    return contract.versions.get(pk=version_id) if version_id else contract.latest_version


def clause_for(user, clause_id):
    return Clause.objects.select_related('version__contract').get(pk=clause_id, version__contract__in=visible_contracts(user))


def clauses_for_version(version):
    return list(Clause.objects.filter(version=version))


def contract_with_clauses(contract_id):
    return Contract.objects.prefetch_related('versions__ai_extract_clauses').get(pk=contract_id)


def lock_version(version):
    return ContractVersion.objects.select_for_update().get(pk=version.pk)


def delete_version_clauses(version):
    return version.ai_extract_clauses.all().delete()

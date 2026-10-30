"""Scoped ledger API reads, separate from transaction/outbox persistence."""
from contracts.models import ContractVersion, ContractLedgerOperation
from NewIdea.repositories.contract_repository import visible_contracts

def version_for(user, version_id):
    return ContractVersion.objects.select_related('contract').get(pk=version_id, contract__in=visible_contracts(user))

def operations_for(user):
    rows = ContractLedgerOperation.objects.filter(version__contract__in=visible_contracts(user))
    if not user.is_superuser:
        rows = rows.filter(company_id=user.company_id)
    return rows.select_related('credential', 'actor', 'version__contract')


def operation_list(user, version=None, accepted_only=False):
    rows = operations_for(user)
    if version is not None:
        rows = rows.filter(version=version)
    if accepted_only:
        rows = rows.filter(accepted_at__isnull=False)
    return list(rows[:50])


def operation_for(user, operation_id):
    return operations_for(user).get(pk=operation_id)


def workspace_versions(user, version_id=None):
    versions = ContractVersion.objects.filter(contract__in=visible_contracts(user)).select_related('contract').order_by('-created_at')
    selected = versions.get(pk=version_id) if str(version_id or '').isdigit() else versions.first()
    return list(versions), selected


def credentials_for(user, manager):
    from contracts.models import ContractSigningCredential
    keys = ContractSigningCredential.objects.select_related('owner').all()
    if not user.is_superuser:
        keys = keys.filter(company_id=user.company_id) if manager and user.company_id else keys.filter(owner=user)
    return list(keys)

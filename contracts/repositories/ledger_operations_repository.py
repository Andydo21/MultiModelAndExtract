"""Database operations for ledger operations."""
from contracts.models import ContractVersion, ContractSigningCredential, ContractLedgerOperation

def get_or_create_contract_signing_credential_for_credentials(fingerprint, normalized, user):
    return ContractSigningCredential.objects.get_or_create(fingerprint=fingerprint, defaults={'owner': user, 'company_id': user.company_id, 'public_key_pem': normalized})

def get_contract_signing_credential_for_credential_action(key_id):
    return ContractSigningCredential.objects.select_for_update().get(pk=key_id)

def save_record_for_credential_action(key):
    return key.save()

def get_contract_ledger_operation_for_operation_action(op):
    return ContractLedgerOperation.objects.select_for_update().get(pk=op.pk)

def save_record_for_operation_action(op):
    return op.save()

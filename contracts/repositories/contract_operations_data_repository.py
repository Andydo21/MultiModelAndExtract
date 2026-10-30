"""Database operations for contract operations workflows."""
from contracts.models import Contract
from contracts.models import User

def get_contract_for_contract_detail(contract_id):
    return Contract.objects.get(id=contract_id)

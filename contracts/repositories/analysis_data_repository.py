"""Database operations for analysis workflows."""
from contracts.models import Contract
from contracts.models import Contract, RiskRule
from contracts.models import ContractVersion
from contracts.models import User

def get_contract_for_analyze_contract(contract_id):
    return Contract.objects.get(id=contract_id)







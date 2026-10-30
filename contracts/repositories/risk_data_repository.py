"""Database operations for risk workflows."""
from contracts.models import RiskFinding, Contract
from contracts.models import User

def distinct_risk_finding_for_list_all_risks(r):
    return RiskFinding.objects.filter(rule=r).values_list('analysis__version__contract_id', flat=True).distinct()


def only_contract_for_list_all_risks(contract_ids):
    return Contract.objects.filter(id__in=contract_ids).only('id', 'title', 'contract_code')


def first_user_for_create_new_risk():
    return User.objects.filter(is_superuser=True).first()

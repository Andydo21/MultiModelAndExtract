from contracts.models import Contract, ContractVersion, ContractFile, RiskRule, AIAnalysis, RiskFinding, Review, AuditLog
from ai_extract.models import Clause
import os


class ContractRepository:
    @staticmethod
    def get_all_contracts():
        return Contract.objects.all().order_by('-created_at')
        
    @staticmethod
    def get_contracts_by_company(company):
        return Contract.objects.filter(company=company).order_by('-created_at')
        
    @staticmethod
    def get_contract_by_id(contract_id):
        try:
            return Contract.objects.prefetch_related(
                'versions__files',
                'versions__ai_extract_clauses', 
                'versions__ai_analyses__findings__rule', 
                'versions__ai_analyses__reviews__user'
            ).get(id=contract_id)
        except Contract.DoesNotExist:
            return None
            
    @staticmethod
    def create_contract(code, title, contract_type, start_date, end_date, contract_value, status='DRAFT', company=None):
        return Contract.objects.create(
            contract_code=code,
            title=title,
            contract_type=contract_type,
            start_date=start_date,
            end_date=end_date,
            contract_value=contract_value,
            status=status,
            company=company
        )

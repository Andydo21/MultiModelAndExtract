from contracts.models import Contract, ContractVersion, ContractFile, RiskRule, AIAnalysis, RiskFinding, Review, AuditLog
from ai_extract.models import Clause
import os


class ClauseRepository:
    @staticmethod
    def create_clause(contract_or_version, title, content):
        if isinstance(contract_or_version, Contract):
            version, _ = ContractVersion.objects.get_or_create(contract=contract_or_version, version_number=1)
        else:
            version = contract_or_version
        return Clause.objects.create(
            version=version,
            clause_title=title,
            clause_content=content
        )

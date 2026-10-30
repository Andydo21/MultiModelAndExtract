from contracts.models import Contract, ContractVersion, ContractFile, RiskRule, AIAnalysis, RiskFinding, Review, AuditLog
from ai_extract.models import Clause
import os


class AIAnalysisRepository:
    @staticmethod
    def get_analysis_by_id(analysis_id):
        try:
            return AIAnalysis.objects.get(id=analysis_id)
        except AIAnalysis.DoesNotExist:
            return None
            
    @staticmethod
    def create_analysis(contract_or_version, model_name, overall_score, summary):
        if isinstance(contract_or_version, Contract):
            version, _ = ContractVersion.objects.get_or_create(contract=contract_or_version, version_number=1)
        else:
            version = contract_or_version
        return AIAnalysis.objects.create(
            version=version,
            model_name=model_name,
            overall_score=overall_score,
            summary=summary
        )

    @staticmethod
    def get_all_analyses():
        return AIAnalysis.objects.select_related('version__contract').prefetch_related(
            'findings__rule'
        ).order_by('-created_at')

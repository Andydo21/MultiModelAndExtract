from contracts.models import Contract, ContractVersion, ContractFile, RiskRule, AIAnalysis, RiskFinding, Review, AuditLog
from ai_extract.models import Clause
import os


class AuditLogRepository:
    @staticmethod
    def log_action(user, action, target_model, target_id):
        contract = None
        if target_model == "Contract":
            try:
                contract = Contract.objects.get(id=target_id)
            except Contract.DoesNotExist:
                pass
        elif target_model == "Review":
            try:
                review = Review.objects.get(id=target_id)
                contract = review.analysis.version.contract
            except Exception:
                pass
        
        return AuditLog.objects.create(
            user=user,
            contract=contract,
            action=action
        )

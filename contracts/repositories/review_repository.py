from contracts.models import Contract, ContractVersion, ContractFile, RiskRule, AIAnalysis, RiskFinding, Review, AuditLog
from ai_extract.models import Clause
import os


class ReviewRepository:
    @staticmethod
    def create_review(analysis, user, comment, final_risk_level):
        return Review.objects.create(
            analysis=analysis,
            user=user,
            note=comment,
            decision=final_risk_level
        )

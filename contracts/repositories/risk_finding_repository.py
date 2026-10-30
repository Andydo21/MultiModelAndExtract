from contracts.models import Contract, ContractVersion, ContractFile, RiskRule, AIAnalysis, RiskFinding, Review, AuditLog
from ai_extract.models import Clause
import os


class RiskFindingRepository:
    @staticmethod
    def create_finding(analysis, clause, risk, risk_level, explanation, recommendation, disadvantaged_party=None):
        return RiskFinding.objects.create(
            analysis=analysis,
            clause=clause,
            rule=risk,
            risk_level=risk_level,
            explanation=explanation,
            recommendation=recommendation,
            disadvantaged_party=disadvantaged_party
        )

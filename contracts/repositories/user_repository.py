from contracts.models import Contract, ContractVersion, ContractFile, RiskRule, AIAnalysis, RiskFinding, Review, AuditLog
from ai_extract.models import Clause
import os


class UserRepository:
    @staticmethod
    def get_all_users_with_roles():
        from contracts.models import User
        return User.objects.all().select_related('role')

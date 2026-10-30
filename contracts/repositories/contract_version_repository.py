from contracts.models import Contract, ContractVersion, ContractFile, RiskRule, AIAnalysis, RiskFinding, Review, AuditLog
from ai_extract.models import Clause
import os


class ContractVersionRepository:
    @staticmethod
    def get_version_by_id(version_id):
        try:
            return ContractVersion.objects.select_related('contract').get(id=version_id)
        except ContractVersion.DoesNotExist:
            return None

    @staticmethod
    def get_versions_by_ids(version_ids):
        return ContractVersion.objects.filter(id__in=version_ids).select_related('contract')

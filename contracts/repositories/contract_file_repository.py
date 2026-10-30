from contracts.models import Contract, ContractVersion, ContractFile, RiskRule, AIAnalysis, RiskFinding, Review, AuditLog
from ai_extract.models import Clause
import os


class ContractFileRepository:
    @staticmethod
    def create_file_record(contract_or_version, file_path):
        if isinstance(contract_or_version, Contract):
            version, _ = ContractVersion.objects.get_or_create(contract=contract_or_version, version_number=1)
        else:
            version = contract_or_version
            
        file_name = os.path.basename(file_path)
        return ContractFile.objects.create(
            version=version,
            file_name=file_name,
            file_path=file_path
        )

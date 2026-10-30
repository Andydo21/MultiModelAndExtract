"""Database operations for document download workflows."""
from contracts.models import ContractFile
from contracts.models import Contract, ContractVersion
from contracts.models import User

def get_contract_file_for_download_file(file_id):
    return ContractFile.objects.get(id=file_id)


def get_contract_for_download_contract_pdf(contract_id):
    return Contract.objects.get(id=contract_id)


def get_versions_for_download_contract_pdf(c, version_id):
    return c.versions.get(id=version_id)


def first_files_for_download_contract_pdf(version):
    return version.files.first()

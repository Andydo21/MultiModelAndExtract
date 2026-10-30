"""Contract authorization policy shared by the web endpoints."""

def check_contract_permission(user, contract):
    if not user or not user.is_authenticated:
        return False
    return user.is_superuser or bool(user.company_id and user.company_id == contract.company_id)


def contract_access_error(user, contract_id):
    """Check scope before reading versions or contacting another service."""
    from contracts.repositories import ContractRepository
    from .results import ServiceResult
    contract = ContractRepository.get_contract_by_id(contract_id)
    if contract is None:
        return ServiceResult({'error': 'Contract not found.'}, status=404)
    if not check_contract_permission(user, contract):
        return ServiceResult({'error': 'Permission denied.'}, status=403)
    return None

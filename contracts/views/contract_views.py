from .http_results import request_data as _body, service_errors
from django.contrib.auth.decorators import login_required
from .access import api_manager_required, api_login_required
from .http_results import respond
from contracts.services import contract_operations_service

@login_required
def contract_detail(request, contract_id):
    return respond(request, contract_operations_service.contract_detail_page(contract_id, user=request.user, query=request.GET))

@api_login_required
@service_errors
def api_contract_detail(request, contract_id):
    return respond(request, contract_operations_service.contract_detail(contract_id, user=request.user, method=request.method, query=request.GET))

@api_login_required
@service_errors
def api_contracts_list(request):
    return respond(request, contract_operations_service.contracts_list(user=request.user, method=request.method, form=request.POST, uploads=request.FILES))

@api_manager_required
@service_errors
def api_create_contract(request):
    return respond(request, contract_operations_service.create_contract(user=request.user, method=request.method, form=request.POST, uploads=request.FILES))

@api_manager_required
@service_errors
def api_manual_extract_contract(request, contract_id):
    return respond(request, contract_operations_service.manual_extract_contract(contract_id, user=request.user, method=request.method, data=_body(request), form=request.POST, query=request.GET))

@api_login_required
@service_errors
def api_contract_versions(request, contract_id):
    return respond(request, contract_operations_service.contract_versions(contract_id, user=request.user, method=request.method, form=request.POST, uploads=request.FILES))

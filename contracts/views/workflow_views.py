from .http_results import request_data as _body, service_errors
from .access import api_manager_required, api_login_required
from .http_results import respond
from contracts.services import workflow_gateway_service

@api_manager_required
@service_errors
def api_push_to_workflow(request, contract_id):
    return respond(request, workflow_gateway_service.push_to_workflow(contract_id, user=request.user, method=request.method, data=_body(request), query=request.GET))

@api_login_required
@service_errors
def api_workflow_status(request, contract_id):
    return respond(request, workflow_gateway_service.workflow_status(contract_id, user=request.user, method=request.method, query=request.GET))

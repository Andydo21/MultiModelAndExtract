from .http_results import request_data as _body, service_errors
from .access import _get_headers, api_manager_required, api_login_required
from .http_results import respond
from contracts.services import analysis_service

@api_login_required
@service_errors
def api_analyze_contract(request, contract_id):
    return respond(request, analysis_service.analyze_contract(contract_id, user=request.user, method=request.method, data=_body(request), form=request.POST, query=request.GET))

@service_errors
def api_submit_review(request, analysis_id):
    return respond(request, analysis_service.submit_review(analysis_id, method=request.method, data=_body(request)))

@api_manager_required
@service_errors
def api_analyses_list(request):
    return respond(request, analysis_service.analyses_list(user=request.user, method=request.method, data=_body(request)))


from .http_results import request_data as _body, service_errors
from .http_results import respond
from contracts.services import risk_catalog_service

@service_errors
def api_risks_list(request):
    return respond(request, risk_catalog_service.risks_list(method=request.method, data=_body(request), form=request.POST))

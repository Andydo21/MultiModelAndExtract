from .http_results import request_data as _body, service_errors
from .access import _get_headers
from .http_results import respond
from contracts.services import identity_service

@service_errors
def api_register_company(request):
    return respond(request, identity_service.register_company(method=request.method, data=_body(request), authorization_headers=_get_headers(request)))

@service_errors
def api_register_user(request):
    return respond(request, identity_service.register_user(user=request.user, method=request.method, data=_body(request), authorization_headers=_get_headers(request)))

@service_errors
def api_companies_list(request):
    return respond(request, identity_service.companies_list(data=_body(request)))

@service_errors
def api_users_list(request):
    return respond(request, identity_service.users_list(data=_body(request)))

@service_errors
def api_roles_list(request):
    return respond(request, identity_service.roles_list(data=_body(request)))

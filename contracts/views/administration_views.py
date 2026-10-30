from .http_results import request_data as _body, service_errors
from .access import _get_headers, api_manager_required
from .http_results import respond
from contracts.services import administration_service

@api_manager_required
@service_errors
def api_admin_stats(request):
    return respond(request, administration_service.admin_stats(query=request.GET))

@api_manager_required
@service_errors
def api_admin_delete_user(request, user_id):
    return respond(request, administration_service.admin_delete_user(user_id, user=request.user))

@api_manager_required
@service_errors
def api_admin_retry_user(request, user_id):
    return respond(request, administration_service.admin_retry_user(user_id, user=request.user, data=_body(request), authorization_headers=_get_headers(request)))

@api_manager_required
@service_errors
def api_admin_toggle_ocr(request):
    return respond(request, administration_service.admin_toggle_ocr(user=request.user, method=request.method, data=_body(request), metadata=request.META))

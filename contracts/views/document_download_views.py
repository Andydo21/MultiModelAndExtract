from .http_results import service_errors
from .access import api_manager_required
from .http_results import respond
from contracts.services import document_download_service

@api_manager_required
@service_errors
def api_download_file(request, file_id):
    return respond(request, document_download_service.download_file(file_id, user=request.user, content_type=request.content_type))

@api_manager_required
@service_errors
def api_download_contract_pdf(request, contract_id):
    return respond(request, document_download_service.download_contract_pdf(contract_id, user=request.user, query=request.GET))

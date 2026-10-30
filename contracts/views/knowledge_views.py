from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from NewIdea.services import contract_search_service
from NewIdea.views.knowledge_views import knowledge_api as knowledge_http_api, json_body
from functools import wraps
from django.core.exceptions import ObjectDoesNotExist
from django.http import Http404
from contracts.services import case_library_service
from .http_results import respond

def knowledge_errors(view):

    @wraps(view)
    def adapter(request, *args, **kwargs):
        try:
            return view(request, *args, **kwargs)
        except ObjectDoesNotExist as error:
            raise Http404('Knowledge resource not found.') from error
        except (ValueError, TypeError) as error:
            return JsonResponse({'success': False, 'error': str(error)}, status=400)
    return adapter

@login_required
@knowledge_errors
def workspace(request, contract_id=None, *, newidea=True):
    return respond(request, case_library_service.workspace(contract_id, user=request.user, query_params=request.GET, newidea=newidea))

@require_http_methods(['GET', 'POST'])
@knowledge_errors
def knowledge_api(request):
    return respond(request, case_library_service.manage_cases(user=request.user, method=request.method, data=json_body(request) if request.method != 'GET' else {}, query_params=request.GET))

@require_http_methods(['POST'])
@knowledge_http_api
def search_api(request, contract_id):
    return JsonResponse(contract_search_service.search_contract(request.user, contract_id, json_body(request), existing_receipt=getattr(request, '_knowledge_search_receipt', None)))

@require_http_methods(['POST'])
@knowledge_errors
def knowledge_draft_api(request):
    return respond(request, case_library_service.draft_expert_case(user=request.user, data=json_body(request) if request.method != 'GET' else {}))

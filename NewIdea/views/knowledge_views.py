"""HTTP adapters: authentication, JSON input and service result serialization."""
import json
from functools import wraps
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods, require_POST
from NewIdea.services import knowledge_search_service, expert_workflow_service, search_job_service, expert_submission_service
from NewIdea.services.errors import KnowledgeError


def json_body(request):
    data = json.loads(request.body)
    if not isinstance(data, dict):
        raise ValueError('JSON phải là object.')
    return data


def knowledge_api(view):
    @wraps(view)
    def adapter(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({'success': False, 'error': 'Authentication required.'}, status=401)
        try:
            return view(request, *args, **kwargs)
        except ObjectDoesNotExist:
            return JsonResponse({'success': False, 'error': 'Không tìm thấy dữ liệu trong phạm vi quyền truy cập.'}, status=404)
        except KnowledgeError as error:
            response = JsonResponse({'success': False, 'error': str(error)}, status=error.status)
            if error.status == 429:
                response['Retry-After'] = '60'
            return response
        except (ValueError, TypeError, ValidationError) as error:
            return JsonResponse({'success': False, 'error': str(error)}, status=400)
    return adapter


@require_POST
@knowledge_api
def search_knowledge(request):
    data = json_body(request)
    return JsonResponse(knowledge_search_service.search_knowledge(request.user, data.get('query'), generate=data.get('generate') is True))


@require_http_methods(['GET', 'POST'])
@knowledge_api
def inquiries(request):
    if request.method == 'GET':
        inquiry_id = request.GET.get('id')
        result = expert_workflow_service.inquiry_detail(request.user, inquiry_id) if inquiry_id else expert_workflow_service.list_inquiries(request.user)
    else:
        result = expert_workflow_service.update_inquiry(request.user, json_body(request))
    return JsonResponse(result)


@require_http_methods(['GET', 'POST'])
@knowledge_api
def search_jobs(request):
    if request.method == 'GET':
        return JsonResponse(search_job_service.read_job(request.user, request.GET.get('id')))
    return JsonResponse(search_job_service.enqueue_search(request.user, json_body(request)), status=202)


@require_http_methods(['GET'])
@knowledge_api
def evidence_sources(request, version_id):
    return JsonResponse(search_job_service.available_sources(request.user, version_id))


@require_http_methods(['GET', 'POST'])
@knowledge_api
def expert_submission(request):
    if request.method == 'GET':
        return JsonResponse(expert_submission_service.read_submission(request.user,request.GET.get('id','')))
    data = request.POST.dict() if request.content_type.startswith('multipart/') else json_body(request)
    return JsonResponse(expert_submission_service.enqueue(request.user, data, request.FILES.getlist('files')), status=202)

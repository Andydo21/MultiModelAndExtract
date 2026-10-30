import json
from functools import wraps
from django.core.exceptions import PermissionDenied, ObjectDoesNotExist
from django.http import JsonResponse, Http404
from django.contrib.auth.decorators import login_required
from contracts.services import ledger_operations_service
from .http_results import respond

def workspace_errors(view):

    @wraps(view)
    def adapter(request, *args, **kwargs):
        try:
            return view(request, *args, **kwargs)
        except ObjectDoesNotExist as error:
            raise Http404('Version not found.') from error
    return adapter

def api(methods):

    def decorate(fn):

        @wraps(fn)
        def view(request, *args, **kwargs):
            if not request.user.is_authenticated or not request.user.is_active:
                return JsonResponse({'error': 'Authentication required.'}, status=401)
            if request.method not in methods:
                return JsonResponse({'error': 'Method not allowed.'}, status=405)
            try:
                return fn(request, *args, **kwargs)
            except PermissionDenied as exc:
                return JsonResponse({'error': str(exc)}, status=403)
            except ObjectDoesNotExist:
                return JsonResponse({'error': 'Không tìm thấy tài nguyên.'}, status=404)
            except (ValueError, TypeError, KeyError) as exc:
                return JsonResponse({'error': str(exc)}, status=400)
        return view
    return decorate

def body(request):
    data = json.loads(request.body or '{}')
    if not isinstance(data, dict):
        raise ValueError('Cần JSON object.')
    return data

@login_required
@workspace_errors
def workspace(request):
    return respond(request, ledger_operations_service.workspace(user=request.user, query=request.GET))

@api(['GET', 'POST'])
def credentials(request):
    return respond(request, ledger_operations_service.credentials(user=request.user, method=request.method, data=body(request) if not request.content_type.startswith('multipart/') else {}))

@api(['POST'])
def credential_action(request, key_id):
    return respond(request, ledger_operations_service.credential_action(key_id, user=request.user, data=body(request) if not request.content_type.startswith('multipart/') else {}))

@api(['GET', 'POST'])
def operations(request):
    return respond(request, ledger_operations_service.operations(user=request.user, method=request.method, data=body(request) if not request.content_type.startswith('multipart/') else {}, query=request.GET))

@api(['GET'])
def workflow_steps(request):
    return respond(request, ledger_operations_service.workflow_steps(user=request.user, query=request.GET))

@api(['POST'])
def operation_action(request, operation_id):
    return respond(request, ledger_operations_service.operation_action(operation_id, user=request.user, data=body(request) if not request.content_type.startswith('multipart/') else {}))

@api(['POST'])
def verify(request):
    return respond(request, ledger_operations_service.verify(user=request.user, data=body(request) if not request.content_type.startswith('multipart/') else {}, form=request.POST, uploads=request.FILES, content_type=request.content_type))

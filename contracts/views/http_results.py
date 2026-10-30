"""Convert service outcomes to HTTP without putting Django responses in services."""
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render, redirect
from contracts.services.results import ServiceResult

def respond(request, result):
    if not isinstance(result, ServiceResult):
        raise TypeError('Expected ServiceResult')
    if result.redirect_to:
        response = redirect(result.redirect_to)
    elif result.template:
        response = render(request, result.template, result.payload, status=result.status)
    elif result.content_type == 'application/json':
        response = JsonResponse(result.payload, status=result.status, safe=isinstance(result.payload, dict))
    else:
        response = HttpResponse(result.payload, status=result.status, content_type=result.content_type)
    for name, value in result.headers.items():
        response[name] = value
    return response

def request_data(request):
    import json
    if request.content_type.startswith(('multipart/', 'application/x-www-form-urlencoded')):
        return request.POST.dict()
    value = json.loads(request.body or '{}')
    if not isinstance(value, dict):
        raise ValueError('JSON body must be an object.')
    return value

def service_errors(view):
    from functools import wraps
    from django.core.exceptions import ObjectDoesNotExist

    @wraps(view)
    def adapter(request, *args, **kwargs):
        try:
            return view(request, *args, **kwargs)
        except ObjectDoesNotExist:
            return JsonResponse({'error': 'Resource not found.'}, status=404)
        except (ValueError, TypeError, KeyError) as error:
            return JsonResponse({'error': str(error)}, status=400)
    return adapter

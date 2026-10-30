"""Authenticated JSON endpoints; extraction policy and persistence live below the view."""
import json
import logging
from functools import wraps
from django.core.exceptions import ObjectDoesNotExist
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from ai_extract.services import extraction_service
logger = logging.getLogger('ai_extract')

def extraction_api(method):

    def decorate(view):

        @require_http_methods([method])
        @wraps(view)
        def adapter(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return JsonResponse({'error': 'Authentication required.'}, status=401)
            try:
                return JsonResponse(view(request, *args, **kwargs))
            except ObjectDoesNotExist:
                return JsonResponse({'error': 'Contract, version or clause not found.'}, status=404)
            except extraction_service.ExtractionError as error:
                return JsonResponse({'error': str(error)}, status=error.status)
            except (ValueError, TypeError) as error:
                return JsonResponse({'error': str(error)}, status=400)
            except RuntimeError as error:
                logger.exception('AI extraction gateway failed')
                return JsonResponse({'error': str(error)}, status=502)
            except Exception:
                logger.exception('AI extraction failed')
                return JsonResponse({'error': 'Internal server error.'}, status=500)
        return adapter
    return decorate

def body(request):
    value = json.loads(request.body or '{}')
    if not isinstance(value, dict):
        raise ValueError('JSON body must be an object.')
    return value

@extraction_api('POST')
def api_summarize_contract(request, contract_id):
    return extraction_service.summarize(request.user, contract_id, body(request))

@extraction_api('GET')
def api_get_summary(request, contract_id):
    return extraction_service.summary(request.user, contract_id)

@extraction_api('POST')
def api_extract_entities(request, contract_id):
    return extraction_service.extract_entities(request.user, contract_id, body(request))

@extraction_api('POST')
def api_extract_from_text(request):
    return extraction_service.extract_text(request.user, body(request))

@extraction_api('GET')
def api_get_entities(request, contract_id):
    return extraction_service.entities(request.user, contract_id, request.GET.get('version_id'))

@extraction_api('POST')
def api_extract_clauses(request, contract_id):
    return extraction_service.extract_clauses(request.user, contract_id, body(request))

@extraction_api('GET')
def api_get_clauses(request, contract_id):
    return extraction_service.clauses(request.user, contract_id, request.GET.get('version_id'))

from django.http import JsonResponse

def _get_headers(request=None):
    from shared.jwt_middleware import get_auth_header, get_system_auth_header
    if request and hasattr(request, 'user') and request.user.is_authenticated:
        role = getattr(request.user.role, 'role_name', 'USER') if hasattr(request.user, 'role') and request.user.role else 'USER'
        return get_auth_header(request.user.id, request.user.username, role)
    return get_system_auth_header()

def manager_required(view_func):

    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({'error': 'Authentication required.'}, status=401)
        user = request.user
        if user.is_superuser:
            return view_func(request, *args, **kwargs)
        if not user.role or user.role.role_name.upper() not in ['MANAGER', 'ADMIN']:
            return JsonResponse({'error': 'Permission Denied: Manager role required.'}, status=403)
        return view_func(request, *args, **kwargs)
    return _wrapped_view

def api_manager_required(view_func):

    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({'error': 'Unauthorized. Please log in.'}, status=401)
        user = request.user
        if user.is_superuser:
            return view_func(request, *args, **kwargs)
        if not user.role or user.role.role_name.upper() not in ['MANAGER', 'ADMIN']:
            return JsonResponse({'error': 'Permission Denied: Only Managers can access this resource.'}, status=403)
        return view_func(request, *args, **kwargs)
    return _wrapped_view

def api_login_required(view_func):

    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({'error': 'Unauthorized. Please log in.'}, status=401)
        return view_func(request, *args, **kwargs)
    return _wrapped_view

from contracts.services.access_service import check_contract_permission as _check_contract_permission

from contracts.services import authentication_service
from .http_results import service_errors
from contracts.services import account_registration_service
import json
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import login, logout, authenticate
from .access import _get_headers

def login_user(request):
    """Render and handle the login page."""
    from contracts.models import Role
    account_registration_service.ensure_manager_role()
    if request.user.is_authenticated:
        return redirect('/')
    error = None
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        next_url = request.POST.get('next', '/') or '/'
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect(next_url)
        else:
            error = 'Tài khoản hoặc mật khẩu không chính xác.'
    next_url = request.GET.get('next', '/')
    return render(request, 'contracts/login.html', {'error': error, 'next': next_url})

def signup_user(request):
    account_registration_service.ensure_manager_role()
    if request.user.is_authenticated:
        return redirect('/')
    user, error = account_registration_service.register_account(request.method, request.POST, _get_headers(request))
    if user:
        login(request, user)
        return redirect('/')
    return render(request, 'contracts/signup.html', account_registration_service.registration_context(error))

def logout_user(request):
    """Log the user out and redirect to the login page."""
    logout(request)
    return redirect('/login/')

@csrf_exempt
@service_errors
def api_obtain_token(request):
    """POST -> Authenticate user and return a JWT token."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed.'}, status=405)
    try:
        import json
        import datetime
        import jwt
        from django.conf import settings
        from django.contrib.auth import authenticate
        if request.content_type == 'application/json':
            body = json.loads(request.body.decode('utf-8'))
            username = body.get('username')
            password = body.get('password')
        else:
            username = request.POST.get('username')
            password = request.POST.get('password')
        if not username or not password:
            return JsonResponse({'error': 'Both username and password are required.'}, status=400)
        user = authenticate(username=username, password=password)
        if user is not None:
            if not user.is_active:
                return JsonResponse({'error': 'User account is inactive.'}, status=400)
            return JsonResponse(authentication_service.token_for(user))
        else:
            return JsonResponse({'error': 'Invalid credentials.'}, status=401)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

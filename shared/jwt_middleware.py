"""
Shared JWT validation for all RiskDL services.
Single source of truth — mounted into every container via Docker volume.

Django services: register JWTMiddleware (and optionally JWTUserMiddleware for main web) in settings.MIDDLEWARE:
    'shared.jwt_middleware.JWTMiddleware'
    'shared.jwt_middleware.JWTUserMiddleware'

FastAPI services: use fastapi_jwt_required as a Depends() dependency:
    from shared.jwt_middleware import fastapi_jwt_required
    @app.post("/endpoint")
    def endpoint(payload: dict = Depends(fastapi_jwt_required)):
        ...
"""

import os
import jwt as pyjwt
from functools import wraps
from pathlib import Path
from dotenv import load_dotenv

# Auto-load root .env (parent of the shared/ directory) to sync SECRET_KEY
shared_dir = Path(__file__).resolve().parent
env_path = shared_dir.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

SECRET_KEY = os.environ.get("SECRET_KEY", "django-insecure-default-change-me")
ALGORITHM = "HS256"


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token. Raises jwt.ExpiredSignatureError or jwt.InvalidTokenError."""
    return pyjwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


def extract_bearer(auth_header: str) -> str | None:
    """Extract token string from 'Bearer <token>' header. Returns None if missing/malformed."""
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    return None


# ── Django middleware ──────────────────────────────────────────────────────────

class JWTMiddleware:
    """
    Lightweight Django middleware — validates JWT and attaches payload to request.
    Does DB user lookup for the main web if the 'contracts' app is installed.
    Safe for stateless microservices without a User database.

    Sets on request:
      - jwt_payload   : dict  — full decoded token payload
      - jwt_user_id   : int   — user ID from token
      - jwt_username  : str   — username from token
      - jwt_role      : str   — role from token
      - _jwt_user     : User  — DB User object (only on main web if user exists)
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        from django.http import JsonResponse
        from django.apps import apps

        token = extract_bearer(request.headers.get("Authorization", ""))
        if not token:
            token = request.COOKIES.get("jwt_token") or request.COOKIES.get("access_token")

        user_attached = False

        if token:
            try:
                payload = decode_token(token)
                request.jwt_payload      = payload
                request.jwt_user_id      = payload.get("user_id") or payload.get("id")
                request.jwt_username     = payload.get("username", "")
                request.jwt_role         = payload.get("role", "USER")
                request.jwt_company_id   = payload.get("company_id")
                request.jwt_is_superuser = payload.get("is_superuser", False)
                # Cookie authentication must retain CSRF protection.
                request._dont_enforce_csrf_checks = bool(extract_bearer(request.headers.get('Authorization', '')))

                if apps.is_installed("contracts"):
                    from django.contrib.auth import get_user_model
                    User = get_user_model()
                    try:
                        user = User.objects.get(id=request.jwt_user_id)
                        request._jwt_user = user
                        request.user = user
                        user_attached = True
                    except User.DoesNotExist:
                        pass
                else:
                    class LightweightRole:
                        def __init__(self, name):
                            self.role_name = name
                        def __str__(self):
                            return self.role_name

                    class LightweightUser:
                        def __init__(self, uid, username, role_name, company_id, is_superuser=False):
                            self.id = uid
                            self.pk = uid
                            self.username = username
                            self.is_authenticated = True
                            self.is_anonymous = False
                            self.is_superuser = is_superuser
                            self.company_id = company_id
                            self.role = LightweightRole(role_name or "USER")

                    request.user = LightweightUser(
                        uid=request.jwt_user_id,
                        username=request.jwt_username,
                        role_name=request.jwt_role,
                        company_id=request.jwt_company_id,
                        is_superuser=request.jwt_is_superuser
                    )
                    user_attached = True
            except Exception:
                pass

        # Fallback to sessionid cookie lookup if user is not attached yet
        if not user_attached and not apps.is_installed("contracts"):
            session_key = request.COOKIES.get("sessionid")
            if session_key:
                try:
                    from django.db import connections
                    db_alias = 'contract_db' if 'contract_db' in connections else 'default'
                    with connections[db_alias].cursor() as cursor:
                        cursor.execute("SELECT session_data FROM django_session WHERE session_key = %s", [session_key])
                        row = cursor.fetchone()
                        if row:
                            from django.contrib.sessions.backends.db import SessionStore
                            s_data = SessionStore().decode(row[0])
                            uid = s_data.get('_auth_user_id')
                            if uid:
                                cursor.execute("""
                                    SELECT u.id, u.username, r.role_name, u.company_id, u.is_superuser
                                    FROM contracts_user u
                                    LEFT JOIN contracts_role r ON u.role_id = r.id
                                    WHERE u.id = %s
                                """, [int(uid)])
                                u_row = cursor.fetchone()
                                if u_row:
                                    u_id, u_name, r_name, c_id, is_sup = u_row
                                    class LightweightRole:
                                        def __init__(self, name):
                                            self.role_name = name
                                        def __str__(self):
                                            return self.role_name

                                    class LightweightUser:
                                        def __init__(self, uid, username, role_name, company_id, is_superuser=False):
                                            self.id = uid
                                            self.pk = uid
                                            self.username = username
                                            self.is_authenticated = True
                                            self.is_anonymous = False
                                            self.is_superuser = is_superuser
                                            self.company_id = company_id
                                            self.role = LightweightRole(role_name or "USER")

                                    request.user = LightweightUser(u_id, u_name, r_name, c_id, is_sup)
                                    request.jwt_user_id = u_id
                                    request.jwt_username = u_name
                                    request.jwt_role = r_name
                                    request.jwt_company_id = c_id
                                    request.jwt_is_superuser = is_sup
                except Exception:
                    pass

        return self.get_response(request)


class JWTUserMiddleware:
    """
    Restores the JWT-authenticated user onto request.user AFTER
    Django's AuthenticationMiddleware overwrites it.
    Only needed on main web service.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if hasattr(request, "_jwt_user"):
            request.user = request._jwt_user
        return self.get_response(request)


def jwt_required(view_func):
    """Django view decorator — returns 401 if no valid JWT present."""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not getattr(request, "jwt_user_id", None):
            from django.http import JsonResponse
            return JsonResponse({"error": "Authentication required."}, status=401)
        return view_func(request, *args, **kwargs)
    return wrapper


def jwt_role_required(*roles):
    """Django view decorator — returns 403 if JWT role not in allowed list."""
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            from django.http import JsonResponse
            if not getattr(request, "jwt_user_id", None):
                return JsonResponse({"error": "Authentication required."}, status=401)
            if request.jwt_role not in roles:
                return JsonResponse(
                    {"error": f"Forbidden. Required roles: {list(roles)}"},
                    status=403,
                )
            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator
try:
    from fastapi import Header
except ImportError:
    Header = None


# ── FastAPI dependency ─────────────────────────────────────────────────────────

def fastapi_jwt_required(authorization: str = Header(None) if Header else None):
    """
    FastAPI dependency for JWT validation.
    Extracts Bearer token from incoming Authorization header and returns decoded payload.
    """
    from fastapi import HTTPException
    # Note: Header(None) is handled inside route definitions, but we fetch it here
    token = extract_bearer(authorization or "")
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid.")
    try:
        return decode_token(token)
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired.")
    except pyjwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


# ── Token generation helpers for inter-service communication ────────────────────

import datetime

def generate_token(user_id, username: str, role: str, expires_in_hours: int = 24) -> str:
    """Generate a JWT token signed with the shared SECRET_KEY."""
    payload = {
        "user_id": user_id,
        "username": username,
        "role": role,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=expires_in_hours),
        "iat": datetime.datetime.utcnow(),
    }
    return pyjwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_auth_header(user_id, username: str, role: str) -> dict:
    """Get request headers with JWT Bearer authorization."""
    token = generate_token(user_id, username, role)
    return {"Authorization": f"Bearer {token}"}

def get_system_auth_header() -> dict:
    """Get system-level authorization header."""
    return get_auth_header(0, "system", "ADMIN")

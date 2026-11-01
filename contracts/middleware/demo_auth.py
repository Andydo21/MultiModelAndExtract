"""
Middleware for demo / guest access without login.
Automatically attaches an admin/manager user to unauthenticated requests
so external evaluators can test contract extraction and AI intelligence without logging in.
"""
from django.contrib.auth import get_user_model


class AutoDemoLoginMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # If user is not yet authenticated via session or JWT, attach demo admin user
        user = getattr(request, 'user', None)
        if not user or not user.is_authenticated:
            User = get_user_model()
            # Prioritize superuser 'admin' so all contracts are visible in visible_contracts()
            demo_user = (
                User.objects.filter(username='admin', is_superuser=True).first()
                or User.objects.filter(is_superuser=True).first()
                or User.objects.filter(role__role_name='ADMIN').first()
                or User.objects.first()
            )
            if demo_user:
                request.user = demo_user
                request._jwt_user = demo_user
                request.jwt_user_id = demo_user.id
                request.jwt_username = demo_user.username
                request.jwt_role = (
                    getattr(demo_user.role, 'role_name', 'ADMIN')
                    if getattr(demo_user, 'role', None)
                    else 'ADMIN'
                )
                request.jwt_company_id = demo_user.company_id
                request.jwt_is_superuser = demo_user.is_superuser
                # Login-free visitors never receive a CSRF cookie, so their POSTs are not checked.
                request._dont_enforce_csrf_checks = True

        return self.get_response(request)

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .access import manager_required

@login_required
def dashboard(request):
    """Render the dashboard SPA template."""
    return render(request, 'contracts/dashboard.html')

@login_required
def workflow_board(request):
    """Chuyển hướng người dùng sang Workflow Board UI của microservice workflow_service."""
    import os
    from django.conf import settings
    wf_url = getattr(settings, 'WORKFLOW_SERVICE_PUBLIC_URL', os.environ.get('WORKFLOW_SERVICE_PUBLIC_URL', 'http://localhost:8003'))
    return redirect(f"{wf_url.rstrip('/')}/board/")

@manager_required
def analysis_history(request):
    """Render the AI analysis history page."""
    return render(request, 'contracts/analysis_history.html')


@login_required(login_url='/login/')
def identity_registry(request):
    """Render the identity registry page."""
    return render(request, 'contracts/identity_registry.html')

@manager_required
def admin_dashboard(request):
    """Render the custom Admin Dashboard."""
    return render(request, 'contracts/admin_dashboard.html')


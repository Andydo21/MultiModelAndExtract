"""Standalone entry point; contract search lives in the authenticated web."""
from django.conf import settings
from django.shortcuts import redirect


def chat_view(request):
    return redirect(settings.CONTRACT_WEB_URL.rstrip('/') + '/newidea/')

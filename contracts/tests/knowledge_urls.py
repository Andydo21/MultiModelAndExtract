"""Test routing for new endpoints; source links are reversed, not fetched."""
from django.http import HttpResponse
from django.urls import path
from contracts.views.knowledge_urls import urlpatterns
from contracts.views.ledger_urls import urlpatterns as ledger_urls
from NewIdea.urls import urlpatterns as newidea_urls

urlpatterns = urlpatterns + ledger_urls + newidea_urls + [
    path('contracts/<int:contract_id>/', lambda request, **kwargs: HttpResponse(), name='contract_detail'),
    path('ai-extract/contracts/<int:contract_id>/pages/<int:page_number>/',
         lambda request, **kwargs: HttpResponse(), name='ai_extract_page'),
]

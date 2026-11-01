from django.urls import path, include
from ai_extract.views import layout_views
urlpatterns = [
    path('', include('contracts.urls')),
    path('', include('contracts.views.knowledge_urls')),
    path('', include('contracts.views.ledger_urls')),
    path('', include('NewIdea.urls')),
    path('api/ai/', include('ai_extract.urls')),
    path('ai-extract/', layout_views.pipeline_dashboard, name='ai_extract_pipeline'),
    path('ai-extract/contracts/<int:contract_id>/', layout_views.contract_pipeline, name='ai_extract_contract'),
    path('ai-extract/contracts/<int:contract_id>/pages/<int:page_number>/', layout_views.page_inspector, name='ai_extract_page'),
]

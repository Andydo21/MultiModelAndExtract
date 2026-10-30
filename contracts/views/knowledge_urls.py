from django.urls import path
from . import knowledge_views as intelligence

urlpatterns = [
    path('contract-intelligence/', intelligence.workspace, name='contract_intelligence'),
    path('contracts/<int:contract_id>/intelligence/', intelligence.workspace, name='contract_intelligence_detail'),
    path('api/contract-knowledge/', intelligence.knowledge_api, name='contract_knowledge_api'),
    path('api/contract-knowledge/draft/', intelligence.knowledge_draft_api, name='contract_knowledge_draft'),
    path('api/contracts/<int:contract_id>/intelligence/search/', intelligence.search_api, name='contract_intelligence_search'),
]

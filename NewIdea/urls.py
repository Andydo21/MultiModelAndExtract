"""Contract-mode NewIdea routes mounted in the main web, not prototype URLs."""
from django.urls import path
from .views import contract_views
from .views.knowledge_views import inquiries, evidence_sources, search_jobs
from .views.knowledge_views import search_knowledge
from .views.knowledge_views import expert_submission

urlpatterns=[
    path('api/newidea/expert-submissions/', expert_submission, name='newidea_expert_submission'),
    path('newidea/',contract_views.workspace,name='newidea_workspace'),
    path('newidea/contracts/<int:contract_id>/',contract_views.workspace,name='newidea_contract'),
    path('api/newidea/ask/',contract_views.ask,name='newidea_ask'),
    path('api/newidea/inquiries/', inquiries, name='newidea_inquiries'),
    path('api/newidea/search/', search_knowledge, name='newidea_knowledge_search'),
    path('api/newidea/jobs/', search_jobs, name='newidea_search_jobs'),
    path('api/newidea/versions/<int:version_id>/sources/', evidence_sources, name='newidea_sources'),
    path('api/newidea/contracts/<int:contract_id>/search/',contract_views.search_contract,name='newidea_search'),
]

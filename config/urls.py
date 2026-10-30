from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from ai_extract.views import layout_views
from contracts.views import workflow_proxy

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('contracts.urls')),
    path('', include('contracts.views.knowledge_urls')),
    path('', include('NewIdea.urls')),
    path('', include('contracts.views.ledger_urls')),
    path('api/ai/', include('ai_extract.urls')),
    path('ai-extract/', layout_views.pipeline_dashboard, name='ai_extract_pipeline'),
    path('ai-extract/contracts/<int:contract_id>/', layout_views.contract_pipeline, name='ai_extract_contract'),
    path(
        'ai-extract/contracts/<int:contract_id>/pages/<int:page_number>/',
        layout_views.page_inspector,
        name='ai_extract_page',
    ),
    # Workflow Service Reverse Proxy (enables seamless single-domain online access)
    path('board/', workflow_proxy.proxy_to_workflow, {'path': 'board/'}, name='workflow_board_proxy'),
    path('board/<path:path>', lambda r, path: workflow_proxy.proxy_to_workflow(r, f'board/{path}')),
    path('workflows/detail/<int:workflow_id>/', lambda r, workflow_id: workflow_proxy.proxy_to_workflow(r, f'workflows/detail/{workflow_id}/')),
    path('workflows/steps/<int:step_id>/approve/', lambda r, step_id: workflow_proxy.proxy_to_workflow(r, f'workflows/steps/{step_id}/approve/')),
    path('workflows/steps/<int:step_id>/<path:path>', lambda r, step_id, path: workflow_proxy.proxy_to_workflow(r, f'workflows/steps/{step_id}/{path}')),
    path('workflows/<int:workflow_id>/<path:path>', lambda r, workflow_id, path: workflow_proxy.proxy_to_workflow(r, f'workflows/{workflow_id}/{path}')),
    path('workflows/<path:path>', lambda r, path: workflow_proxy.proxy_to_workflow(r, f'workflows/{path}')),
    path('dependencies/<path:path>', lambda r, path: workflow_proxy.proxy_to_workflow(r, f'dependencies/{path}')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


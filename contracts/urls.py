from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('workflows/', views.workflow_board, name='workflow_board'),
    path('contracts/<int:contract_id>/', views.contract_detail, name='contract_detail'),
    path('analyses/', views.analysis_history, name='analysis_history'),
    path('api/contracts/', views.api_contracts_list, name='api_contracts_list'),
    path('api/contracts/create/', views.api_create_contract, name='api_create_contract'),
    path('api/contracts/<int:contract_id>/', views.api_contract_detail, name='api_contract_detail'),
    path('api/contracts/<int:contract_id>/analyze/', views.api_analyze_contract, name='api_analyze_contract'),
    path('api/contracts/<int:contract_id>/manual-extract/', views.api_manual_extract_contract, name='api_manual_extract_contract'),
    path('api/contracts/<int:contract_id>/workflow/', views.api_push_to_workflow, name='api_push_to_workflow'),
    path('api/contracts/<int:contract_id>/workflow/status/', views.api_workflow_status, name='api_workflow_status'),
    path('api/contracts/<int:contract_id>/versions/', views.api_contract_versions, name='api_contract_versions'),
    path('api/contracts/files/<int:file_id>/download/', views.api_download_file, name='api_download_file'),
    path('api/contracts/<int:contract_id>/download-pdf/', views.api_download_contract_pdf, name='api_download_contract_pdf'),
    path('api/analyses/', views.api_analyses_list, name='api_analyses_list'),
    path('api/analyses/<int:analysis_id>/review/', views.api_submit_review, name='api_submit_review'),
    path('api/risks/', views.api_risks_list, name='api_risks_list'),
    # Identity Registry
    path('identity/', views.identity_registry, name='identity_registry'),
    path('api/identity/companies/', views.api_companies_list, name='api_companies_list'),
    path('api/identity/companies/register/', views.api_register_company, name='api_register_company'),
    path('api/identity/users/', views.api_users_list, name='api_users_list'),
    path('api/identity/users/register/', views.api_register_user, name='api_register_user'),
    path('api/identity/roles/', views.api_roles_list, name='api_roles_list'),
    path('login/', views.login_user, name='login'),
    path('signup/', views.signup_user, name='signup'),
    path('logout/', views.logout_user, name='logout'),
    # Custom Admin Dashboard
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('api/admin/stats/', views.api_admin_stats, name='api_admin_stats'),
    path('api/admin/users/<int:user_id>/delete/', views.api_admin_delete_user, name='api_admin_delete_user'),
    path('api/admin/users/<int:user_id>/retry/', views.api_admin_retry_user, name='api_admin_retry_user'),
    path('api/admin/config/ocr/', views.api_admin_toggle_ocr, name='api_admin_toggle_ocr'),
    path('api/token/', views.api_obtain_token, name='api_obtain_token'),
]

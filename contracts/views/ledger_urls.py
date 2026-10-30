from django.urls import path
from . import ledger_views as v

urlpatterns = [
    path('contract-ledger/', v.workspace, name='contract_ledger'),
    path('api/ledger/credentials/', v.credentials, name='ledger_credentials'),
    path('api/ledger/credentials/<int:key_id>/', v.credential_action, name='ledger_credential_action'),
    path('api/ledger/operations/', v.operations, name='ledger_operations'),
    path('api/ledger/operations/<uuid:operation_id>/', v.operation_action, name='ledger_operation_action'),
    path('api/ledger/verify/', v.verify, name='ledger_verify'),
    path('api/ledger/workflow-steps/', v.workflow_steps, name='ledger_workflow_steps'),
]

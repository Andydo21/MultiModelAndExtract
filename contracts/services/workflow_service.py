from contracts.repositories import workflow_data_repository as data_repository
import os
from contracts.repositories import ContractRepository, AuditLogRepository

class WorkflowService:

    def __init__(self):
        self.contract_repo = ContractRepository()
        self.audit_repo = AuditLogRepository()

    def push_to_workflow(self, contract_id, version_id=None):
        """Đẩy contract lên workflow-service để tạo quy trình phê duyệt."""
        import requests
        import os
        import urllib.parse
        from django.conf import settings
        from contracts.models import ContractVersion, ContractContext
        contract = self.contract_repo.get_contract_by_id(contract_id)
        if not contract:
            raise ValueError('Contract not found.')
        if version_id:
            try:
                version = data_repository.get_versions_for_push_to_workflow(contract, version_id)
            except ContractVersion.DoesNotExist:
                raise ValueError('Version not found.')
        else:
            version = contract.latest_version
            if not version:
                raise ValueError('Contract has no version to push.')
        raw_content = ''
        contexts = data_repository.order_by_contexts_for_push_to_workflow(version)
        if contexts.exists():
            raw_content = '\n\n'.join([ctx.content for ctx in contexts])
        if not raw_content:
            latest_file = data_repository.first_files_for_push_to_workflow(version)
            if latest_file and latest_file.file_path:
                media_prefix = settings.MEDIA_URL
                rel_path = urllib.parse.unquote(latest_file.file_path)
                if rel_path.startswith(media_prefix):
                    rel_path = rel_path[len(media_prefix):]
                full_path = os.path.join(settings.MEDIA_ROOT, rel_path.replace('/', os.sep))
                if os.path.exists(full_path):
                    try:
                        from contracts.services.document_encryption_service import decrypt_pdf
                        with open(full_path, 'rb') as f:
                            encrypted_bytes = f.read()
                        decrypted_bytes = decrypt_pdf(encrypted_bytes)
                        raw_content = decrypted_bytes.decode('utf-8', errors='ignore')
                    except Exception:
                        pass
        if not raw_content:
            clauses = data_repository.all_ai_extract_clauses_for_push_to_workflow(version)
            if clauses.exists():
                raw_content = '\n\n'.join([f'--- {cl.clause_title} ---\n{cl.clause_content}' for cl in clauses])
        clause_types = list(data_repository.distinct_ai_extract_clauses_for_push_to_workflow(version))
        workflow_url = getattr(settings, 'WORKFLOW_SERVICE_URL', 'http://workflow-service:8000')
        payload = {'version_id': version.id, 'workflow_name': f'Approval Workflow – {contract.title}', 'contract_text': raw_content, 'clause_types': clause_types, 'contract_type': contract.contract_type or ''}
        try:
            from shared.jwt_middleware import get_system_auth_header
            resp = requests.post(f'{workflow_url}/workflows/', json=payload, headers=get_system_auth_header(), timeout=300)
            if resp.status_code == 409:
                get_resp = requests.get(f'{workflow_url}/workflows/{version.id}/', headers=get_system_auth_header(), timeout=300)
                get_resp.raise_for_status()
                result = get_resp.json()
            else:
                resp.raise_for_status()
                result = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f'Workflow service error: {e}')
        admin_user = data_repository.first_user_for_push_to_workflow()
        self.audit_repo.log_action(admin_user, 'PUSHED_TO_WORKFLOW', 'Contract', contract.id)
        return {'contract_id': contract.id, 'workflow_id': result.get('workflow_id'), 'workflow_name': result.get('workflow_name'), 'status': result.get('status'), 'steps': result.get('steps', [])}

    def get_workflow_status(self, contract_id, version_id=None):
        """Lấy trạng thái workflow từ workflow-service."""
        import requests
        from django.conf import settings
        from contracts.models import ContractVersion
        contract = self.contract_repo.get_contract_by_id(contract_id)
        if not contract:
            raise ValueError('Contract not found.')
        if version_id:
            try:
                version = data_repository.get_versions_for_push_to_workflow(contract, version_id)
            except ContractVersion.DoesNotExist:
                raise ValueError('Version not found.')
        else:
            version = contract.latest_version
            if not version:
                return None
        workflow_url = getattr(settings, 'WORKFLOW_SERVICE_URL', 'http://workflow-service:8000')
        try:
            from shared.jwt_middleware import get_system_auth_header
            resp = requests.get(f'{workflow_url}/workflows/{version.id}/', headers=get_system_auth_header(), timeout=10)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f'Workflow service error: {e}')

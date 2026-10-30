from contracts.services import WorkflowService
from contracts.services.results import ServiceResult
from .access_service import contract_access_error
workflow_service = WorkflowService()

def push_to_workflow(contract_id, user, method, data, query):
    """
    POST: Đẩy contract lên workflow-service để khởi tạo quy trình phê duyệt.
    """
    access_error = contract_access_error(user, contract_id)
    if access_error:
        return access_error
    if method == 'POST':
        version_id = query.get('version_id')
        if not version_id and data:
            try:
                body = data
                version_id = body.get('version_id')
            except Exception:
                pass
        try:
            result = workflow_service.push_to_workflow(contract_id, version_id=version_id)
            return ServiceResult({'success': True, **result})
        except Exception as e:
            return ServiceResult({'error': str(e)}, status=400)
    return ServiceResult({'error': 'Method not allowed.'}, status=405)

def workflow_status(contract_id, user, method, query):
    """
    GET: Lấy trạng thái workflow hiện tại của contract từ workflow-service.
    """
    access_error = contract_access_error(user, contract_id)
    if access_error:
        return access_error
    if method == 'GET':
        version_id = query.get('version_id')
        try:
            result = workflow_service.get_workflow_status(contract_id, version_id=version_id)
            if result is None:
                return ServiceResult({'workflow': None})
            return ServiceResult({'workflow': result})
        except Exception as e:
            return ServiceResult({'error': str(e)}, status=400)
    return ServiceResult({'error': 'Method not allowed.'}, status=405)

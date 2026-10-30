from contracts.repositories import administration_data_repository as data_repository
import os as _os
import requests as _req
from contracts.services.results import ServiceResult
_BC_URL = _os.environ.get('BLOCKCHAIN_SERVICE_URL', 'http://blockchain-service:8000')

def admin_stats(query):
    """GET → stats, service health checks, and recent audit logs."""
    from django.conf import settings
    from contracts.models import User, Company, Contract, AuditLog
    import time
    total_users = data_repository.count_user_for_admin_stats()
    total_companies = data_repository.count_company_for_admin_stats()
    total_contracts = data_repository.count_contract_for_admin_stats()
    health_services = {'AI Service': _os.environ.get('AI_SERVICE_URL', 'http://localhost:8001'), 'Blockchain Service': _BC_URL, 'Workflow Service': _os.environ.get('WORKFLOW_SERVICE_URL', 'http://localhost:8003'), 'MCP Server': _os.environ.get('LEGAL_MCP_SERVER_URL', 'http://127.0.0.1:8005/rpc')}
    health_results = {}
    for name, url in health_services.items():
        status = 'OFFLINE'
        latency = 0.0
        try:
            start_time = time.time()
            if name == 'MCP Server':
                r = _req.post(url, json={'jsonrpc': '2.0', 'method': 'ping', 'id': 1}, timeout=2)
            else:
                r = _req.get(url, timeout=2)
            latency = round((time.time() - start_time) * 1000, 1)
            if r.status_code < 500:
                status = 'ONLINE'
        except Exception:
            status = 'OFFLINE'
        health_results[name] = {'status': status, 'latency': latency, 'url': url}
    try:
        page = int(query.get('page', 1))
    except ValueError:
        page = 1
    limit = 10
    offset = (page - 1) * limit
    total_logs = data_repository.count_audit_log_for_admin_stats()
    logs = data_repository.order_by_audit_log_for_admin_stats()[offset:offset + limit]
    logs_data = []
    for l in logs:
        logs_data.append({'id': l.id, 'user': l.user.username if l.user else 'System', 'action': l.action, 'contract_code': l.contract.contract_code if l.contract else None, 'ip_address': l.ip_address or 'N/A', 'payload': l.payload, 'created_at': l.created_at.strftime('%Y-%m-%d %H:%M:%S')})
    users = data_repository.order_by_user_for_admin_stats()
    users_data = []
    for u in users:
        users_data.append({'id': u.id, 'username': u.username, 'email': u.email, 'company_name': u.company.company_name if u.company else 'N/A', 'role_name': u.role.role_name if u.role else 'Staff', 'status': u.status, 'block_number': u.block_number, 'tx_hash': u.tx_hash})
    return ServiceResult({'stats': {'total_users': total_users, 'total_companies': total_companies, 'total_contracts': total_contracts}, 'health': health_results, 'logs': logs_data, 'logs_pagination': {'page': page, 'total_logs': total_logs, 'limit': limit, 'total_pages': (total_logs + limit - 1) // limit if total_logs > 0 else 1}, 'users': users_data, 'skip_ocr': _os.environ.get('SKIP_OCR', 'False').lower() in ('true', '1', 'yes')})

def admin_delete_user(user_id, user):
    if user.pk == user_id:
        return ServiceResult({'error': 'Cannot delete your own account'}, status=400)
    'POST -> Delete a user from the local database.'
    from contracts.models import User
    try:
        user = data_repository.get_user_for_admin_delete_user(user_id)
        data_repository.delete_user_for_admin_delete_user(user)
        return ServiceResult({'status': 'success', 'deleted_id': user_id})
    except User.DoesNotExist:
        return ServiceResult({'error': 'User not found'}, status=404)
    except Exception as e:
        return ServiceResult({'error': str(e)}, status=500)

def admin_retry_user(user_id, user, data, authorization_headers):
    """POST -> Retry anchoring a user stuck in ERROR state to Hyperledger Fabric."""
    import requests
    from contracts.models import User
    try:
        user = data_repository.get_user_for_admin_delete_user(user_id)
        if not user.company or not user.role:
            return ServiceResult({'error': 'User missing company or role'}, status=400)
        resp = requests.post(f'{_BC_URL}/user/register/', json={'user_id': user.id, 'username': user.username, 'company_id': user.company.id, 'role': user.role.role_name, 'status': 'ACTIVE', 'sender': 'System'}, headers=authorization_headers, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            user.tx_hash = data.get('tx_hash')
            user.block_number = data.get('block_number')
            user.block_hash = data.get('block_hash')
            user.status = 'ACTIVE'
            data_repository.save_user_for_admin_retry_user(user)
            return ServiceResult({'status': 'success', 'user_id': user.id, 'block_number': user.block_number})
        else:
            return ServiceResult({'error': resp.json().get('error', 'Blockchain error')}, status=400)
    except User.DoesNotExist:
        return ServiceResult({'error': 'User not found'}, status=404)
    except Exception as e:
        return ServiceResult({'error': str(e)}, status=500)

def admin_toggle_ocr(user, method, data, metadata):
    """GET or POST → view or toggle dynamic OCR engine setting."""
    from contracts.models import AuditLog
    if method == 'GET':
        skip_ocr = _os.environ.get('SKIP_OCR', 'False').lower() in ('true', '1', 'yes')
        return ServiceResult({'skip_ocr': skip_ocr})
    elif method == 'POST':
        try:
            body = data
            skip_ocr = body.get('skip_ocr', True)
            _os.environ['SKIP_OCR'] = 'True' if skip_ocr else 'False'
            data_repository.create_audit_log_for_admin_toggle_ocr(metadata, skip_ocr, user)
            return ServiceResult({'status': 'success', 'skip_ocr': skip_ocr})
        except Exception as e:
            return ServiceResult({'error': str(e)}, status=500)
    return ServiceResult({'error': 'Method not allowed'}, status=405)

from contracts.repositories import identity_data_repository as data_repository
import os as _os
from contracts.services.results import ServiceResult
_BC_URL = _os.environ.get('BLOCKCHAIN_SERVICE_URL', 'http://blockchain-service:8000')

def register_company(method, data, authorization_headers):
    """POST: Register a new Company in DB and anchor it to the blockchain."""
    import requests
    if method != 'POST':
        return ServiceResult({'error': 'Method not allowed.'}, status=405)
    try:
        body = data
        name = body.get('company_name')
        tax_code = body.get('tax_code')
        if not name or not tax_code:
            return ServiceResult({'error': 'Thiếu tên công ty hoặc mã số thuế'}, status=400)
        from contracts.models import Company
        existing_comp = data_repository.find_company_by_tax_code(tax_code) or data_repository.find_company_by_name(name)
        if existing_comp:
            return ServiceResult({'error': f'Công ty "{existing_comp.company_name}" (MST: {existing_comp.tax_code}) đã tồn tại trong hệ thống.'}, status=400)
        company = data_repository.create_company_for_register_company(name, tax_code)
        try:
            resp = requests.post(f'{_BC_URL}/company/register/', json={'company_id': company.id, 'company_name': name, 'tax_code': tax_code, 'status': 'ACTIVE', 'sender': 'System'}, headers=authorization_headers, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                company.tx_hash = data.get('tx_hash')
                company.block_number = data.get('block_number')
                company.block_hash = data.get('block_hash')
                data_repository.save_company_for_register_company(company)
            else:
                error_msg = resp.json().get('error', 'Unknown blockchain error')
                company.status = 'ERROR'
                data_repository.save_company_for_register_company(company)
                return ServiceResult({'error': f'Lỗi lưu blockchain: {error_msg}'}, status=400)
        except Exception as e:
            company.status = 'ERROR'
            data_repository.save_company_for_register_company(company)
            return ServiceResult({'error': f'Không kết nối được dịch vụ blockchain: {str(e)}'}, status=500)
        return ServiceResult({'status': 'success', 'company_id': company.id, 'company_name': company.company_name, 'tax_code': company.tax_code, 'tx_hash': company.tx_hash, 'block_number': company.block_number, 'block_hash': company.block_hash})
    except Exception as e:
        return ServiceResult({'error': str(e)}, status=500)

def register_user(user, method, data, authorization_headers):
    """POST: Register a new User in DB and anchor them to the blockchain."""
    import requests
    if method != 'POST':
        return ServiceResult({'error': 'Method not allowed.'}, status=405)
    try:
        body = data
        username = body.get('username')
        email = body.get('email')
        password = body.get('password')
        company_id = body.get('company_id')
        role_id = body.get('role_id')
        if not username or not email or (not password) or (not company_id) or (not role_id):
            return ServiceResult({'error': 'Thiếu thông tin đăng ký nhân sự'}, status=400)
        from contracts.models import Company, Role, User
        try:
            company = data_repository.get_company_for_register_user(company_id)
            role = data_repository.get_role_for_register_user(role_id)
        except (Company.DoesNotExist, Role.DoesNotExist) as e:
            return ServiceResult({'error': 'Công ty hoặc Vai trò không tồn tại'}, status=404)
        existing_user = data_repository.first_user_for_register_user(username)
        if existing_user:
            if existing_user.status == 'ACTIVE' and existing_user.block_number:
                return ServiceResult({'error': f'Tài khoản "{username}" đã tồn tại và đã được Anchor trên Blockchain.'}, status=400)
            user = existing_user
            user.email = email
            user.set_password(password)
            user.company = company
            user.role = role
            user.status = 'ACTIVE'
            data_repository.save_user_for_register_user(user)
        else:
            user = data_repository.create_registered_user(username, email, password, company, role)
        try:
            resp = requests.post(f'{_BC_URL}/user/register/', json={'user_id': user.id, 'username': username, 'company_id': company.id, 'role': role.role_name, 'status': 'ACTIVE', 'sender': 'System'}, headers=authorization_headers, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                user.tx_hash = data.get('tx_hash')
                user.block_number = data.get('block_number')
                user.block_hash = data.get('block_hash')
                user.status = 'ACTIVE'
                data_repository.save_user_for_register_user(user)
                try:
                    import random
                    clean_company_name = ''.join((x for x in company.company_name.upper() if x.isalnum() or x == ' '))
                    serial_number = f"CERT-{clean_company_name.replace(' ', '-')[:8]}-{user.id:04d}-{random.randint(1000, 9999)}"
                    requests.post(f'{_BC_URL}/certificates/create/', json={'user_id': user.id, 'serial_number': serial_number, 'issuer': 'ContractGuard CA', 'valid_days': 365}, headers=authorization_headers, timeout=10)
                except Exception as cert_err:
                    print(f'Warning: Failed to automatically issue certificate: {str(cert_err)}')
            else:
                error_msg = resp.json().get('error', 'Unknown blockchain error')
                user.status = 'ERROR'
                data_repository.save_user_for_register_user(user)
                return ServiceResult({'error': f'Lỗi lưu blockchain: {error_msg}'}, status=400)
        except Exception as e:
            user.status = 'ERROR'
            data_repository.save_user_for_register_user(user)
            return ServiceResult({'error': f'Không kết nối được dịch vụ blockchain: {str(e)}'}, status=500)
        return ServiceResult({'status': 'success', 'user_id': user.id, 'username': user.username, 'email': user.email, 'company_name': company.company_name, 'role_name': role.role_name, 'tx_hash': user.tx_hash, 'block_number': user.block_number, 'block_hash': user.block_hash})
    except Exception as e:
        return ServiceResult({'error': str(e)}, status=500)

def companies_list(data):
    """GET: Get list of all companies."""
    from contracts.models import Company
    companies = data_repository.order_by_company_for_companies_list()
    seen = set()
    data = []
    for c in companies:
        key = (c.tax_code or c.company_name).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        data.append({'id': c.id, 'company_name': c.company_name, 'tax_code': c.tax_code, 'status': c.status, 'tx_hash': c.tx_hash, 'block_number': c.block_number, 'block_hash': c.block_hash})
    return ServiceResult(data)

def users_list(data):
    """GET: Get list of all users."""
    from contracts.models import User
    users = data_repository.order_by_user_for_users_list()
    data = []
    for u in users:
        data.append({'id': u.id, 'username': u.username, 'email': u.email, 'company_name': u.company.company_name if u.company else 'N/A', 'role_name': u.role.role_name if u.role else 'N/A', 'status': u.status, 'tx_hash': u.tx_hash, 'block_number': u.block_number, 'block_hash': u.block_hash})
    return ServiceResult(data)

def roles_list(data):
    """GET: Get list of all roles."""
    from contracts.models import Role
    data_repository.get_or_create_role_for_roles_list()
    roles = data_repository.order_by_role_for_roles_list()
    data = [{'id': r.id, 'role_name': r.role_name} for r in roles]
    return ServiceResult(data)

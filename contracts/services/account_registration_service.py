from contracts.repositories import account_registration_repository as workflow_repository
'Account registration policy and persistence, independent of session handling.'
import os
_BC_URL = os.environ.get('BLOCKCHAIN_SERVICE_URL', 'http://blockchain-service:8000')

def register_account(method, form, authorization_headers):
    error = None
    if method == 'POST':
        username = form.get('username')
        email = form.get('email')
        password = form.get('password')
        company_id = form.get('company_id')
        role_id = form.get('role_id')
        if not username or not email or (not password) or (not company_id) or (not role_id):
            error = 'Vui lòng điền đầy đủ các trường thông tin.'
        elif workflow_repository.exists_user_for_register_account(username):
            error = 'Tên đăng nhập đã tồn tại.'
        elif workflow_repository.email_already_registered(email):
            error = 'Email đã được sử dụng.'
        else:
            try:
                company = workflow_repository.get_company_for_register_account(company_id)
                role = workflow_repository.get_role_for_register_account(role_id)
                user = workflow_repository.create_user_user_for_register_account(company, email, password, role, username)
                try:
                    import requests
                    resp = requests.post(f'{_BC_URL}/user/register/', json={'user_id': user.id, 'username': username, 'company_id': company.id, 'role': role.role_name, 'status': 'ACTIVE', 'sender': 'System'}, headers=authorization_headers, timeout=15)
                    if resp.status_code == 200:
                        data = resp.json()
                        user.tx_hash = data.get('tx_hash')
                        user.block_number = data.get('block_number')
                        user.block_hash = data.get('block_hash')
                        workflow_repository.save_record_for_register_account(user)
                        try:
                            import random
                            clean_company_name = ''.join((x for x in company.company_name.upper() if x.isalnum() or x == ' '))
                            serial_number = f"CERT-{clean_company_name.replace(' ', '-')[:8]}-{user.id:04d}-{random.randint(1000, 9999)}"
                            requests.post(f'{_BC_URL}/certificates/create/', json={'user_id': user.id, 'serial_number': serial_number, 'issuer': 'ContractGuard CA', 'valid_days': 365}, headers=authorization_headers, timeout=10)
                        except Exception as cert_err:
                            print(f'Warning: Failed to automatically issue certificate: {str(cert_err)}')
                    else:
                        user.status = 'ERROR'
                        workflow_repository.save_record_for_register_account(user)
                except Exception as e:
                    print(f'Warning: Failed to register user on blockchain: {str(e)}')
                    user.status = 'ERROR'
                    workflow_repository.save_record_for_register_account(user)
                return (user, None)
            except Exception as e:
                error = f'Lỗi đăng ký: {str(e)}'
    return (None, error)

def ensure_manager_role():
    return workflow_repository.get_or_create_role_for_ensure_manager_role()

def registration_context(error=None):
    return dict(companies=workflow_repository.order_by_company_for_registration_context(), roles=workflow_repository.order_by_role_for_registration_context(), error=error)

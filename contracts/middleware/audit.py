import re
from ..models import AuditLog, Contract

class UserActionAuditMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Process request to let authentication happen first
        response = self.get_response(request)

        # Only audit authenticated users
        if hasattr(request, 'user') and request.user and request.user.is_authenticated:
            path = request.path
            
            # Skip noise (static files, stats API poll, internal assets, dynamic pings)
            if (path.startswith('/static/') or 
                path.startswith('/media/') or 
                path == '/api/admin/stats/' or 
                'favicon' in path):
                return response

            # Only log successful or client-redirect actions (status codes < 400)
            if response.status_code >= 400:
                return response

            action = None
            contract_obj = None

            # Pattern matching for specific endpoints to make them human-readable
            # 1. Login/Logout
            if path == '/login/' and request.method == 'POST':
                action = "Logged in successfully"
            elif path == '/logout/':
                action = "Logged out"
            
            # 2. Admin operations
            elif path == '/admin-dashboard/':
                action = "Accessed custom Admin Dashboard"
            elif path.startswith('/api/admin/users/') and '/delete/' in path:
                action = f"Admin request to delete user (ID: {path.split('/')[-3]})"
            elif path.startswith('/api/admin/users/') and '/retry/' in path:
                action = f"Admin request to retry anchoring user (ID: {path.split('/')[-3]})"
            elif path == '/api/admin/config/ocr/' and request.method == 'POST':
                action = "Admin changed system OCR configuration"

            # 3. Contract operations
            elif path == '/api/contracts/' or path == '/api/contracts/list/':
                if request.method == 'POST':
                    action = "Uploaded/Created a new contract"
            elif '/analyze/' in path:
                # Extract contract ID
                m = re.search(r'/contracts/(\d+)/analyze/', path)
                if not m:
                    m = re.search(r'/api/contracts/(\d+)/analyze/', path)
                if m:
                    try:
                        contract_obj = Contract.objects.get(id=int(m.group(1)))
                        action = f"Triggered AI analysis for contract: {contract_obj.contract_code}"
                    except Exception:
                        action = f"Triggered AI analysis for contract (ID: {m.group(1)})"
            elif '/review/' in path:
                m = re.search(r'/contracts/(\d+)/review/', path)
                if not m:
                    m = re.search(r'/api/contracts/(\d+)/review/', path)
                if m:
                    try:
                        contract_obj = Contract.objects.get(id=int(m.group(1)))
                        action = f"Submitted expert review for contract: {contract_obj.contract_code}"
                    except Exception:
                        action = f"Submitted expert review for contract (ID: {m.group(1)})"
            elif '/detail/' in path:
                m = re.search(r'/contracts/(\d+)/detail/', path)
                if not m:
                    m = re.search(r'/api/contracts/(\d+)/detail/', path)
                if m:
                    try:
                        contract_obj = Contract.objects.get(id=int(m.group(1)))
                        action = f"Viewed contract details: {contract_obj.contract_code}"
                    except Exception:
                        pass

            # 4. Identity registry operations
            elif path == '/identity/':
                action = "Accessed Identity Registry"
            elif path == '/api/companies/' and request.method == 'POST':
                action = "Registered a new company node"
            elif path == '/api/register-user/' and request.method == 'POST':
                action = "Registered a new decentralized staff identity"

            # 5. Workflow operations
            elif path == '/workflows/':
                action = "Accessed Workflow Board"
            elif '/workflows/detail/' in path:
                action = "Viewed workflow detail page"
            elif '/approve/' in path:
                action = "Approved/Rejected workflow step"

            # Fallback for any other write or page visit to satisfy "lưu tất cả các hành động"
            if not action:
                if request.method in ('POST', 'PUT', 'DELETE', 'PATCH'):
                    action = f"Performed {request.method} request on {path}"
                elif request.method == 'GET' and not path.startswith('/api/') and not path.startswith('/admin/'):
                    action = f"Visited page: {path}"

            # Save the audit log if action identified
            if action:
                try:
                    # Resolve client IP address
                    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
                    if x_forwarded_for:
                        ip = x_forwarded_for.split(',')[0]
                    else:
                        ip = request.META.get('REMOTE_ADDR')

                    # Extract payload safely
                    payload_data = None
                    import json
                    if request.method in ('POST', 'PUT', 'DELETE', 'PATCH'):
                        try:
                            # Safely check content type and body
                            if hasattr(request, 'content_type') and request.content_type == 'application/json' and request.body:
                                body_json = json.loads(request.body.decode('utf-8', errors='ignore'))
                                if isinstance(body_json, dict):
                                    for key in ('password', 'password_hash', 'secret'):
                                        if key in body_json:
                                            body_json[key] = '******'
                                payload_data = json.dumps(body_json)
                            else:
                                post_dict = request.POST.copy()
                                for key in ('password', 'password_hash', 'secret'):
                                    if key in post_dict:
                                        post_dict[key] = '******'
                                if post_dict:
                                    payload_data = json.dumps(dict(post_dict))
                        except Exception:
                            try:
                                post_dict = request.POST.copy()
                                for key in ('password', 'password_hash', 'secret'):
                                    if key in post_dict:
                                        post_dict[key] = '******'
                                if post_dict:
                                    payload_data = json.dumps(dict(post_dict))
                            except Exception:
                                pass
                    elif request.method == 'GET' and request.GET:
                        try:
                            payload_data = json.dumps(request.GET.dict())
                        except Exception:
                            pass

                    AuditLog.objects.create(
                        user=request.user,
                        contract=contract_obj,
                        action=action[:100],  # Ensure length limit is respected
                        ip_address=ip,
                        payload=payload_data[:4000] if payload_data else None
                    )
                except Exception as e:
                    # Never block requests due to logging errors
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error writing audit log in middleware: {e}")

        return response

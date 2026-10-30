from contracts.services import RiskService
from contracts.services.results import ServiceResult
risk_service = RiskService()

def risks_list(method, data, form):
    """
    GET: Get list of all master risks.
    POST: Create a new master risk.
    """
    if method == 'GET':
        try:
            risks = risk_service.list_all_risks()
            return ServiceResult(risks)
        except Exception as e:
            return ServiceResult({'error': str(e)}, status=500)
    elif method == 'POST':
        try:
            body = data if data else {}
            name = body.get('risk_name')
            description = body.get('description', '')
            severity_level = body.get('severity_level', 'MEDIUM')
        except (ValueError, KeyError, TypeError):
            name = form.get('risk_name')
            description = form.get('description', '')
            severity_level = form.get('severity_level', 'MEDIUM')
        if not name:
            name = form.get('risk_name')
            description = form.get('description', '') or description
            severity_level = form.get('severity_level', 'MEDIUM') or severity_level
        try:
            risk = risk_service.create_new_risk(name=name, description=description, severity_level=severity_level)
            return ServiceResult({'success': True, 'risk_id': risk.id, 'risk_name': risk.rule_name, 'severity_level': risk.severity})
        except Exception as e:
            return ServiceResult({'error': str(e)}, status=400)
    return ServiceResult({'error': 'Method not allowed.'}, status=405)

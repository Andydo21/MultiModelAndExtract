from contracts.repositories import risk_data_repository as data_repository
from contracts.repositories import RiskRepository, AuditLogRepository

class RiskService:

    def __init__(self):
        self.risk_repo = RiskRepository()
        self.audit_repo = AuditLogRepository()

    def list_all_risks(self):
        risks = self.risk_repo.get_all_risks()
        result = []
        for r in risks:
            from contracts.models import RiskFinding, Contract
            contract_ids = data_repository.distinct_risk_finding_for_list_all_risks(r)
            contracts = data_repository.only_contract_for_list_all_risks(contract_ids)
            contracts_list = [{'id': c.id, 'title': c.title, 'contract_code': c.contract_code} for c in contracts]
            result.append({'id': r.id, 'risk_name': r.rule_name, 'risk_code': r.rule_code, 'description': r.description, 'severity_level': r.severity, 'contracts': contracts_list})
        return result

    def create_new_risk(self, name, description, severity_level):
        if not name:
            raise ValueError('Risk name is required.')
        if severity_level not in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
            raise ValueError('Invalid severity level. Must be LOW, MEDIUM, HIGH, or CRITICAL.')
        risk = self.risk_repo.create_risk(name, description, severity_level)
        admin_user = data_repository.first_user_for_create_new_risk()
        self.audit_repo.log_action(admin_user, 'RISK_CREATED', 'Risk', risk.id)
        return risk

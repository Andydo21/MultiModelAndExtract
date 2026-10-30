from contracts.repositories import analysis_history_data_repository as data_repository
from contracts.repositories import AIAnalysisRepository

class AnalysisHistoryService:

    def __init__(self):
        self.analysis_repo = AIAnalysisRepository()

    def list_all_analyses(self, company=None):
        if company:
            from contracts.models import AIAnalysis
            analyses = data_repository.order_by_ai_analysis_for_list_all_analyses(company)
        else:
            analyses = self.analysis_repo.get_all_analyses()
        result = []
        for a in analyses:
            risk_names_seen = set()
            findings = []
            for f in data_repository.all_findings_for_list_all_analyses(a):
                if f.rule.rule_name not in risk_names_seen:
                    risk_names_seen.add(f.rule.rule_name)
                    findings.append({'risk_name': f.rule.rule_name, 'risk_level': f.risk_level, 'severity_level': f.rule.severity})
            score = int(a.overall_score) if a.overall_score is not None else 0
            if score >= 80:
                risk_label = 'HIGH'
            elif score >= 50:
                risk_label = 'MEDIUM'
            else:
                risk_label = 'LOW'
            result.append({'id': a.id, 'contract_id': a.version.contract.id, 'contract_code': a.version.contract.contract_code, 'contract_title': a.version.contract.title, 'contract_status': a.version.contract.status, 'model_name': a.model_name, 'overall_score': score, 'risk_label': risk_label, 'summary': a.summary, 'findings_count': data_repository.count_findings_for_list_all_analyses(a), 'findings_preview': findings[:3], 'created_at': a.created_at.strftime('%Y-%m-%d %H:%M')})
        return result

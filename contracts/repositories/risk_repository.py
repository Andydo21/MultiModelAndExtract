from contracts.models import Contract, ContractVersion, ContractFile, RiskRule, AIAnalysis, RiskFinding, Review, AuditLog
from ai_extract.models import Clause
import os


class RiskRepository:
    @staticmethod
    def get_all_risks():
        return RiskRule.objects.all().order_by('rule_name')
        
    @staticmethod
    def get_risk_by_id(risk_id):
        try:
            return RiskRule.objects.get(id=risk_id)
        except RiskRule.DoesNotExist:
            return None
            
    @staticmethod
    def create_risk(name, description, severity_level):
        code = name.upper().replace(" ", "_").replace("-", "_")
        return RiskRule.objects.create(
            rule_code=code,
            rule_name=name,
            description=description,
            severity=severity_level
        )

    @staticmethod
    def get_or_create_risk(name, defaults):
        name_clean = name.strip()
        # Look up case-insensitively first to avoid duplicate risk rules
        existing = RiskRule.objects.filter(rule_name__iexact=name_clean).first()
        if existing:
            return existing, False
            
        mapped_defaults = {}
        for k, v in defaults.items():
            if k == 'severity_level':
                mapped_defaults['severity'] = v
            else:
                mapped_defaults[k] = v
        if 'rule_code' not in mapped_defaults:
            # Clean up non-alphanumeric chars for clean code
            import re
            cleaned_code = re.sub(r'[^A-Z0-9_]', '', name_clean.upper().replace(" ", "_").replace("-", "_"))
            mapped_defaults['rule_code'] = cleaned_code
        return RiskRule.objects.get_or_create(
            rule_name=name_clean,
            defaults=mapped_defaults
        )

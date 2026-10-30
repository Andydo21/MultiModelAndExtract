"""Database operations for analysis history workflows."""
from contracts.models import AIAnalysis
from contracts.models import User

def order_by_ai_analysis_for_list_all_analyses(company):
    return AIAnalysis.objects.filter(version__contract__company=company).order_by('-created_at')


def all_findings_for_list_all_analyses(a):
    return a.findings.all()


def count_findings_for_list_all_analyses(a):
    return a.findings.count()

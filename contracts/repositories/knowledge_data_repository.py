"""Version-bound case queries and locked revision updates."""
from django.core.exceptions import ObjectDoesNotExist
from contracts.models import Contract, ContractVersion, ContractKnowledgeCase, AuditLog


def get_record(records, **filters):
    if hasattr(records, '_default_manager'):
        records = records._default_manager
    return records.get(**filters)

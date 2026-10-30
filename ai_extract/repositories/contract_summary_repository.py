from decimal import Decimal
from contracts.models import ContractVersion
from ai_extract.models import ContractSummary, Clause, ExtractedEntity


class ContractSummaryRepository:
    @staticmethod
    def get_by_version(version: ContractVersion) -> ContractSummary:
        try:
            return ContractSummary.objects.get(version=version)
        except ContractSummary.DoesNotExist:
            return None

    @staticmethod
    def create_or_update(version: ContractVersion, summary: str, model_id: str = "") -> ContractSummary:
        obj, _ = ContractSummary.objects.update_or_create(
            version=version,
            defaults={
                "summary": summary,
                **({"model_id": model_id} if model_id else {}),
            },
        )
        return obj

    @staticmethod
    def get_all():
        return (
            ContractSummary.objects.select_related(
                "version__contract"
            ).order_by("-created_at")
        )

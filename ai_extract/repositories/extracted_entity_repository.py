from decimal import Decimal
from contracts.models import ContractVersion
from ai_extract.models import ContractSummary, Clause, ExtractedEntity


class ExtractedEntityRepository:
    """
    Wraps contracts.ExtractedEntity.
    Entity extraction results from Kaggle AI are stored in the same table
    as rule-based entities (contracts_extractedentity), distinguished by entity_type.

    Kaggle AI entity_type conventions:
      COMPANY_NAME, TAX_CODE, CONTRACT_VALUE, DATE_EFFECTIVE, DATE_EXPIRE
    """

    # Confidence score assigned to AI-extracted entities
    AI_CONFIDENCE = Decimal("0.85")

    @staticmethod
    def get_by_clause(clause: Clause):
        return ExtractedEntity.objects.filter(clause=clause)

    @staticmethod
    def get_by_version(version: ContractVersion):
        return ExtractedEntity.objects.filter(
            clause__version=version
        ).select_related("clause").order_by("clause__id", "entity_type")

    @staticmethod
    def save_from_kaggle(clause: Clause, raw_entities: dict) -> list:
        """
        Persist every non-empty key from raw_entities into ExtractedEntity.
        Keys → entity_type mapping (both snake_case and UPPER_CASE supported):
          COMPANY_NAME / company_name  → entity_type="COMPANY_NAME"
          TAX_CODE / tax_code          → entity_type="TAX_CODE"
          CONTRACT_VALUE / contract_value → entity_type="CONTRACT_VALUE"
          DATE_EFFECTIVE / date_effective → entity_type="DATE_EFFECTIVE"
          DATE_EXPIRE / date_expire    → entity_type="DATE_EXPIRE"
          (any other key)              → entity_type=key.upper()

        Uses get_or_create so re-running is idempotent.

        Returns: list of ExtractedEntity objects saved.
        """
        saved = []
        # Normalise keys: try UPPER then snake
        normalised = {}
        for k, v in raw_entities.items():
            normalised[k.upper()] = v

        for entity_type, entity_value in normalised.items():
            if not entity_value:
                continue
            obj, _ = ExtractedEntity.objects.get_or_create(
                clause=clause,
                entity_type=entity_type,
                entity_value=str(entity_value),
                defaults={
                    "normalized_value": str(entity_value).upper(),
                    "confidence_score": ExtractedEntityRepository.AI_CONFIDENCE,
                },
            )
            saved.append(obj)

        return saved

    @staticmethod
    def delete_kaggle_entities_for_version(version: ContractVersion) -> int:
        """Delete AI-sourced entities for all clauses in the version."""
        kaggle_types = [
            "COMPANY_NAME", "TAX_CODE", "CONTRACT_VALUE",
            "DATE_EFFECTIVE", "DATE_EXPIRE",
        ]
        count, _ = ExtractedEntity.objects.filter(
            clause__version=version,
            entity_type__in=kaggle_types,
        ).delete()
        return count

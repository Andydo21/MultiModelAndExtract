from django.contrib import admin
from .models import (
    ContractSummary,
    Clause,
    ExtractedEntity,
    LayoutExtractionRun,
    LayoutExtractionPage,
    LayoutExtractionBlock,
)


@admin.register(ContractSummary)
class ContractSummaryAdmin(admin.ModelAdmin):
    list_display = ("__str__", "model_id", "created_at", "updated_at")
    list_filter = ("model_id",)
    search_fields = (
        "version__contract__contract_code",
        "version__contract__title",
        "summary",
    )
    readonly_fields = ("created_at", "updated_at")
    ordering = ("-created_at",)


@admin.register(Clause)
class ClauseAdmin(admin.ModelAdmin):
    list_display = ("clause_title", "version", "clause_type")
    list_filter = ("clause_type", "version__contract")
    search_fields = ("clause_title", "clause_content")


@admin.register(ExtractedEntity)
class ExtractedEntityAdmin(admin.ModelAdmin):
    list_display = ("entity_type", "entity_value", "clause", "confidence_score")
    list_filter = ("entity_type",)
    search_fields = ("entity_value", "clause__clause_title")


@admin.register(LayoutExtractionRun)
class LayoutExtractionRunAdmin(admin.ModelAdmin):
    list_display = (
        "version", "pipeline_name", "status", "total_pages", "total_blocks",
        "dense_index_status", "dense_points_count", "colpali_index_status",
        "colpali_points_count", "started_at",
    )
    list_filter = ("status", "dense_index_status", "colpali_index_status", "pipeline_name")
    readonly_fields = ("started_at", "completed_at")


@admin.register(LayoutExtractionPage)
class LayoutExtractionPageAdmin(admin.ModelAdmin):
    list_display = ("run", "page_number", "width", "height")
    list_filter = ("run__status",)


@admin.register(LayoutExtractionBlock)
class LayoutExtractionBlockAdmin(admin.ModelAdmin):
    list_display = ("page", "block_order", "layout_type", "confidence")
    list_filter = ("layout_type",)

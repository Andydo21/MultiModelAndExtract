from django.urls import path
from . import views
from .views import layout_views

urlpatterns = [
    path('contracts/<int:contract_id>/reset-graph/',layout_views.api_reset_graph,name='reset_document_graph'),
    path('contracts/<int:contract_id>/extraction-status/',
         layout_views.api_extraction_status,name='extraction_status'),
    path('contracts/<int:contract_id>/resolve-entities/',
         layout_views.api_resolve_entities,name='resolve_document_entities'),
    path(
        "contracts/<int:contract_id>/layout-extract/",
        layout_views.api_run_layout_extraction,
        name="layout_extract",
    ),
    # ── Summarize ──────────────────────────────────────────────────────────
    # POST  /api/ai/contracts/<id>/summarize/   → run summarize & save
    path(
        "contracts/<int:contract_id>/summarize/",
        views.api_summarize_contract,
        name="ai_summarize_contract",
    ),
    # GET   /api/ai/contracts/<id>/summary/     → retrieve saved summary
    path(
        "contracts/<int:contract_id>/summary/",
        views.api_get_summary,
        name="ai_get_summary",
    ),

    # ── Extract Entities ────────────────────────────────────────────────────
    # POST  /api/ai/contracts/<id>/extract-entities/  → run per-clause extraction & save
    path(
        "contracts/<int:contract_id>/extract-entities/",
        views.api_extract_entities,
        name="ai_extract_entities",
    ),
    # GET   /api/ai/contracts/<id>/entities/          → list saved entities
    path(
        "contracts/<int:contract_id>/entities/",
        views.api_get_entities,
        name="ai_get_entities",
    ),

    # ── Free-text extraction (no contract binding) ──────────────────────────
    # POST  /api/ai/extract-from-text/
    path(
        "extract-from-text/",
        views.api_extract_from_text,
        name="ai_extract_from_text",
    ),

    # ── Extract Clauses ─────────────────────────────────────────────────────
    # POST  /api/ai/contracts/<id>/extract-clauses/  → AI clause splitting & save
    path(
        "contracts/<int:contract_id>/extract-clauses/",
        views.api_extract_clauses,
        name="ai_extract_clauses",
    ),
    # GET   /api/ai/contracts/<id>/clauses/           → list saved clauses
    path(
        "contracts/<int:contract_id>/clauses/",
        views.api_get_clauses,
        name="ai_get_clauses",
    ),

    # ── Free-text extraction (no contract binding) ──────────────────────────
]

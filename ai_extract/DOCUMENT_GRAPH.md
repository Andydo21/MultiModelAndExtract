# A3 source and layout graph

A3: all pages -> real Surya OCR/layout and supplied reading/containment relations
-> GLiNER review of those existing relations -> entity extraction
-> same-name context review and physical entity-node compaction -> indexing -> atomic
Neo4j publication. Qwen is used by A2 for answers, never for A3 identity decisions.

The current document's graph is not an input to its analysis. GLiNER extracts
untyped entities and source-grounded relations from each Surya region. Its one
internal extraction tag is `entity`; PRODUCT/PERSON/TAX_CODE subtypes are not
saved on graph nodes. Same-name occurrences are reviewed in their document
context by GLiNER2. Accepted occurrences share one graph entity node; their
MENTIONS links retain every source ID, character span and occurrence key. No
additional canonical registry node or RESOLVES_TO layer is created. Unknown
or conflicting occurrences stay separate. Names use only Unicode/case/whitespace
normalization; different names are not merge candidates.

Structure uses model labels and explicit heading levels. Unknown levels remain
unknown. Model reading order and one text-only OCR pass inside images are kept.
No reference-number parser or generative excerpt-pair stage runs in ingestion.

## Relationships

- CONTAINS: source-backed parent/child structure and image contents.
- HAS_EVIDENCE: nodes linked to original source text and location.
- NEXT: original Surya reading order, annotated with GLiNER supported/disputed/
  unresolved review. GLiNER does not add, reverse or rebuild the reading order.
- HAS_VISUAL / APPEARS_IN: source text/blocks linked to page images.
- HAS_EMBEDDING: figure block linked to its acknowledged Qdrant point.
- HAS_ROW / HAS_CELL: native table structure and source values.
- MENTIONS: source text contains a GLiNER entity span.
- Merged entities: one entity node per accepted context group; source occurrences
  remain on MENTIONS links, and semantic relations are redirected to the shared node.
- Typed semantic relationships: GLiNER proposals with source quotations.
- The ordinary source extraction endpoint returns entities and optional
  relationships between their exact source spans from each Surya region.
  It does not replace Surya's reading order. Existing Surya NEXT/CONTAINS
  proposals go through a separate GLiNER review before graph publication.
- LEFT_OF / ABOVE / NEAR: supplementary geometry, not semantic support.

All semantic model proposals remain unreviewed after source validation. Each
snapshot is isolated by company, contract, version, run and source fingerprint.

A2 retrieves text and images, expands stored graph relationships and passes
prioritized, evidence-bound context to Qwen. Geometry has a small supplementary
budget. Source excerpts remain the evidence; graph links help navigate them.

## Reprocessing

    python manage.py sync_document_graph --run-id ID --analyze-relations

Graph publication requires a matching completed pregraph analysis. The retired
graph-first identity resolver raises an error. The existing identity worker
reprocesses legacy queued jobs using the same whole-document preparation before
replacing a graph.

For a fresh stability run, use the project `layout-extract` HTTP endpoint with
`force_ocr: true`. Use `defer_colpali: true` to postpone visual embeddings. The
`reset-graph` endpoint verifies deletion within one authorized contract and checks
that unrelated graph nodes are unchanged. It preserves source PDFs and run logs.

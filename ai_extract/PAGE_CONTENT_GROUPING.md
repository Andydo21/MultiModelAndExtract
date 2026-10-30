# Column regions and paragraph continuation

`document_structure_service.build_document_structure` calls
`page_content_grouping_service.annotate_page_content` before assigning headings.

Each page/image is independent. Repeated body boxes establish persistent horizontal
whitespace gutters; wide headings, figures and tables do not bridge body columns.
Source-aligned Granite order takes precedence. Without model order, detected
columns are read top-to-bottom within bands separated by spanning elements.
Unresolved model order remains unresolved, rather than being silently replaced.

Adjacent body blocks can share `paragraph_group_id` only when their column,
downward direction, left alignment, horizontal overlap, adaptive vertical gap,
available font attributes and unfinished-text continuation agree. Headings,
lists, tables, figures, page changes and uncertain order prevent grouping.
Spacing estimates use the median bbox height per extracted text line within
the column. Missing OCR line boundaries make this estimate less reliable.
This is a conservative heuristic, not semantic model verification; uncertain
and non-lowercase continuations remain separate, including most cross-page
continuations and scripts without letter case.

Original text, bbox, block IDs and source offsets are retained. Source chunks are
not concatenated. Multi-block groups with the same structure owner are projected
as `paragraph_group` nodes with unreviewed `CONTAINS` links to original blocks
and `HAS_EVIDENCE` links to their leaves. Singleton groups add no graph nodes.

`NEXT` remains reading order, not paragraph membership. `LEFT_OF`, `ABOVE` and
`NEAR` remain independent geometric relations. This change does not prune those
geometric edges. Existing saved runs require rebuilding their structure and
graph; changing code does not retroactively alter stored results.

Validation: regression tests cover two columns, downward continuation, large
gaps, punctuation, new list items, heading barriers, spanning headings,
font changes, uncertain model order, missing bbox and original-source graph links.
These tests verify behavior, not extraction accuracy on a labeled corpus.

## Verified application — 2026-10-12

45 focused tests passed (grouping, layout graph, contract structure, generic
document ingestion and structure visualization). The visualization tests also
exercise the unavailable-Neo4j fallback in isolated test settings.

Applied to contract 120/version 151/run 8 from cached Granite observations.
Source leaf IDs, exact text, bbox, hierarchy and vector-relevant fields were
asserted unchanged before writing. Neo4j sync returned ready: 1,110 nodes and
11,486 edges. Two-column annotations cover 350 leaves; one-column annotations
cover 24 leaves. No block pair met the conservative continuation conditions,
so no paragraph group nodes were added for this document. This is not an
accuracy claim. No models reran and no vectors were reembedded.

Web and knowledge worker restarted. HTTP navigation returned 200 (unauthenticated
navigation can resolve to login; this check is not a browser visual inspection).
Backup: `scratch/run8-before-page-grouping.json`.
Report: `scratch/page-content-grouping-report.json`.

# A3 extraction and contract AI

PDF pages are rendered to PNG before Surya layout/OCR. PDF text layers are never read.
Surya reads text once inside page-level image regions; nested pictures are not scanned further.
Each source region remains one chunk with its text, page and bbox.
Unreadably small text inside images (below 6 preview pixels on a 1755px-high page,
scaled with preview resolution) is ignored before OCR, not reported as a failure.
`SURYA_VISUAL_MIN_TEXT_HEIGHT=0` disables this filter. Page text is unaffected.
Visual OCR preserves its first two recognition attempts. Only a crop still looping
gets one final contrast/framing rescue, with a smaller repetition budget. A failed
image layout is retried once with source-coordinate remapping; healthy siblings
are not repeated. Unresolved text remains flagged, rather than counted as complete.

The active flow is Surya/native Office parsers ? source structure and GLiNER ? Neo4j ? BGE/ColPali indexes.
GLiNER-RelEx multilingual v1.0 extracts typed spans and allowlisted entity relation proposals independently per source region. Code validates IDs, character spans, same-region endpoints, quotes and scores before Neo4j projection; semantic links remain unreviewed.
Qwen is used for A2 answers and expert submissions, never A3 relationship creation.

Granite and LayoutLM runtime services, HTTP endpoints and training launcher were removed on 2026-10-14.
Historical observations remain readable in saved runs; no extraction data or graph snapshots were deleted.

- `surya_inference_service.py`: model prompts, layout/OCR and one image-text pass.
- `layout_extractor.py`, `office_parser.py`: source normalization and media.
- `document_structure_service.py`: project source/model labels and supplied hierarchy levels; resolve GLiNER references.
- `document_graph_service.py`: graph projection, source links and graph context.
- `graph_visualization_service.py`: persisted graph and source previews.
- `layout_service.py`: per-page persistence, parallel ColPali and final dense indexing.

ColPali visual indexing is figure-only: after Surya blocks are persisted, crop each
`figure` from the rendered page into PNG and embed it. Text and table blocks never
fall back to a full-page visual embedding. The visual branch overlaps page structure
and entity processing; pages without figures are skipped normally.

Qdrant stores one multivector point per figure, with run, page, block ID, page bbox
and crop path. After an acknowledged write, Neo4j projects
`layout_block(kind=figure) -[:HAS_EMBEDDING]-> visual_embedding`, whose properties
identify the Qdrant collection and point. Vector arrays remain in Qdrant.
Final graph sync follows visual retries. A2 retrieves the matching crop and source
block; legacy full-page points are excluded. Re-extract/reindex existing documents
to create figure vectors; no automatic data migration occurs.

Reading order uses Surya 2's native layout/OCR sequence, not a custom prompt or
the completion order of parallel OCR calls. `native_reading_order()` preserves
the model rank, source and scope; each figure's inner text has a separate scope.
The hierarchy sorts source records by those ranks before assigning headings.
Neo4j creates `NEXT` only between distinct, unambiguous model ranks in the same
page/scope. A page sequence does not detour through inner-image text, and image
text sequences do not cross between figures. Missing/duplicate ranks retain
source text but suppress that scope's `NEXT`. Rank validity is a structural
check, not a guarantee of correct visual reading order on every document.
Older OCR caches are invalidated by the revised policy on the next extraction;
existing persisted runs/graphs are not rewritten automatically.

A3 uses Surya for OCR/layout and GLiNER for source-region entities and typed
relations. Code projects their grounded outputs into Neo4j. BGE and ColPali only
create retrieval indexes. A3 does not call Qwen or judge pairs of excerpts.

A2 retrieves authorized evidence, expands existing graph links, and supplies
source text plus prioritized graph context to Qwen for cited answers. Similarity
and repeated wording never create new graph relationships.

## Source-grounded captions, references and supplementary geometry — 2026-10-14

Final suite: **410 discovered, 408 passed, 2 optional integration skips**.
Log: `scratch/source-relations-full-tests.log`. Tests cover heading continuity,
unresolved order at root, matching own visual labels, missing/conflicting labels,
ambiguous figures/duplicate captions, column separation, source preservation,
idempotence, conditional candidate expansion and bounded supplementary geometry.
The prior test that borrowed a figure number from proximity was updated to
require an unreviewed candidate instead. This intentionally changes that policy.

Fixed caption normalization (`figurecaption` → `figure_caption` and table
equivalent). A3 makes no Qwen calls for these rules. `HAS_EVIDENCE` caption links
require independently present source numbering plus unique local caption match;
otherwise `RELATED`/`caption_candidate` remains unreviewed. Cross-page explicit
references link to a uniquely numbered caption in the same version. Source text,
bbox and heading paths were checked unchanged during stored-tree refresh.

Live Neo4j read-back confirmed four latest ready graphs; run 19 gained **13
references + 9 caption candidates**, not 9 verified caption ownership links.
No caption-evidence links were forced onto this document's unlabeled figures.
Report: `scratch/source-relations-live.json`. No model inference or vector writes.

A2 prioritizes hierarchy/table/source edges across documents. Valid selected-source
geometry is supplementary (3 edges normally, up to 8 for visual/layout questions;
at most 20% graph budget, with 15% reserved when geometry exists). Quote, origin,
review status, explicit NEXT meaning and candidate meaning reach the prompt.
Final run-19/page-7 context check: **21 sent edges including 5 supplementary**,
partial under the current graph budget. `scratch/check_source_relations_live.py`
checks this without calling Qwen. This is not measured hallucination reduction
or document accuracy; a live generated-answer evaluation remains separate.

## Local layout relations and A2 context — 2026-10-14

Suite: **400 discovered, 398 passed, 2 optional integration tests skipped**.
Raw log: `scratch/spatial-normalization-full-tests.log`. Regression tests cover
diagonal/far/weakly aligned boxes, intervening blocks, nearest-neighbor sparsity,
input-order independence, and filtering old spatial edges before the A2 budget.

Live Neo4j synchronization rebuilt the four latest ready version graphs from
stored sources. Persisted relation counts were read back and matched the new
projection; report: `scratch/spatial-normalization-live.json`. Run 20: LEFT_OF
172 → 9, ABOVE 401 → 29, NEAR 79 → 30. Run 19: LEFT_OF 1115 → 91, ABOVE
2027 → 113, NEAR 573 → 161. Their hierarchy/source/table/NEXT counts stayed
unchanged. The older run 1 also acquired source/layout edges implemented since
its original snapshot; this is a current-code reprojection, not simply pruning.
No OCR, model inference, embedding or vector updates were performed.

A2 context was built from the actual run 20 Neo4j graph and supplied evidence:
state ready, no LEFT_OF/ABOVE/NEAR in bound generation context. It remained
partial under the existing source/context budgets; this does not claim full
graph delivery or measured answer accuracy. Qwen inference was not invoked.
Web/worker were restarted to load the code; A2/A3 and local runtime health
returned HTTP 200. The existing Granite timeout and Kaggle 404 are independent
of these geometry fixes and were not repaired by graph synchronization.

## Qwen raw JSON output — 2026-10-13

Suite: **344 discovered, 342 passed, 2 optional integration tests skipped**.
Raw test log: `scratch/qwen-output-tests.log`. These tests include model API
fixtures; they are not performance or accuracy measurements. They verify exact
API response preservation, invalid/truncated output visibility, independent
capture for concurrent calls, legacy JSON serialization, citation-repair and
verification stages, history persistence and access checks. Django system check
and JavaScript syntax check passed.

New API/job results include `qwen_outputs`, with the actual Qwen API envelopes
and task/stage labels. The browser shows expandable **JSON output của Qwen**
and a **Tải JSON** button for A2 answers, expert semantic extraction, single
document answers and whole-document review. Original model content is retained
before parsing/rendering, including unsuccessful validation attempts. Prompts,
request headers and credentials are not added to this output capture. Old jobs
remain without fabricated/reconstructed raw responses.

**Live worker/API acceptance passed**, search job 33, after Kaggle became ready
with `max_model_len=32768`. Result: `generation_status=generated`, citations
accepted. Both actual Qwen calls ended with `finish_reason=stop`:

- Answer: **2672 input + 119 output = 2791 tokens**.
- Verification: **2458 input + 45 output = 2503 tokens**.

The API returned both envelopes, and job history retained the same outputs.
This single run took **14612.99 ms** for retrieval and **42966.38 ms** total;
it is not an accuracy benchmark or p95 measurement. Artifacts:
`scratch/qwen-output-live-check.log` and `scratch/qwen-output-live.json`.

## Unified knowledge search and full-answer budgets — 2026-10-13

Current suite: **337 discovered, 335 passed, 2 optional integration tests skipped**.
The suite includes explicit model-protocol fixtures; its runtime is not real
model latency or an accuracy benchmark. Raw output:
`scratch/unified-knowledge-tests.log`. Django system check and browser JavaScript
syntax check passed (template variables substituted for syntax checking).

A2 now uses one authorized text corpus for A3 document leaves, stored expert
questions/answers/notes, semantic units and approved current cases. One dense
lookup, one visual lookup and one BM25 stream feed shared fusion and reranking;
document and expert context expand after ranking. Expert questions cannot by
themselves support an answer; drafts stay marked unreviewed. Tenant and saved
source fingerprints are checked when hydrating each vector hit.

Real backfill indexed **23/23 expert records** through local CUDA BGE into the
existing document text collection. It did not rerun Qwen, OCR or document
extraction. Repeat upserts reuse stable point IDs. Actual artifact:
`scratch/unified-knowledge-backfill.log`.

The pre-upgrade live EVP2 query retrieved both document and expert sources. Streams
reported BM25=50 candidates, dense=60 and ColPali=60; stored expert records=23.
The repeated question unit no longer displaced the saved expert answer.
This individual warm retrieval took **7107.16 ms**, total **7819.58 ms** including
the rejected generation request. It is a functional check, not retrieval
accuracy and not the latency of a successful long generated answer. Artifact:
`scratch/unified-knowledge-live-answer.log`.

Web and worker were restarted and confirmed defaults: output budget **8192
tokens**, up to **96 cited sentences**, source selection limit **24**. Selected
source text is sent without the previous 1200-character crop or 13-source
generation cap. Graph context/conditions have reserved selection slots.
Citation checks and semantic verification remain enabled; incomplete output
(`finish_reason=length`) is not treated as a complete answer.

**Resolved:** after the user restarted the Kaggle model cell, the actual
`/v1/models` response reported `max_model_len=32768`. Full A2 acceptance then
passed against real local BGE/reranking, ColPali and Qwen services.

That live run also exposed a provenance bug: original answers/notes explicitly
marked `[DEMO]` or `Mẫu trả lời demo:` were being hydrated with `is_demo=False`.
Code now propagates that explicit marker to the submission's question, answer
and semantic units. Original content is unchanged. The same indexer refreshed
23 existing expert points so their provenance fingerprints remain current.
Ordinary document questions exclude illustrative demo advice from generation;
expert/demo questions can still retrieve it with its provenance preserved.
The verifier also checks draft/demo attribution and applicability limits.

Final real query, search log 30: **generated**, **citation_valid=true**, no
generation failures. It explains Guidance as PC setup and Runtime as execution
using the robot controller/camera, and explicitly answers the PC question.
The answer cites the primary PDF passages E3-E6 rather than demo advice E2.
Retrieval: **13123.46 ms**; generation/checking after retrieval: approximately
**15811.54 ms**; total: **28935.0 ms**. BM25/dense/ColPali streams were ready
with 50/60/60 candidates. These are one real query's observed times, not p95
or an accuracy benchmark; model verification is not proof of factual truth.
Raw result: `scratch/unified-knowledge-32768-live-answer.log`.

## GPU preference and corpus query optimization — 2026-10-13

Supersedes the automatic CPU handoff described below. At the user's request,
BGE embedding and reranker now stay on CUDA: removed the conservative free-VRAM
threshold and automatic offload. Startup explicitly sets `RAG_TEXT_DEVICE=cuda`;
CUDA absence fails instead of silently selecting CPU. Both models remain cached.
Embedding/rerank use inference mode and bounded batches (8/4). A real CUDA OOM
retries smaller batches down to one, with no source truncation change, item loss
or CPU fallback. Granite defaults to CUDA; structure health reports configured
and actual loaded device separately.

Corpus discovery now prefetches clause/context fallback evidence in bulk.
Completed-run lookup also consumes existing prefetched runs instead of issuing
additional queries for hierarchy, fusion and graph expansion. A database test
verifies that reading all prefetched sources/runs issues zero additional SQL
queries and retains clause/context fallback behavior. No document or graph data
was rewritten.

Final tests: 264 discovered, 263 passed, one optional live Neo4j fixture skipped.
Django system check passed. Web/worker and the local inference process were
restarted. Actual model health showed BGE and reranker both on `cuda:0`, each
loaded exactly once through alternating requests. Before loading Granite, warm
short-request embedding took 23.27/34.86 ms and rerank 34.10/35.55 ms; cold calls
including loading took 13,759.03/6,804.83 ms. Source:
`scratch/optimized-inference-gpu-before-granite.json`.

Granite was called on a 1000×233 crop of the original run-8 first-page image:
HTTP 200, four parsed items, 27,844.41 ms including first use. Actual health:
`ibm-granite/granite-docling-258M`, `loaded=true`, `device=cuda:0`. This is a device
and functionality check, not whole-page extraction or an accuracy measurement.
Source: `scratch/granite-gpu-live.json`. With Granite also resident, alternating
short embedding/rerank requests succeeded in 40.20–98.02 ms and neither text
model was reloaded (`scratch/optimized-inference-live.json`).

Actual A2 retrieval receipt 24: 76 documents scanned, one searched deeply;
dense, ColPali, neural rerank and Neo4j ready; six citations, 26 graph nodes and
74 edges; 17,505.78 ms wall time. This probe overlapped Granite's first-use test
and is not a controlled comparison with the earlier 23,433.94 ms result.

Actual full A2 answer receipt 25: four indexed documents scanned, one deeply
searched; BGE embedding, ColPali embedding and rerank each invoked once; six
citations; all retrieval branches and Neo4j ready. Measured BGE call 99.47 ms,
ColPali query embedding 1,022.39 ms, reranking 24 real candidates 2,133.85 ms.
Generation began at 10.039 s; semantic verification at 22.475 s; total 27.174 s.
Answer status `generated`, citations valid, semantic verification accepted.
There was one answer-generation attempt plus the separate verification call;
the old `generation_call_count=1` field in this probe counts generation attempts
only, not all Qwen calls. Kaggle had become available by these probes and was
actually called. Source: `scratch/optimized-answer-live.json`.

These are measured single-run timings with the stated inputs, not latency
guarantees or a corpus-wide accuracy evaluation. The models/services were left
running. Granite still follows the existing A3 release-after-extraction policy.

## A2/A3 retrieval and warm inference optimization — 2026-10-13

Completed request-local BGE/ColPali query embedding reuse (including cached
failures), shared corpus reranking before graph expansion, offline warm model
cache with CUDA/CPU placement, and persisted job phases shown under the active
question. Existing access filters, source binding, citation checks and Qwen
verification remain in place. No document extraction, graph rewrite, migration
or checkpoint download was performed by this optimization.

Validation: 261 tests discovered; 260 passed, one optional live Neo4j fixture
skipped. Ten new checks cover cross-version ColPali query reuse with distinct
scope filters, failure retry boundaries, one rerank for two documents followed
by graph expansion, authorized progress visibility, generation phase order,
offline warm weights, CUDA/CPU placement and device-aligned inputs. JavaScript
syntax and Django's system check passed. Web and worker were restarted.

Actual local model requests (no model mocks), `scratch/optimized-inference-live.json`:

- BGE first request including cold startup: 15,142.63 ms; next short GPU request:
  24.55 ms. After moving cached BGE to CPU to accommodate reranking: 341.79 ms.
- Reranker first request including load: 7,413.18 ms; subsequent short requests:
  29.67 and 32.42 ms. Inputs contained only two or three short test passages.
- Final health showed BGE on CPU and reranker on CUDA, each with load count 1
  despite alternating requests. Before the offload repair, alternating requests
  reloaded BGE (count 2), adding 6,598.99 ms. The pre-repair sample is retained
  in `scratch/optimized-inference-before-offload.json`.

Actual corpus retrieval receipt 23, `scratch/optimized-search-live.json`:
query `EVP2 Guidance Runtime PC`; 76 visible latest versions scanned, one
document searched deeply; six citation sources; 23,433.94 ms wall time.
Instrumentation counted one actual BGE embedding, one ColPali query-embedding
attempt and one neural rerank. Dense/reranker/graph were ready; graph context
contained 26 nodes and 74 edges. ColPali returned HTTP 404. Separate Kaggle
health checks returned HTTP 404 for both `/v1/models` and `/colpali/health`;
generation was intentionally not requested in this retrieval probe. This is
not an end-to-end Qwen benchmark, an accuracy benchmark, or a measured overall
speedup over the previous pipeline. Multiple-document rerank reuse is verified
by the integration fixture, not by this one-document live result.

Runtime, Surya backend, web, worker, Qdrant and Neo4j were left running.

## Applied multi-column grouping repair — 2026-10-12

Confirmed on the source page and actual graph that Dead data-stack / Dead call-stack paragraphs (blocks 665/667) belonged to 5.1 Optimizations but were attached to 6 Implementation. The source extractor interleaved columns vertically; stored Granite reading_order was not used by the hierarchy builder.

The builder now orders aligned source blocks by Granite's stored reading order. A uniquely overlapping source/model bbox can recover order without changing source text or asserting semantic meaning. Unresolved blocks stay under the document root and do not create NEXT links. Number-only first lines of multi-line headings now retain the full heading title and numeric depth; subsection prefixes can select their existing parent.

Applied to run 8 using cached observations (no OCR/Granite inference, Qwen or image transfer). Rebuilt actual local BGE vectors: 374 points. ColPali retained 14 page vectors. Actual graph: 1,110 nodes, 11,486 edges. Both paragraphs now have HAS_EVIDENCE from 5.1 Optimizations and its parent 5 Trace Tree Optimization, with no links from 6 Implementation. All dense point leaf IDs/node IDs/heading paths matched the new tree. Authenticated page HTML returned 200. Report: `scratch/reading-order-repair-report.json`; backup: `scratch/run8-before-reading-order-fix.json`.

37 focused tests passed, including a model-order multi-column regression and existing contract clause hierarchy checks. The first completion script hit an IndexResult field-name error after vector creation; corrected the script and finalized against existing verified vectors without re-embedding. Remaining limitation: 116 blocks have unresolved reading order and are deliberately not assigned to a section. Model-derived table cells remain unreviewed, including the page-8 cell issue recorded below. This repair validates the reported grouping error, not every semantic relationship in the document.

## Heading-to-content graph navigation — 2026-10-12

Added a snapshot/page-scoped node focus filter and a "Xem các liên kết của node này" action. Overview edge pagination could omit a selected heading's content links; the focus view fetches only incident edges instead of relying on the displayed slice. Verified actual heading 6.1 on page 8: 3 incident edges (1 CONTAINS, 2 HAS_EVIDENCE), authenticated HTML 200. Focused service tests (4) and the live Neo4j scope/page/focus test passed. NEXT remains observed source-block order; CONTAINS denotes hierarchy. Existing hierarchy and multi-column accuracy limitations below remain unresolved; this change does not reclassify content or regenerate vectors.

## Relation correctness audit — 2026-10-12

Audited all stored run-8 graph edges independently against SQL source blocks, page membership, bbox directions/distances, table ownership, node categories, source text and final projection. All 367 source blocks were represented. Found one duplicated layout-block→table CONTAINS on page 12; fixed containment generation and refreshed only the graph (1,118 nodes, 11,738 edges). Source tree/chunks and vector embeddings were retained.

Accuracy findings beyond graph consistency:

- Four body-text lines had been promoted to headings, including "Section 9 and an outlook...", "part of recording LIR..." and a bibliography line "300. ACM New York...". Changed the builder so non-legal numbered headings need heading evidence when model observations are available. A cached-observation rebuild preview removed four nodes and changed 378 leaves to 374. This new tree was NOT applied to the active run: dense heading vectors and leaf IDs must be regenerated together.
- Headings 5.1, 5.2, 6.1 and 6.6 are currently level 1 under the document root. Page text order also interleaves columns (5.2 follows 6.1). Valid source links do not establish correct semantic grouping. Multi-column reading order and numbered hierarchy still require repair.
- Visual spot-check of original page 8 found Granite's last table row has Tag="100" where the original image cell is blank. Source substring validation misses this because "100" occurs elsewhere in the table. All 291 model-derived cells remain unreviewed; row/cell semantic accuracy is NOT certified.

Regression checks: 29 passed (layout graph, generic documents and contract hierarchy). Reports: `scratch/relation-integrity-audit.json`, `scratch/run8-corrected-structure.json` (preview only), and a source-tree backup `scratch/run8-structure-before-relation-audit.json`. No OCR/model rerun or vector reindex was performed. The graph should not be described as fully correct or enterprise-validated.

## Neo4j web visualization — 2026-10-12

Page inspector now embeds its own graph, locked to the URL page number even if a different `graph_page` query parameter is supplied. Both edge endpoints must belong to that page, with shared heading ancestors included as context. Source text from other pages is excluded. Per-type counts now reflect the selected page. Edge pagination interleaves types deterministically rather than sorting alphabetically (which previously showed only APPEARS_IN on the first slice).

Actual page 8 check: 98 related nodes; CONTAINS 6, HAS_EVIDENCE 101, HAS_ROW 7, HAS_CELL 21, NEXT 30, HAS_VISUAL 31, APPEARS_IN 31, LEFT_OF 238, ABOVE 438, NEAR 105. All ten types appeared in the first bounded slice; authenticated page HTML returned 200. Focused service/UI tests: 11 passed. Actual Neo4j fixture test passed for per-page counts, cross-page exclusion, scope isolation and non-overlapping pagination. One isolated negative-ID fixture left by a test variable-name error was cleaned; user snapshots were untouched. Report: `scratch/page-graph-audit.json`.

The graph has a document-wide structural hierarchy, but the current run has no semantic cross-page relation inference (no EXPLAINS, RELATED or MENTIONS).

Added an interactive graph to the contract extraction screen, reading actual Neo4j nodes/edges from the exact company, contract, version, run and source-fingerprint snapshot. The viewer supports page/type filters, bounded edge pagination, category colors, directed arrows, zoom/pan, node dragging and a source/provenance detail panel. Geometry is distinguished from semantic meaning; table proposals retain their unreviewed status. No extraction or vector reindex was triggered by this UI change.

Read-only live checks for contract 120/version 151/run 8 confirmed 1,118 nodes and 11,739 edges. Structural filters matched 2,628 edges; page 8 HAS_CELL matched 21; page 1 ABOVE matched 264. Each graph slice contains at most 60 edges. Authenticated HTML returned 200 and invalid relation filters returned 400. A separate live fixture test passed with pagination, table page filtering, full snapshot scope isolation and cleanup. Result: `scratch/graph-visualization-audit.json`.

Browser screenshot verification was unavailable because the computer-use tool failed to initialize (kernel asset path error). HTML/data, JavaScript syntax and focused service/security tests were checked; appearance has not been screenshot-reviewed.

## Fresh extraction run 8 — 2026-10-12

Contract 120, version 151 was extracted as a new run from the original file using the new pipeline. Historical run 7 was retained. After a Windows/WSL memory allocation failure interrupted the container at page 7, the new run resumed from its six completed pages. No historical run 7 structure was reused.

Final verified result:

- COMPLETED: 14 pages, 367 source blocks.
- Granite returned observations for all 14 pages; 226/367 blocks aligned, 0 failed pages. `model_partial` describes unmatched blocks, not a pending run. Original source text remains available.
- Neo4j: 1,118 nodes, 11,739 edges. Actual relationship counts: CONTAINS 34; HAS_EVIDENCE 1,171; HAS_ROW 34; HAS_CELL 291; NEXT 353; HAS_VISUAL 378; APPEARS_IN 367; LEFT_OF 2,902; ABOVE 5,065; NEAR 1,144. No MENTIONS/RELATED/EXPLAINS edges in this snapshot.
- Qdrant exact scope counts: 378 BGE points, 14 ColPali page points. Every dense point's leaf ID, node ID and heading path matched the final run structure.
- Authenticated Django rendering returned HTTP 200 for the contract extraction screen and page 8 inspector.
- The user explicitly authorized sending page images to their configured Kaggle/ColPali endpoint. Qwen did not choose relations.

Fixed the Docling TableCell API field names discovered on real pages 8 and 12; both were actually inferred again successfully after loading the fix. Verified a real Docling table export/parse roundtrip. Focused graph/visualization checks: 26 tests, 25 passed, 1 opt-in live test skipped. The earlier live Neo4j smoke test also passed.

Resource changes: web runs without Django autoreload; Granite is released before dense indexing. Attempts to stop the unrelated Windows SQL Server services were denied by Windows permissions, so those services remain running. No SQL Server data was deleted.

This is functional/provenance validation, not a labeled accuracy or retrieval-quality benchmark. Model-derived table grids retain their Granite origin and unreviewed status.

Audit details: `scratch/run8-final-audit.json`.

## Granite Docling and source layout graph — 2026-10-12

- Restored only `ibm-granite/granite-docling-258M` to D:; model weights verified (515,093,104 bytes). Existing Surya/BGE/torch were retained.
- Host CUDA runtime started; web and knowledge-worker configured for Granite. Web login and host runtime health returned HTTP 200.
- Actual Granite image inference on page 1 of Robot_DENSO_Manual.pdf returned HTTP 200 in 72.5 seconds: 4 section headers, 10 text blocks, 10 list items. Result: `scratch/granite-real-page-result.json`. This is a functional check, not a model quality benchmark.
- Local focused suites: 54 tests run, 53 passed, 1 live test skipped. Separate opt-in Neo4j test passed with actual write/read, repeat-write counts, tenant isolation and cleanup, including rows/cells and layout edges.
- Native PDF source alignment on that same page matched 5/8 original blocks; remaining blocks retained their source structure (`model_partial`). No claim of complete model understanding.
- Semantic inference disabled, no Qwen fallback for relations. New extraction generates CONTAINS, HAS_EVIDENCE, HAS_ROW, HAS_CELL, NEXT, HAS_VISUAL, APPEARS_IN, LEFT_OF, ABOVE and NEAR when source data supports them. Nearby text does not imply EXPLAINS.
- Existing user runs were not re-extracted or backfilled. Qwen/ColPali remote availability and whole-document quality were not evaluated in this run.

## Table context and conditional relations — 2026-10-12

Added source cell/header/row metadata for native Word, spreadsheet and PDF tables.
Graph extraction, dense indexing and answering now receive table context. Missing
cell structure is reported instead of guessed. Relation proposals retain polarity,
modality, source-backed conditions and unreviewed status in Neo4j projection.
Conservative source checks reject dropped qualifiers and permission/obligation swaps.

Validation: 69 tests run, 68 passed, 1 opt-in live Neo4j test skipped. This verifies
source mapping, qualification guards and integration regressions, not model accuracy.
Existing document indexes were not rebuilt. GLiNER migration and Granite removal
remain unfinished; this change applies to the existing configured graph backend.
See `ai_extract/RELATION_CONTEXT.md` for behavior and limitations.

## Native PDF text and reusable page OCR — 2026-10-12

A3 now reads usable native PDF text with original coordinates before choosing OCR. Scans, damaged text layers and pages containing significant raster imagery stay on Surya. The text quality gate is conservative, not a semantic accuracy guarantee. Native PDF tables use geometry where available; Granite still analyzes rendered pages, and ColPali indexing is unchanged. Source metadata and the inspector distinguish PDF text/geometry from Surya OCR/layout and Office parsers.

Successful Surya page responses are atomically cached outside served media, keyed by image pixels, protocol version and inference endpoint. Fresh extractor instances reuse completed page OCR; changed pixels and failed requests do not reuse results. Rendering, Granite analysis and downstream indexes are not cached by this change. Existing uncached runs cannot recover OCR responses that were never saved.

25 targeted tests passed, including rotated native coordinates, mixed-image/scan fallback, successful-page reuse, invalidation after pixel changes, failure retry and existing pipeline/visualization tests. A real source-pipeline verification on the user's 14-page PDF took 4.56 seconds, rendered all pages and emitted 369 blocks tagged pdf_text_layer. The separate native text/geometry read took 3.127 seconds. These timings exclude Granite, BGE, ColPali, Neo4j and full end-to-end completion.

## Surya OCR timeout investigation — 2026-10-12

The previous fixed 60-second HTTP read timeout interrupted local Surya page inference. A real rerun of the newly uploaded 14-page document also failed at a 180-second timeout after the first two pages succeeded. Surya may fall back from full-page recognition to layout and block OCR when full-page output is invalid or repeated.

The OCR request now uses a 5-second connect timeout and a configurable SURYA_PAGE_READ_TIMEOUT (default 600 seconds, bounded 30–600). A timeout reports the OCR stage explicitly. Four existing Surya protocol/source normalization tests passed.

Verification of the previously failing page 3 alone succeeded with real surya-ocr-v2 inference after 314.67 seconds, returning 9 layout regions and 9 OCR blocks. The web container loaded the new setting. This verifies the timeout fix for that page; it does not measure OCR accuracy or establish successful end-to-end extraction/indexing of the entire 14-page document. The earlier full rerun remains FAILED and was not marked complete.

## A3 extraction and A2 knowledge QA alignment ? 2026-10-12

A3 extracts and indexes uploaded contract versions. A2 now supports queued company-wide questions over completed, indexed A3 knowledge without requiring a contract selection. The optional single-document QA and expert knowledge workflows remain available. Company scope is checked both when jobs are queued/processed and when their results are read.

Validation: 62 focused tests passed, including indexed-source selection, grounded-answer rejection, asynchronous company jobs and company-change access restrictions. The rendered A2 template contains the company question form; its inline JavaScript passed `node --check`. Interactive browser behavior has not been verified.

A real isolated A3-to-A2 run indexed two synthetic Word contracts (3 dense points each; ColPali skipped for native Word). The company question job returned HTTP 202, then COMPLETED. Qwen answered the separate 30-day and 60-day payment terms with citations E1 and E2 to the respective contracts; independent answer validation accepted the answer. This is a functional smoke test, not an accuracy benchmark. Raw results: `scratch/live_a2_company_results.json`.

The web and knowledge-worker containers were restarted to load the changes. Temporary SQLite, media and vector collections were cleaned; existing production contracts were not modified or automatically indexed. A2 requires users to complete A3 extraction/indexing for their saved contracts first.

# Flow test results

## Labeled metric smoke evaluation — 2026-10-12

Added `python manage.py evaluate_live_models`, with label-based metric helpers in `ai_extract/services/evaluation_metrics_service.py`. The command ran against actual Surya, BGE-M3, BGE reranker and Kaggle Qwen; it uses synthetic sources and performs no database writes. New measurements completed in 39.24 seconds without stage errors. Three metric tests passed, covering accent preservation, ranking deduplication/recall denominator and incorrect-field FP/FN accounting.

- OCR: 3 rendered images (Vietnamese contract, Vietnamese technical text, English text with slight rotation/blur), 293 reference characters / 61 reference words. Micro CER 0.3413%, WER 1.6393%. Normalization: Unicode NFC and collapsed whitespace only; case, accents and punctuation are retained.
- Text retrieval: 12 manually labeled passages and 6 questions. BM25 Recall@3=0.9167, MRR@3=0.8889, nDCG@3=0.8844. BGE and equal-weight RRF plus reranker each scored Recall@3=1, MRR@3=1, nDCG@3=1. This evaluates text ranking only; it does not exercise Qdrant, ColPali page ranking, graph/context expansion or representative enterprise data. No training or tuning was performed on these labels.
- Entity extraction: 2 source sentences / 4 positive fields. Positive-field Exact Match=1, micro precision/recall/F1=1 (TP=4, FP=0, FN=0). These strict value matches on a tiny fixture do not demonstrate general extraction accuracy.
- QA is taken from the preceding real web PDF run, not rerun in this metric command: 2 of 3 answerable questions yielded accepted generated answers with the required fixture values. One unsupported generated answer was blocked. A general hallucination/faithfulness rate and full semantic completeness were not measured.
- Document classification Precision/Recall/F1 remain unmeasured; numeric-regression MAE/RMSE are not applicable to the current extraction pipeline. Image object detection/relations require labeled boxes and relations before IoU/mAP/relation F1 can be reported.

Raw sources, predictions, qrels, rankings and counts: `contracts/tests/results/model_metrics_latest.json`. Interactive canvas: `%USERPROFILE%/.cursor/projects/d-Django-project/canvases/A3-model-evaluation.canvas.tsx`. All displayed metrics must retain the small synthetic-fixture scope; they are not a quality benchmark on real contracts or complex engineering drawings.

## Live local-model and Kaggle rerun — 2026-10-12

After the user explicitly requested startup despite low available RAM, local inference was launched with the new opt-in `start-a3-inference.ps1 -AllowLowMemory`. The default 4 GB startup guard remains unchanged. No model weights or Torch dependencies were downloaded. Surya backend and local inference health both returned HTTP 200.

All four real-model endpoint checks passed on a small synthetic image/text fixture: Granite identified the heading and paragraph (18.93 s); Surya recognized both lines (4.49 s); BGE returned two 1024-dimensional vectors (5.49 s); reranker scored the maintenance passage above the unrelated payment passage (5.17 s). Logs: `scratch/local_a3_smoke_latest.log`. These timings are single-fixture observations, not benchmarks.

The configured Kaggle tunnel became available: `/v1/models` and `/colpali/health` returned HTTP 200 from Windows and the web container. `/health` currently returns an application 404 without ERR_NGROK_3200; it is not a valid readiness route for this gateway. Earlier ERR_NGROK_3200 responses were an offline tunnel.

The isolated synthetic raster PDF completed extraction: HTTP 200, 1 page, 9 blocks, 4 clauses, BGE 8 indexed points and ColPali 1 indexed page. Neo4j projection was ready with 28 nodes and 74 edges; semantic knowledge extraction completed. Granite aligned 8 units with 1 unclassified (`model_partial`). Visual relation analysis was partial: the full page succeeded, one crop failed validation. Hybrid retrieval reported keyword, dense, ColPali and reranker ready during all four queries.

Payment QA generated the supported 30-day term with invoice and acceptance record. Image QA correctly generated that QF1 is left of M1. Maintenance QA returned source excerpts after uncited_claim validation failure. The absent-penalty question returned source excerpts after model_rejected_unsupported_claim; unsupported generated content was not presented as an accepted answer. This run does not demonstrate arbitrary-document accuracy or a fully successful answer-generation flow.

Entity extraction and summarization returned HTTP 200. The initial PAYMENT_TERM contained only the price; the backend prompt was corrected to request deadline, trigger and documents separately from CONTRACT_VALUE. A real Qwen recheck returned PAYMENT_TERM="30 ngày kể từ ngày nhận đủ hóa đơn và biên bản nghiệm thu". Six targeted gateway/live-flow regression tests passed. Recheck: `scratch/live_payment_entity_recheck.json`.

The queued frontend search completed with source excerpts and uncited_claim diagnostics. All three authenticated extraction/search template requests returned HTTP 200. The harness finished with exit code 0 and cleaned its isolated SQLite, media and scoped graph/vector fixtures. Raw complete results: `scratch/live_pdf_flow_results.json`. Production worker consumption and interactive browser behavior were not tested.

Windows paging increased during startup: observed pagefile usage rose from approximately 1.94 GB to 4.42 GB; free system-drive space fell from 11.31 GB to 3.72 GB during the run. This is not evidence that model weights were downloaded to C:. Local requests and repeated BGE/reranker switching succeeded in this run, but long-term stability after the earlier native crashes/BSOD remains unproven. Runtime remains running at the user's request.

## Latest live rerun — 2026-10-12

The four AI-mode containers are running: web, knowledge-worker, Qdrant and Neo4j. Host available RAM measured 1.41 GB; no local inference process was started. The launcher retains its precautionary 4 GB free-RAM threshold after the earlier system interruption.

Ran `scratch/live_web_flow.py --graph --extract-details` against encrypted synthetic Word input, isolated SQLite and scoped graph/vector fixtures. Layout extraction returned HTTP 200 with 7 blocks and native Office structure; clause extraction returned HTTP 200 with 4 clauses. Neo4j projection succeeded with 15 nodes and 17 edges. This is structural projection; semantic knowledge extraction was unavailable.

Both configured Kaggle health endpoints (`/health`, `/colpali/health`) returned HTTP 404 with ERR_NGROK_3200, confirming an offline tunnel. Entity extraction and summarization returned HTTP 502. All three search requests returned HTTP 200 with source excerpts: two reported gateway_unavailable and the missing-penalty question reported insufficient_evidence. The queued search completed with source excerpts, not a generated answer. Three authenticated extraction/search template requests returned HTTP 200; interactive browser behavior was not tested.

Dense indexing FAILED because local inference was not running; ColPali indexing was SKIPPED for the text-only Word fixture. Full PDF/Surya/Granite/BGE extraction and generated Qwen answers remain pending. No model quality or complete hybrid-search success is claimed. Raw results: `scratch/live_web_flow_results.json`. Fixture database, media and scoped graph/vector data were cleaned up by the harness.

The sections below are historical runs and must not be interpreted as current service availability.

## Latest visual graph verification

After adding image-local object proposals, source-to-source explanations and geometric relationships, the main suite discovers 202 tests: **200 passed, 2 skipped**. The skipped tests are the opt-in real Fabric and real Neo4j smoke tests. The Neo4j smoke test was separately enabled and passed against the existing `newidea-neo4j`: actual snapshot writes, idempotency, semantic and image-to-text traversal, NEAR edge storage, tenant/fingerprint isolation and scoped cleanup. Django configuration and JavaScript syntax checks also passed.

The old graph container had stopped after Docker restarted. It was restarted and given an `unless-stopped` restart policy; its existing data and credentials were preserved. The configured vision gateway health returned HTTP 404. Model recognition accuracy, object identities and explanatory meaning remain unverified on real documents. Controlled vision responses in tests validate protocol, provenance checks and geometry, not model accuracy. No user contract was sent to the unavailable gateway during this verification.

The results below record the earlier full multi-service test run.

218 tests discovered across five suites: **217 passed, 1 skipped, 0 failures**.

- Main contract web, A3, integrated NewIdea and Qwen gateway protocol: 182 tests, 181 passed, 1 skipped.
- Blockchain service: 6 passed.
- Workflow service: 11 passed.
- Standalone NewIdea core: 16 passed.
- Preserved A3 legacy schema: 3 passed.

The workflow package lacked `__init__.py`, causing Django discovery to import its tests outside their package. Restoring the package fixed discovery; all 11 workflow tests then passed. Installing `qdrant-client` in the project's `.venv` enabled two previously skipped dense-index tests.

## What was exercised

The suites exercise tenant access, version scope, persisted extraction data and structure reuse, Office parsing and embedded images, Surya protocol handling, Granite DocTags parsing and source alignment, hybrid fusion and reference expansion, evidence/citation checks, structure visualization payloads and template rendering, expert case approval and revision, signing policy, ledger signatures, idempotency and failure recovery.

External inference and Fabric responses are replaced with controlled fixtures in relevant tests. The DocTags parser and Office parsers run against actual parser inputs. These results verify application behavior, not model accuracy on real contracts.

## Live checks and limits

- Docker Engine's named pipe was absent, including when checked outside the sandbox.
- Web, AI, summary, workflow, blockchain, Fabric gateway and Qdrant local endpoints all refused connections.
- The configured inference gateway's `/health` returned HTTP 404.
- The live Fabric commit/recovery test remained skipped: `LEDGER_LIVE_SMOKE=1` requires a running Fabric gateway and writes an isolated test operation to the real ledger.
- Go was unavailable on the host and Docker was unavailable, so `fabric/chaincode/enterprise_test.go` could not run.
- JavaScript syntax checks passed for structure visualization, page inspector, contract pipeline and Fabric gateway.
- Django's main configuration check passed.

No existing Neon data or existing contract versions were changed. Django tests used temporary SQLite test databases. PostgreSQL migrations, a complete real-model upload-to-answer run, browser interaction and live chaincode execution remain unverified in this run.

## Reproduce

From the directory containing the main `manage.py`:

```powershell
.\.venv\Scripts\python.exe manage.py test contracts.tests ai_extract.tests NewIdea.tests.test_enterprise NewIdea.tests.test_contract_search shared.test_qwen_gateway --settings=config.intelligence_test_settings --noinput
```

From `blockchain_service`:

```powershell
..\.venv\Scripts\python.exe manage.py test blockchain --settings=config.test_settings --noinput
```

From `workflow_service`:

```powershell
..\.venv\Scripts\python.exe manage.py test workflow --settings=config.test_settings --noinput
```

From `NewIdea`:

```powershell
..\.venv\Scripts\python.exe manage.py test core --noinput
```

From `ai_extract/a3`:

```powershell
..\..\.venv\Scripts\python.exe manage.py test ai_extract.legacy_schema.tests --noinput
```

The root `test_anchor_existing_contracts.py` was not executed: it creates new versions of existing contracts and attempts real ledger writes. It is an operational script, not an isolated test suite. The root `test_blockchain_api.py` requires a running web server and hard-coded existing contract IDs; the web endpoint was unreachable.
## Service modes and WSL limit — 2026-10-12

Applied `%USERPROFILE%\.wslconfig`: `[wsl2] memory=3GB`, `[experimental] autoMemoryReclaim=dropCache`. Docker Desktop was stopped, WSL shut down (only docker-desktop distribution was present), and Docker Desktop started again. Linux reports MemTotal=2,978,400 kB, confirming the configured limit after kernel reservations.

`start-services.ps1` and the compatibility entry point `start_project.ps1` now support Ai (default), Ledger and All. AI mode runs only web, knowledge-worker, Qdrant and the existing Neo4j container, and attempts local inference only with at least 4 GB free host RAM. Ledger mode stops that AI group except web and starts workflow/blockchain/Fabric/ledger-worker; it stops only identified project inference processes. Optional containers have restart policy `no` in both the Compose overlay and existing runtime containers, preventing their return on Docker restart. No images were built, databases migrated, graph containers replaced, or volumes deleted.

Live mode checks passed: after restarting Docker, only the four AI containers returned; switching to Ledger left web plus seven ledger services; switching back to Ai returned to precisely those four AI containers. Final mode is Ai. Local inference was explicitly NOT started because the host had approximately 0.7–1 GB free RAM at the final check. Memory limits and mode switching do not resolve the previously observed native model crashes; a full local-model rerun is still pending.

## Latest interruption — 2026-10-11

### Latest authorized rerun after closing applications

Docker was restarted; Qwen and ColPali health checks returned HTTP 200. The user authorized temporarily stopping the other project containers for the A3/NewIdea test. All ten stopped containers were subsequently started again, and the knowledge worker was restarted to load the updated backend. No local model runtime was started: initial free RAM was approximately 4.15 GB before Docker, but fell below the 4 GB startup threshold after Docker started, including after clearing WSL filesystem caches and temporarily stopping the unrelated flow containers. Local native crash stability therefore remains unverified.

The real-model Word smoke completed against encrypted synthetic source data, isolated SQLite and unique vector/graph fixture scopes. Layout extraction API: HTTP 200, 7 blocks, native document structure. Clause materialization API: HTTP 200, 4 clauses. Neo4j projection: ready, 15 nodes / 17 edges; semantic extraction complete. Payment QA returned 30 days after receiving invoice and acceptance record; maintenance QA returned disconnect power and lock the breaker. The missing-penalty question returned source excerpts with insufficient-evidence status. The queued frontend job completed with a generated maintenance answer. Extraction and NewIdea template requests returned HTTP 200. The fixture job used the real worker function within the isolated process; separate production worker consumption and interactive browser behavior were not tested.

Dense indexing explicitly FAILED because the local inference endpoint was not running; dense/reranker retrieval was unavailable. ColPali indexing was SKIPPED for this text-only Word fixture. This rerun validates lexical/structure/graph retrieval and Qwen QA, not the full BGE/ColPali fusion or PDF OCR pipeline. Detailed raw results: `scratch/live_web_flow_results.json`.

Additional extraction APIs initially exposed two model errors: duration placed in DATE_EXPIRE, and an unsupported Bên B in the summary. Backend prompts were corrected to distinguish calendar dates from relative durations, include explicit maintenance obligations and avoid adding unsupported parties/roles/risks. Real gateway rechecks returned DURATION="12 tháng kể từ ngày ký", empty DATE_EXPIRE, the power-disconnection obligation, and a summary containing only the three supplied clauses. Results: `scratch/live_extraction_detail_recheck.json`. These prompt fixes passed this fixture; they do not prove semantic correctness for arbitrary documents. All 43 targeted architecture/intelligence/regression tests passed after this change.

Post-reboot lightweight live checks: Qwen evidence verification accepted the supported 30-day payment term and rejected the unsupported claim that the contract price is a penalty. Visual analysis of a newly generated QF1/M1 test image completed without schema errors and produced LEFT_OF, with proposals still unreviewed. ColPali query endpoint returned HTTP 200 with 20 vectors of dimension 128; image endpoint returned HTTP 200 with 1031 vectors of dimension 128. Raw diagnostics are in `scratch/post_reboot_remote_checks.json`. These checks use real Kaggle responses, not mock responses; they do not constitute a complete web extraction/search run or a quality benchmark on technical drawings.

The requested full rerun was not started: Docker engine was unavailable after reboot, no local inference process was running, and Windows still reported approximately 0.4 GB available RAM after the user closed applications. Background Code and four SQL Server processes remained. No unrelated user applications or database services were terminated. Heavy runtime startup remains gated on sufficient memory.

Windows restarted with bugcheck 0xEF (CRITICAL_PROCESS_DIED), recorded by System event 1001 after boot at 23:04 local time. Dump: `C:\Windows\Minidump\101126-24203-01.dmp`. No inference processes were running after reboot. Root cause has not been determined from the dump. Earlier separate Python crashes referenced `torch_cpu.dll` (access violation); the experimental non-mmap loader also raised a safetensors native panic. That experimental loader was removed. Heavy inference was not restarted after the BSOD; startup now checks at least 4 GB free RAM and 4 GB free system disk before loading models. This check is not a guarantee of stability.

Before interruption, a real-model synthetic raster PDF completed extraction: 1 page, 9 blocks, BGE index 8 points, ColPali index 1 point, Neo4j 19 nodes / 23 edges, semantic extraction complete. Granite aligned 8 units and left 1 unclassified (`model_partial`). Payment QA returned 30 days with invoice and acceptance record. Visual relationship proposals failed validation, and some later dense/reranker requests failed after the local runtime crashed. A subsequent QA incorrectly described the contract price as a penalty despite valid citation IDs. A model evidence-verification step was added to reject unsupported generated answers; it has not yet been verified end to end against the restored live runtime. Pixel-coordinate validation/conversion and a bounded visual-format repair were also added; live validation remains pending.

After the last code changes, the lightweight main suite completed: 207 discovered, 205 passed, 2 skipped. Separate earlier real Neo4j smoke: 1 passed. Workflow suite: 15 passed. Blockchain suite: 10 passed (gateway mocks, not a new real ledger transaction). HTTP template smoke returned 200 for extraction and NewIdea pages; browser interaction was unavailable. Do not interpret historical reports below as proof that all current model flows are stable.


## 2026-10-13: A2/A3 service simplification

Consolidated expert-case lookup and Qwen answer/citation-repair/semantic-verification into existing A2 services. Corpus search and single-version search share these steps. A3 local and remote OCR now share successful-result cache handling. IndexResult centralizes page aggregation and run fields; reindexing captures Dense and ColPali failures separately, preserving a successful branch when the other raises. No endpoint, database schema, extraction model, relation policy or access scope was changed.

Validation: 251 tests discovered across every NewIdea and ai_extract test module plus contract intelligence; 250 passed and the optional live Neo4j test was skipped. Added regressions for citation repair preserving sources/graph, bounded invalid-answer repair, independent index failure and partial page indexing. Tests used isolated SQLite and mocked model transports; no new live Qwen, OCR or Neo4j measurement was performed. Docker was not running during this change; deployment restart remains necessary when services are started.

## 2026-10-12: A2 searches the full authorized knowledge corpus

Removed the extra current-company discovery filter. Superusers discover documents and approved expert cases across all companies; ordinary accounts retain existing read permissions. New corpus jobs use scope `knowledge` and an access snapshot, rechecked by the worker and when reading receipts. Qdrant discovery binds each authorized company/contract/version/run tuple. Explicit identifiers found in the corpus narrow discovered candidates to the named products, retaining both documents for comparisons. Insufficient evidence now produces abstention rather than unrelated excerpts as the answer. Verbose retrieval diagnostics remain in the UI and are omitted from Qwen's prompt coverage.

Validation: 113 targeted NewIdea, contract-intelligence and hybrid-retrieval tests passed on isolated SQLite. Web and knowledge-worker were restarted. Live search receipt 12 used the existing admin account and the real Neon corpus, Qdrant/Neo4j and Kaggle configuration, without mocks or fixture documents. It scanned 4 indexed latest versions among 76 visible latest versions, selected EVP2 version 153 in another company, and returned only EVP2 page-1 sources. Retrieval: 12.394 s; processing: 14.048 s. This is a live smoke measurement, not a quality benchmark.

Live limitations: BGE/reranker runtime on port 8010 was not listening; approximately 0.75 GB host RAM was free, so heavy inference was not restarted. ColPali retrieval succeeded. Qwen generation returned HTTP 404 with ERR_NGROK_3200 at `/v1/chat/completions`; no successful generated answer was recorded. The result therefore used source excerpts. Raw live receipt: `scratch/global_knowledge_live_result.json`; repeatable queue probe: `scratch/check_global_knowledge_live.py`.

## 2026-10-12: A2 per-claim citation fix and successful live answer

The user's real receipt 17 retrieved the EVP2 document correctly and completed two Qwen generation calls. Both answers were rejected with `uncited_claim`: Qwen cited only the last sentence of a multi-sentence paragraph. A real replay reproduced this behavior. This was a generation-format failure, not an offline gateway or missing document.

A2 now requests schema-constrained claims with a text sentence and model-selected source IDs for each claim. The adapter validates source bindings and renders their citations. Existing sentence/quantity validation and the separate Qwen evidence-verification call remain required; missing citations, unknown IDs, unsupported quantities and unsupported semantics are not accepted. Free-text generation in other callers remains compatible.

Validation: 53 targeted NewIdea answer, extraction regression, visualization and contract-intelligence tests passed against isolated SQLite. Real Kaggle generation and semantic verification succeeded. Web and knowledge-worker were restarted, then normal queued knowledge search receipt 18 completed with `generation_status=generated`, valid citations and a positive model evidence check. Its answer explains Guidance on PC versus Runtime with controller and camera, citing EVP2 page 1. Retrieval: 5.860 s; processing: 19.081 s. The 49.829 s receipt age when inspected is not the processing latency.

This live run used existing Neon documents/cases and actual Qwen/ColPali calls, without model mocks. BGE and the neural reranker were still unavailable; this result does not validate all hybrid branches or establish general answer accuracy. Expert seed cases remain labeled DEMO. Source evidence and generation trace: `scratch/global_knowledge_live_result_18.json`. The original receipt 17 is preserved; re-asking in the web uses the corrected worker.

## 2026-10-12: A2 sends Neo4j nodes and edges to Qwen

Previously Neo4j only expanded candidate passage IDs; the generator received text passages and heading paths, not the connected graph. Global search also discarded expanded context after two passages per document. It now preserves up to two ranked matches plus two context passages per selected document within the shared result budget. A scoped Neo4j read supplies connected hierarchy, evidence bindings, table structure, visual bindings, semantic proposals and geometry/order edges that actually exist in the selected snapshot. Nodes/edges are rehydrated from current A3 data and bound to citation IDs delivered to Qwen. Unsent/foreign sources are removed, draft relations remain navigation hints, and geometry/order never establishes semantic continuation.

The prompt now explains every supported question part rather than requiring an ultra-short answer. Both generation and verification receive the graph. Graph delivery is bounded and marks omissions as partial. The UI displays the delivered node/edge counts and a relation preview. Text questions reuse OCR text plus graph; visual questions and primary image-only hits retain source images. DEMO advice stays visible but is only included in generation when asking for expert/demo advice. Actual approved advice remains available normally.

An intermediate live run (20) exposed vLLM's 4,096-token limit and returned source excerpts after HTTP 400; it is not a successful generation measurement. Context now omits redundant source previews/provenance fields and an output-reservation retry preserves the prompt. A shared-page binding bug found in the probe was also fixed: APPEARS_IN does not bind unrelated layout blocks to the same paragraph.

Final validation: 154 tests discovered, 153 passed, 1 optional synthetic Neo4j smoke skipped, on isolated SQLite. Separate actual Neo4j read confirmed ready context and rejection of changed company, contract, version, run and fingerprint scopes. Existing corpus coverage: 76 visible latest versions, 4 with completed A3 extraction/current graph snapshots. Web and worker were restarted. Real queued receipt 21 generated a two-part EVP2 answer, with 12 nodes and 17 relations confirmed delivered to Qwen, valid citations and a positive model evidence check. Processing latency: 19.523 s. Graph context was partial; this is a selected subgraph, not a full-document review. ColPali succeeded; BGE/reranker remained unavailable. Raw receipt: `scratch/global_knowledge_live_result_21.json`; read-only graph probe: `scratch/a2_graph_context_live.json`. No document re-extraction, graph data rewrites or model checkpoint changes were performed.


## 2026-10-14: Retry incomplete ColPali pages at the end of A3

A3 now stores visual results by page and performs one final retry pass for FAILED/PARTIAL pages after graph reconciliation and dense indexing. COMPLETED/SKIPPED pages are not sent again. The retry does not repeat Surya, Granite, graph inference or BGE embedding. It also runs when all OCR pages failed, without marking OCR successful. Deterministic Qdrant point IDs preserve acknowledged writes and prevent duplicate counts across partial attempts. Persistent failures remain explicit in index status, page errors and processing.colpali_missing_pages; logs expose colpali_retry/colpali_retry_page and pages record colpali_attempts. No stored extraction data was edited. This is one in-session final pass, not durable resume of a terminated process or a repair of existing failed runs.

Validation: 396 tests discovered, 394 passed, 2 optional integration tests skipped, using isolated SQLite. Added regression coverage verifies that an early visual failure allows the next page to be processed, only failed pages are retried after dense indexing, source/model work is not repeated, partial attempts retain and deduplicate acknowledged point IDs, final failures do not loop, and failed OCR remains failed despite successful visual recovery. Another test exercises a successful five-point write followed by a failing final Qdrant batch. These are controlled regression tests, not live model benchmarks. Log: scratch/colpali-final-retry-full-tests.log.

Runtime activation remains pending: Windows C: reports 0 free bytes and the bounded Docker API probe times out. The updated files are saved on D:, but the existing web/worker processes have not been restarted to load this change.


Runtime activation follow-up, 2026-10-14: Disk cleanup removed unused npm/pip download cache, closed crash dumps, old large temporary files, and unlocked swap.vhdx files from previous WSL sessions. The active swap and all model/document/database files were retained. Docker build cache cleanup reclaimed 609.7 MB internally; this is not an equal measured host-space gain. Final C: free-space reading was 4,150,124,544 bytes (about 3.9 GiB). Windows pagefile allocation remained 27,069 MiB with system management unchanged. Docker Desktop was recovered and web/knowledge-worker restarted. A container probe confirmed the final retry method and acknowledged-point tracking were loaded, Neon SELECT 1 succeeded, and runtime/Qdrant/ColPali health requests returned HTTP 200. This resolves the runtime activation blocker above; no full document extraction was rerun and no old failed extraction state was edited.


## 2026-10-14: Actual complete A3 rerun, contract 120 / version 151

Normal LayoutExtractionService.extract_version completed new run 19 in 372.959 seconds: 14 pages, 247 source blocks, 255 BGE dense points and 14 ColPali page points. The prior failed runs were preserved. Independent scoped Qdrant reads confirmed these point counts and ColPali pages 1 through 14; Neo4j reads confirmed 582 nodes and 5,285 edges in the final source-fingerprint snapshot. All page graph writes and final graph reconciliation returned ready. No OCR page or visual index failed; the final ColPali retry pass was not needed in this run. No Qwen call is part of this A3 execution.

Cache usage: {"surya": {"cached": 13, "fresh": 1}, "granite": {"cached": 12, "fresh": 2}}. This wall time measures a rerun with reused OCR/Granite results, not extraction of an unseen document. ColPali and BGE indexing made real inference requests. Granite received all 14 page images, with no provider failures, but source alignment remains model_partial (225/247 source blocks aligned); successful completion does not establish OCR or semantic relationship accuracy. Geometry/source/table edges are present; knowledge and visual semantic inference providers are disabled in this snapshot.

Artifacts: scratch/a3-rerun-151.log, scratch/a3-rerun-151-summary.json, scratch/a3-rerun-151-audit.json. Temporary decrypted input was created on D: through the mounted project scratch directory and removed by the normal service.

## Visual-region OCR acceptance — 2026-10-14

- Full local regression suite: 418 tests discovered, 416 passed, 2 optional
  dependency tests skipped. Log: `scratch/visual-ocr-full-tests.log`.
- Real GPU inference through the production extractor used run 19, page 7,
  two saved figures: visual OCR 42.027 s, end-to-end probe 45.918 s. The
  ordinary body OCR came from the exact-image legacy cache.
- Recovered printed text includes `Tree Call`, `Nested Tree`, `Exit Guard`.
  Figure types and parent bboxes remained unchanged; inner text carries
  source-region bboxes. Tiny symbols remain imperfect (for example i3/i6
  can be read as 13/16). One optional glyph crop failed; one overly long
  candidate output was rejected. No CER/WER or whole-PDF accuracy claim
  follows from this smoke test.
- Regression cases cover skipped figures, HTML/JSON descriptions rejected
  as OCR text, source-coordinate preservation after upscaling, legacy body
  cache reuse, retry after failed visual upgrade, and retention of body
  text after internal OCR failure.
- Probe JSON: `scratch/visual-ocr-live.json`. Source run 19's persisted
  blocks, graph and vectors were not rewritten by the probe. No Qwen or
  ColPali request was needed for this local OCR acceptance.

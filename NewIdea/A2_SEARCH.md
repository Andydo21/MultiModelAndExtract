# A2 source and expert search

Users enter a question at `/newidea/`. Corpus discovery searches authorized latest
versions using shared lexical/dense/visual streams, RRF and one shared reranker.
The result shows source citations and expandable content/heading nodes with
links to the exact source page and snapshot-scoped Neo4j focus view.

Dense vectors represent A3 source leaves, including heading paths and extracted
context. They carry node IDs and ancestor IDs; structure nodes do not each have
an independent vector. A2 resolves headings from retrieved leaves rather than
claiming every Neo4j node is independently embedded. Expert submission records
share corpus retrieval; legacy approved cases also have a scoped BM25 lookup.

Generation normally receives up to 24 selected sources, with up to 6 matches
per document (`A2_SOURCES_PER_DOCUMENT`) and reserved expansion context.
Full selected passages are retained. Source-backed semantic relations have
priority in Neo4j reads and reserved room in the 6000-character graph payload;
their evidence IDs and unreviewed status remain attached.

Cross-language questions can pass the relevance gate using a positive BGE
reranker logit; lexical overlap is not required in that case. This is permission
to attempt generation, not answer confidence. Numeric/reference checks and
subsequent citation/semantic verification still apply.

Stored `document-structure-v2` Granite/source trees are now reused when the
source fingerprint matches, alongside the older model schema. Previously A2
accepted only the older schema, potentially losing Granite order and hierarchy
during search. A changed fingerprint still requires reanalysis.

Approved expert cases with current source hashes become separate expert-opinion
citations for company answers and ordinary single-document questions. Full
reasoning, resolution, verification, applicability, source text, author and
revision are supplied. Prompt/verification distinguishes opinions from document
obligations and does not assume applicability. Whole-document batch review
continues to review document sources; related expert cases remain separate.
Drafts and stale cases are excluded by server-side retrieval. Citation validation
and model checks are safeguards, not proof of answer correctness.

Graph context expansion no longer treats the document root as a common parent
that connects unrelated evidence. Spatial relations remain geometry, not proof
of semantic relevance. Discovery examines the visible indexed latest versions
and retains up to 60 fused candidates for reranking; partial coverage is
reported. Similarity/retrieval scores are not confidence values.

Live check on 2026-10-15 (search 43): BM25, dense retrieval, neural reranking
and scoped graph expansion ran against the EVP2 corpus. The prepared Qwen
payload included Guidance/Runtime/PC passages and DEPENDS_ON with evidence.
Qwen returned HTTP 404 / ERR_NGROK_3200, so final generation and verification
were not validated in this check.

Historical validation: 100 focused tests ran successfully, 1 optional live Neo4j test
skipped. Includes real database approval/stale-source exclusion, generation
payload integrity, source-schema reuse, authorization and graph/source checks.
Model responses in unit tests are protocol fixtures, not an accuracy benchmark.

Live contract 120/version 151/run 8: query `Dead data-stack store elimination`
retrieved the paragraph under `5. Trace Tree Optimization > 5.1 Optimizations`,
using the stored Granite tree. BM25 and dense retrieval ready; Neo4j expansion
ready; ColPali unavailable (HTTP error). Authenticated `/newidea/` HTML returned
200 with node navigation. Qwen generation was not requested in this live check.
Report: `scratch/a2-node-search-report.json`.

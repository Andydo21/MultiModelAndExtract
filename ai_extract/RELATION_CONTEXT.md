# Table context and qualified relationships

Native Word, spreadsheet and PDF tables preserve source cell values and column
headers in extraction metadata. PDF table leaves retain their source row index.
The first row of an Office table is only a header candidate, not a verified
semantic label; ragged rows remain ambiguous. OCR tables without cell structure
are not silently assigned headers. Explicit units remain in the original header
or value; missing units are never generated.

`relation_context_service` supplies the row context to graph extraction,
dense indexing and source-based answering. The original evidence text is kept.
Oversized context is omitted and reported rather than clipped inside a cell.

Semantic relation proposals include `polarity`, `modality`, `condition_quotes`,
`scope_quote` and `qualification_quote`. Exact source matching and conservative
English/Vietnamese cue checks reject dropped negations/conditions and obvious
permission/obligation swaps. These checks are not a semantic entailment model:
ambiguous scope can cause abstention and expressions outside the cue vocabulary
still need evaluation. Accepted relations retain `review_status=unreviewed`.
The same qualifications are persisted as Neo4j relationship properties.

Existing runs/graphs/embeddings are not rewritten automatically. Re-extraction
populates new table metadata; rebuilding structure/graph and reindexing are needed
to refresh old results. Current runtime uses Surya/native Office observations
for source structure and GLiNER for source-bound entity/relation proposals.
Proposals remain unreviewed; source/bbox edges do not establish semantic truth.

A2 reads the current tenant/version/run snapshot from Neo4j. Context edges are
interleaved by type so evidence bindings cannot crowd out reading-order edges.
Source presentation follows unambiguous stored NEXT bindings; cycles preserve
retrieval order rather than inventing an order. Prompt-local node aliases do
not change graph identities or stored relationships.

Answers select native source IDs. A2 sends the original question, native source
passages and source-bound Neo4j nodes/edges with a relation legend. Every retrieved
type is eligible; spatial links are supplementary and never semantic proof.
Context limits are explicit. No entity-name decoder constraints or fixed answer
glossary are used. Each claim is verified by the model using its selected sources,
followed by a separate coverage check. Model verification can still be wrong;
live acceptance requires source review.

Validation: 2026-10-12, 69 tests run, 68 passed, 1 opt-in live Neo4j test skipped.
Includes row-to-column mapping, units, ragged rows, conditional Vietnamese
obligations, negated English obligations, permission swaps, fabricated conditions,
and qualification propagation through graph projection. Model accuracy and
production Neo4j writes were not measured in this test run.

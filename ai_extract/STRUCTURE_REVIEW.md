# Structural relation review

The final extraction stage, GLiNER enrichment and `sync_document_graph` review
the source structural relations before replacing the Neo4j snapshot. A3 uses
Surya and GLiNER; this stage does not call Qwen.

`infer_relations` routes to the same GLiNER span validation used by enrichment.
The retired LayoutLM callback adapter and its keyword qualification rules are
removed. `PageExtractionSession` is the extraction persistence path; indexing
and artifact retirement finish there, rather than a second batch-save hook.
The unused vision-relation inference adapter and its no-op `--analyze-images`
option are removed. Existing visual observations remain readable; source-image
freshness checks, source geometry and ColPali indexing are retained.

Every structural edge receives an audit entry. Endpoint/source bindings and
spatial direction receive integrity checks. NEXT and CONTAINS proposals with
source text on both sides are sent to GLiNER for a separate semantic proposal.
Links without two-sided text are explicitly marked not applicable to this text
model. They are not described as model-confirmed.

GLiNER can support, dispute or abstain. Its result does not rewrite the native
Surya order or hierarchy. A2 receives these statuses, excludes disputed NEXT
constraints and treats unresolved native order as provisional presentation
order rather than semantic evidence. Invalid source edges are excluded without
creating replacement NEXT links across the gap.

Completed reviews are cached against the source/structure fingerprint. A changed
source triggers a new review. Runtime failure leaves pending states and a partial
review; it cannot become a positive confirmation. Complete means all applicable
checks ran, not that the document structure is guaranteed correct.

Real run 43: 616 structural edges audited; 513 integrity checks, 64 links without
two-sided text, 39 GLiNER abstentions, no supported/disputed/pending/invalid edges.
See `scratch/structure-review-real.log`. This run did not demonstrate that GLiNER
can resolve ambiguous reading order; no automatic correction is claimed.

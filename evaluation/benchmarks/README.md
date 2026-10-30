# Benchmark inputs

`pilot50_gold.jsonl` is intentionally empty: no independently reviewed pilot
questions have been supplied. An empty file fails dataset validation. It does
not claim that a Pilot-50 exists. Do not copy illustrative answers into gold.

`historical_audit.json` contains file/row selectors, not hand-entered results.
E0 reads actual Markdown cells and raw JSON values on each invocation.
Q1–Q4 are regression only and must never be counted as independent test data.

Each real JSONL question needs question_id, split (pilot/test/regression),
question, categories, reference_answer, ground_truth_tuples, gold_evidence_ids,
required_evidence_ids, evidence_locations, reviewer and review_status=approved.
Each tuple contains subject, attribute, value, condition (possibly null),
and evidence_ids. Locations must bind document_id/version/page/region to an
actual evidence record. Text-layer regions are not automatically gold evidence.

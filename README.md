# Hierarchical Evidence Pods

Code, evaluation scripts and logs for the paper *Hierarchical Evidence Pods: Structure-Preserving Hybrid Retrieval and Graph-Grounded Question Answering over Complex Technical Documents* (Do Dang An, Dao Thi Thuy Quynh, PTIT).

The system analyses document structure offline. The generator then receives explicitly scoped evidence **as text**, so a language model without a vision encoder can answer from tables, footnotes and figures. Vision is used offline (layout, OCR) and for retrieval (ColPali).

## Repository layout

| Path | Contents |
|---|---|
| `ai_extract/` | **Offline extraction (A3).** Rendering, Surya 2 layout/OCR (`services/surya_inference_service.py`, `layout_extractor.py`), TATR table grids (`layout_service.refine_page_tables`), heading hierarchy (`document_structure_service.py`), GLiNER entities and GLiNER2 identity (`gliner_entity_service.py`, `gliner_identity_classifier.py`), Neo4j graph projection and related passages (`document_graph_service.py`), BGE-M3 / ColPali indexing |
| `NewIdea/` | **Online retrieval and answering (A2).** Hybrid search, RRF, reranking (`services/knowledge_search_service.py`), seed selection (`search_evidence_service.py`), Evidence Pods and checked generation (`knowledge_answer_service.py`) |
| `shared/` | BM25 and RRF (`retrieval.py`), Qwen gateway client |
| `contracts/`, `config/` | Django project, document/version models and access control used by the two apps above |
| `evaluation/` | Evaluation framework. `runners/pilot_freeze.py` freezes inputs, `runners/pilot_mainflow.py` runs the production flow, `runners/pilot_run.py` runs the E1/E2/E4 ablations, `runners/*_report.py` produces reports |
| `research_evaluation_20261030/` | Pilot set and results (see below) |
| `paper/` | LaTeX source of the paper |

## Models

| Role | Model |
|---|---|
| Layout, OCR, reading order | Surya 2 (GGUF, llama.cpp) |
| Table structure | `microsoft/table-transformer-structure-recognition` |
| Entities | `knowledgator/gliner-relex-multi-v1.0` (NER only) |
| Same-name identity, passage topic check | `fastino/gliner2-base-v1` |
| Dense retrieval | `BAAI/bge-m3` |
| Reranking | `BAAI/bge-reranker-v2-m3` |
| Visual retrieval (figure crops) | `vidore/colpali-v1.2` |
| Generator / planner | `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` on vLLM (zero-shot), 2× NVIDIA T4 |

Stores: PostgreSQL (sources, BM25 postings), Qdrant (dense and ColPali vectors), Neo4j (per-document graph snapshots).

## Running

1. `pip install -r requirements.txt` (plus `ai_extract/requirements-vision.txt` and `ai_extract/requirements-gliner.txt` for the extraction runtime).
2. Copy `.env.example` to `.env` and fill in the database, Neo4j, Qdrant, the generator URL (`VLLM_BASE_URL`) and ColPali URL (`COLPALI_BASE_URL`).
3. Start PostgreSQL, Qdrant and Neo4j, then `python manage.py migrate`.
4. The local inference runtime (`ai_extract/views/inference_app.py`) serves Surya, TATR, GLiNER, BGE-M3 and the reranker on port 8010.

## Reproducing the evaluation

```bash
# 1. Freeze inputs: gold set, production retrieval, Neo4j graphs, production pods (needs Django, DB, Neo4j)
python -m evaluation.runners.pilot_freeze research_evaluation_20261030/frozen_20261030
python -m evaluation.runners.pilot_freeze research_evaluation_20261030/frozen_20261030 --leaves

# 2. Production main flow, 50 questions
python -m evaluation.runners.pilot_mainflow research_evaluation_20261030/frozen_20261030 research_evaluation_20261030/mainflow_20261030

# 3. Worked-example ablations
python -m evaluation.runners.pilot_run e1 FROZEN OUT --questions P002,P018,P023,P029
python -m evaluation.runners.pilot_run e2 FROZEN OUT --questions P003,P007,P022,P027
python -m evaluation.runners.pilot_run e4 FROZEN OUT --questions P013,P028,P039,P040
```

The protocol (fixed before generation; SHA-256 in `protocol_sha256.txt`) is `research_evaluation_20261030/PILOT50_PROTOCOL.md`. The grading rubric is `RUBRIC_DANH_GIA.md`.

## Results and logs

`research_evaluation_20261030/`:
- `frozen_20261030/gold_questions.jsonl`: 50 pilot questions with reference answers, subject–attribute–value–condition tuples, gold evidence IDs and locations.
- `frozen_20261030/evidence.jsonl`, `retrieval.json`, `production_pods.json`: frozen evidence, production retrieval rankings (top 10 per question) and rendered pods.
- `KET_QUA_20261030/`: per-question answers of the production flow (`mainflow_answers*.jsonl/csv`), manual grades (`mainflow_manual_grades.csv`, `E4_manual_grades.csv`), retrieval and citation metrics (`mainflow_retrieval_citation*.csv`), worked examples (`E1/E2/E4_examples.csv`), and workbooks (`*.xlsx`). See its `README.md`.
- `runs/E1`, `runs/E2`, `runs/E4`: raw ablation logs with prompts, selected evidence IDs, outputs, citations, token counts and graph traces.

## Notes on this release

- Source documents (product brochures and manuals), databases, extracted images, model checkpoints and graph exports are not included. Evidence texts in the logs are short excerpts used for evaluation.
- Gateway host names and two e-mail addresses found in third-party source text were redacted in the released logs. The raw ablation logs are hash-chained (`evaluation/io.py`); because of this redaction, verify the chain against the original logs, not against these copies.
- The pilot set uses documents that were also used during development. Answers were graded by one of the authors, so the grades are not an independent human evaluation.

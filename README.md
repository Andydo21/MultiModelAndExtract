# Hierarchical Evidence Pods

Code, evaluation scripts, logs and paper for *Hierarchical Evidence Pods: Structure-Preserving Hybrid Retrieval and Graph-Grounded Question Answering over Technical Documents* (Do Dang An; supervisor Dao Thi Thuy Quynh; Posts and Telecommunications Institute of Technology, PTIT).

The system analyses document structure offline (layout, OCR, table grids, figures, a per-document graph). Online, it retrieves source regions with BM25, BGE-M3 and figure-level ColPali, expands them inside their heading scope, and gives the generator scoped evidence **as text** (Evidence Pods) with stable citation IDs.

## Two ways to look at it

### 1. Results viewer (no installation)

Open [`viewer/index.html`](viewer/index.html) in a browser (download the repository, then double-click the file; with GitHub Pages enabled on this repository it is served at `/viewer/`).

For every one of the 50 pilot questions it shows the question, reference answer, system answer, each claim with the sources it cites (green = gold region, red = not gold), the gold evidence and whether it reached the generator, the grade and the per-question metrics. It can be filtered by label (correct / partial / wrong / no answer), by category (table, footnote, condition, ...) and by text. Further tabs show the E2 graph-expansion and E1 table-question runs. All data are read from the stored logs; nothing is recomputed in the page.

### 2. The full system (web interface)

The Django web application provides the interface used to build and query the index:

| Page | URL (local) | What it does |
|---|---|---|
| Dashboard | `/` | contracts / documents and their versions |
| Extraction pipeline | `/ai-extract/` and `/ai-extract/contracts/<id>/` | extraction status per document |
| Page inspector | `/ai-extract/contracts/<id>/pages/<n>/` | every extracted region drawn on the rendered page, with its label, reading rank, OCR text and graph links |
| Knowledge search and answers | `/newidea/` | ask a question; see retrieved sources, evidence pods, the cited answer |

Start it with Docker (Windows PowerShell scripts are included):

```bash
cp .env.example .env            # fill in DB, Neo4j, Qdrant passwords and the generator URLs
docker compose -f docker-compose.yml -f docker-compose.web-local.yml -f docker-compose.graph.yml \n  up -d postgres qdrant neo4j a3-inference web knowledge-worker
docker compose -f docker-compose.yml -f docker-compose.web-local.yml exec web python manage.py migrate
# then open http://127.0.0.1:8000/
```

`start-services.ps1` and `start-a3-inference.ps1` do the same on Windows; `RUNTIME.md` describes the services. The local inference runtime (port 8010) serves Surya, TATR, GLiNER, BGE-M3 and the reranker. The generator (Qwen2.5-VL-7B-Instruct-AWQ on vLLM) and ColPali run behind one gateway; `serving/kaggle_vllm_qwen_colpali_gateway.ipynb` starts both on a Kaggle 2×T4 machine (set your own `HF_TOKEN` and `NGROK_AUTH_TOKEN`) and the resulting URL goes into `VLLM_BASE_URL` / `COLPALI_BASE_URL`.

## Repository layout

| Path | Contents |
|---|---|
| `ai_extract/` | Offline extraction: rendering, Surya 2 layout/OCR, TATR table grids, heading hierarchy, GLiNER-RelEx entities and GLiNER2 identity, Neo4j graph projection, BGE-M3 / ColPali indexing, page inspector UI |
| `NewIdea/` | Online retrieval and answering: hybrid search, RRF, reranking, seed selection, graph expansion, Evidence Pods (`services/knowledge_answer_service.py`: `answer_source_pods`, `render_source_pods`), checked generation |
| `shared/`, `contracts/`, `config/` | BM25/RRF, generator client, Django project, document/version models and access control |
| `evaluation/` | Evaluation runners: `pilot_freeze.py` (freeze inputs), `pilot_mainflow.py` (production main flow), `pilot_run.py` (E1/E2/E4), `pilot_deliverables.py` (tables, charts) |
| `research_evaluation_20261030/` | Pilot set, logs and results (below) |
| `viewer/` | Static results viewer |
| `paper/` | LaTeX source of the paper (Springer LNICST) and figures |
| `serving/` | Notebook that serves the generator and ColPali |

## Models

| Role | Model |
|---|---|
| Layout, OCR, reading order | Surya 2 (`datalab-to/surya-ocr-2`, GGUF via llama.cpp) |
| Table structure | `microsoft/table-transformer-structure-recognition` |
| Entities | `knowledgator/gliner-relex-multi-v1.0` (NER only) |
| Same-name identity, topic check | `fastino/gliner2-base-v1` |
| Dense retrieval / reranking | `BAAI/bge-m3` / `BAAI/bge-reranker-v2-m3` |
| Visual retrieval (figure crops) | `vidore/colpali-v1.2` |
| Generator / planner | `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` on vLLM, zero-shot, 2× NVIDIA T4 |

## Evaluation data and deliverables

| Deliverable | Where |
|---|---|
| Configuration | `evaluation/runners/pilot_run.py` (`CONFIG`), `research_evaluation_20261030/PILOT50_PROTOCOL.md` (+ SHA-256), `.env.example` |
| Git commit | this repository's history; `research_evaluation_20261030/SOURCE_COMMIT.txt` records the state of the development repository the runs were made from |
| Corpus list | `research_evaluation_20261030/CORPUS_LIST.csv` (20 documents asked about; the index holds 201 document versions) |
| Questions and gold evidence | `research_evaluation_20261030/frozen_20261030/gold_questions.jsonl` (50 questions, reference answers, tuples, gold region IDs) |
| Per-question JSONL logs | main flow: `research_evaluation_20261030/mainflow_20261030/answers_images_off.jsonl` (sources sent, claims, citations, timing, raw model outputs for the reruns); ablations: `runs/E1`, `runs/E2`, `runs/E4` `questions.jsonl` (full prompts, selected evidence IDs, outputs, citations, token counts) |
| Retrieved / selected evidence | `frozen_20261030/retrieval.json` (top-10 per question with channel), `production_pods.json` (rendered pods) |
| CSV / XLSX with mean ± SD, right/wrong counts, error cases, full comparison, charts | `research_evaluation_20261030/KET_QUA_20261030/` (`BANG_TONG_HOP.xlsx`, `summary_mean_sd.csv`, `error_cases.csv`, `comparison_full.csv`, `charts/`) |

Main-flow results (text-only, 50 questions): Hit@1 0.800, MRR 0.869, required-evidence recall 0.955, answer accuracy 0.800 (33 correct, 14 partial, 2 wrong, 1 no answer), attribute leakage 5/98 claims, citation faithfulness 87/98 claims. See `research_evaluation_20261030/KET_QUA_20261030/README.md`.

Reproduce:

```bash
python -m evaluation.runners.pilot_freeze research_evaluation_20261030/frozen_20261030          # needs DB, Neo4j
python -m evaluation.runners.pilot_mainflow research_evaluation_20261030/frozen_20261030 research_evaluation_20261030/mainflow_20261030 --modes off
python -m evaluation.runners.pilot_run e1 FROZEN OUT --questions P002,P018,P023,P029
python -m evaluation.runners.pilot_deliverables research_evaluation_20261030
```

## Honesty notes

- Numbers are reported as produced by the scripts; nothing was edited to improve results. Negative results are kept (for example, Evidence Pods gave no gain over row-wise text on the five table questions of E1).
- Five main-flow answers come from a second run: four after a temporary server outage, one (P007) after an output-language check that rejected its first answer was removed from the system. The first-run rows are kept.
- Missing logs: the raw output of P007's rejected first answer was not stored; main-flow prompts are rebuilt by the production code and not stored per question.
- The pilot uses documents also used during development. Labels were assigned with LLM assistance and reviewed by the first author, without a second independent grader.
- The ablation schedules were stopped part-way; only questions that completed every configuration are used.
- Source documents, databases, extracted images, model weights and graph exports are not included. Gateway host names and e-mail addresses found in third-party text are redacted in the released logs, so verify the hash chain of the ablation logs against the originals.

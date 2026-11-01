# Hierarchical Evidence Pods

Code and evaluation evidence for *Hierarchical Evidence Pods: Structure-Preserving Hybrid Retrieval and Graph-Grounded Question Answering over Technical Documents* (Do Dang An; supervisor Dao Thi Thuy Quynh; Posts and Telecommunications Institute of Technology, PTIT).

The system analyses document structure offline (layout, OCR, table grids, figures, a per-document graph). Online, it retrieves source regions with BM25, BGE-M3 and figure-level ColPali, expands them through the graph, and gives the generator scoped evidence as text, with stable citation IDs.

## Folders

The folder names come from the project's history, so some do not describe their contents. In the code, "A3" means the offline extraction stage and "A2" the online question-answering stage.

| Folder | What it is | Main entry points |
|---|---|---|
| `ai_extract/` | **Offline extraction (A3).** Reads a document once and stores its structure: page rendering, layout and OCR with Surya, table grids with the Table Transformer, heading hierarchy, entities with GLiNER, the per-document Neo4j graph, and the BGE-M3 and ColPali indexes. Also holds the extraction pages of the website and the local GPU inference service | `services/layout_service.py`, `services/surya_inference_service.py`, `services/document_structure_service.py`, `services/document_graph_service.py`, `views/inference_app.py` |
| `NewIdea/` | **Online question answering (A2).** Despite the name, this is the core of the paper: it takes a question, retrieves source regions (BM25, BGE-M3, ColPali, rank fusion, reranking), selects seeds, expands them through the graph, builds the **Hierarchical Evidence Pods** and generates the cited answer. Also serves the `/newidea/` page | `services/knowledge_search_service.py`, `services/search_evidence_service.py`, `services/knowledge_answer_service.py` |
| `contracts/` | **The web application around both stages.** The project began as a contract-management system, so "contract" here means any uploaded document. Holds the document and version models, upload and download, login and access control, the dashboard pages, and the background commands | `models.py`, `urls.py`, `views/`, `templates/` |
| `shared/` | **Code used by several apps:** BM25 and rank fusion, the client that calls the generator, and JWT login middleware | `retrieval.py`, `qwen_gateway.py`, `jwt_middleware.py` |
| `config/` | **Django project settings:** environment variables, database connections, and the top-level URL routes | `settings.py`, `urls.py` |
| `runtime/` | **Scripts for the GPU inference service** (the `a3-inference` container): one installs its Python environment and llama.cpp, the other starts it on port 8010 | `setup_a3_linux_runtime.sh`, `start_a3_linux_runtime.sh` |
| `serving/` | **Notebook for the remote models.** Runs the generator (Qwen2.5-VL on vLLM) and ColPali on Kaggle behind one public gateway | `kaggle_vllm_qwen_colpali_gateway.ipynb` |
| `research_evaluation_20261030/` | **Evidence for the paper,** from the evaluation of 30–31 October 2026. Nothing here is needed to run the system | See below |

Inside `research_evaluation_20261030/`:

| Folder | What it is |
|---|---|
| `frozen_20261030/` | The fixed inputs: 50 questions with gold evidence, the evidence texts, retrieval rankings and rendered pods |
| `mainflow_20261030/` | Raw per-question logs of the full system on the 50 questions (`images_off` is the text-only run in the paper) |
| `runs/` | Raw logs of the smaller experiments: E1 evidence format, E2 graph expansion, E4 visual input |
| `documents/` | The test documents: the exact PDFs in the index. `pilot_20/` holds the 20 documents the questions ask about; `arxiv_180/` holds 180 arXiv papers that act as distractors. `DOCUMENT_INDEX.csv` maps all 201 indexed versions to a file, SHA-256 and source link (one version is a repeat upload of `DigitalTwinSoftware_en`) |
| `KET_QUA_20261030/` | Final results ("kết quả" is Vietnamese for "results"): CSV and XLSX tables, grades, error cases and charts |

## Run

Requirements: Docker with an NVIDIA GPU for `a3-inference`, and a gateway serving the generator and ColPali.

1. **Configure.** `cp .env.example .env`, then set the passwords and the gateway URLs. To get the gateway, run `serving/kaggle_vllm_qwen_colpali_gateway.ipynb` on a Kaggle 2×T4 machine with your own `HF_TOKEN` and `NGROK_AUTH_TOKEN`.
2. **Add the models** under `scratch/a3-models/` (not included in this repository):

   | Folder | Model |
   |---|---|
   | `surya/` | Surya 2 (`datalab-to/surya-ocr-2`) as `surya-2.gguf` and `surya-2-mmproj.gguf`, with `chat_template.jinja` and its tokenizer/processor configs |
   | `table-transformer/` | `microsoft/table-transformer-structure-recognition` |
   | `bge-m3/` | `BAAI/bge-m3` |
   | `gliner-relex-multi-v1.0/` | `knowledgator/gliner-relex-multi-v1.0` |
   | `gliner2-base-v1/` | `fastino/gliner2-base-v1` |
   | `nli-deberta-v3-xsmall/` | NLI checkpoint for relation review |

   `BAAI/bge-reranker-v2-m3` is read from the Hugging Face cache at `A3_HF_CACHE_DIR`. The runtime works offline, so download everything first.
3. **Build and start.**

   ```bash
   docker compose build web
   docker compose up -d postgres qdrant neo4j
   docker compose run --rm --no-deps a3-inference sh /app/runtime/setup_a3_linux_runtime.sh   # first time only
   docker compose up -d a3-inference web knowledge-worker entity-worker
   docker compose exec web python manage.py migrate
   ```

4. **Open** http://127.0.0.1:8000/

| Page | URL |
|---|---|
| Dashboard: documents and versions | `/` |
| Extraction status | `/ai-extract/`, `/ai-extract/contracts/<id>/` |
| Page inspector: regions drawn on each page | `/ai-extract/contracts/<id>/pages/<n>/` |
| Questions and cited answers | `/newidea/` |

## Evidence for the paper

Everything is under `research_evaluation_20261030/`.

| Item | File |
|---|---|
| Protocol and grading rubric | `PILOT50_PROTOCOL.md` (with `protocol_sha256.txt`), `RUBRIC_DANH_GIA.md` |
| Corpus list | `CORPUS_LIST.csv` (20 documents asked about; the index holds 201 document versions) |
| Questions and gold evidence | `frozen_20261030/gold_questions.jsonl`, `evidence.jsonl` |
| Frozen retrieval and pods | `frozen_20261030/retrieval.json`, `production_pods.json` |
| Main-flow answers per question | `mainflow_20261030/answers_images_off.jsonl`: sources sent, claims, citations, timing. The last row of each question is the reported run |
| E1, E2, E4 logs | `runs/E1`, `runs/E2`, `runs/E4`: prompts, selected evidence IDs, outputs, citations, tokens |
| Summary with mean ± SD | `KET_QUA_20261030/BANG_TONG_HOP.xlsx`, `summary_mean_sd.csv` |
| Grades, error cases, comparison | `KET_QUA_20261030/mainflow_manual_grades.csv`, `E4_manual_grades.csv`, `error_cases.csv`, `comparison_full.csv` |
| Retrieval and citation metrics | `KET_QUA_20261030/mainflow_retrieval_citation.csv`, `mainflow_ranking_metrics.csv`, `recall_precision.xlsx` |
| Worked examples, charts | `KET_QUA_20261030/E1_examples.csv`, `E1_table_examples.csv`, `E2_examples.csv`, `E4_examples.csv`, `charts/` |

Main flow, text-only generator input, 50 questions:

| Metric | Value |
|---|---|
| Hit@1 / MRR | 0.800 (40/50) / 0.869 |
| Evidence recall@10 / precision@10 | 0.905 / 0.126 (ceiling 0.146) |
| Required-evidence recall | 0.955 |
| Citation precision / recall | 0.839 / 0.810 |
| Answer accuracy (partial = 0.5) | 0.800 = (33 + 0.5 × 14) / 50 |
| Strict accuracy | 0.660 (33/50) |
| Correct / partial / wrong / no answer | 33 / 14 / 2 / 1 |
| Attribute leakage | 0.051 (5/98 claims) |
| Citation faithfulness | 0.888 (87/98 claims) |
| Key-value recall | 0.826 (24 questions) |

### Test documents

The PDFs are publicly published documents and remain the property of their publishers and authors; each paper in `arxiv_180/` is under the licence shown on its arXiv page. They are included only so the evaluation can be checked. To request removal of a file, open an issue.

### Reading the evidence

- Gold evidence: 73 question–region pairs over 65 distinct regions, including 21 question–table pairs over 16 distinct tables.
- Five answers come from a second run: P047–P050 after a temporary server outage, and P007 after the output-language check that rejected its first answer was removed. First-run rows are kept in `mainflow_20261030/answers_images_off.before_rerun_20261031.jsonl` and `KET_QUA_20261030/before_rerun_20261031/`.
- Labels were assigned with LLM assistance and reviewed by the first author, without a second independent grader.
- The E1, E2 and E4 schedules were stopped part-way, so only questions that completed every configuration are compared.
- Evidence Pods gave no gain over row-wise text on the five E1 table questions. This negative result is kept.
- The image-input main-flow run (`answers_images_on.jsonl`) has 45 of 50 answers, was not rerun and is not used in the paper.
- `mainflow_20261030/judgments.jsonl` (automatic judge) stopped at 14 of 100 answers and is not used, so it is not included.

### Missing logs

- The raw output of P007's rejected first answer was not stored.
- Main-flow prompts are rebuilt by the production code and not stored per question. The `runs/` logs hold full prompts.

### Not in this repository

- Databases, extracted images, model weights and graph exports.
- Gateway host names and e-mail addresses found in third-party text are redacted in the logs, so the hash chain of the `runs/` logs only verifies against the originals.
- The evaluation scripts, the paper source and a static results viewer are in commit [`43d5ff1`](https://github.com/Andydo21/MultiModelAndExtract/tree/43d5ff1). `SOURCE_COMMIT.txt` refers to the evaluation code at that commit.

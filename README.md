# Hierarchical Evidence Pods

Code and evaluation evidence for *Hierarchical Evidence Pods: Structure-Preserving Hybrid Retrieval and Graph-Grounded Question Answering over Technical Documents* (Do Dang An; supervisor Dao Thi Thuy Quynh; Posts and Telecommunications Institute of Technology, PTIT).

The system analyses document structure offline (layout, OCR, table grids, figures, a per-document graph). Online, it retrieves source regions with BM25, BGE-M3 and figure-level ColPali, expands them through the graph, and gives the generator scoped evidence as text, with stable citation IDs.

## Layout

| Path | Contents |
|---|---|
| `ai_extract/` | Offline extraction, indexing, graph projection, inference runtime, pipeline and page-inspector pages |
| `NewIdea/` | Online retrieval, seed selection, graph expansion, Evidence Pods, checked generation |
| `contracts/`, `shared/`, `config/` | Web application, data models, access control, BM25/RRF, generator client, Django settings |
| `runtime/` | Install and start scripts for the GPU inference runtime |
| `serving/` | Kaggle notebook that serves the generator and ColPali behind one gateway |
| `research_evaluation_20261030/` | Evidence for the paper (see below) |

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

- Source documents, databases, extracted images, model weights and graph exports.
- Gateway host names and e-mail addresses found in third-party text are redacted in the logs, so the hash chain of the `runs/` logs only verifies against the originals.
- The evaluation scripts, the paper source and a static results viewer are in commit [`43d5ff1`](https://github.com/Andydo21/MultiModelAndExtract/tree/43d5ff1). `SOURCE_COMMIT.txt` refers to the evaluation code at that commit.

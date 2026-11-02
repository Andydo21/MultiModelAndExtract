# Hierarchical Evidence Pods

Code and evaluation evidence for *Hierarchical Evidence Pods: Structure-Preserving Hybrid Retrieval and Graph-Grounded Question Answering over Technical Documents* (Do Dang An; supervisor Dao Thi Thuy Quynh; Posts and Telecommunications Institute of Technology, PTIT). The paper is at [`paper/Hierarchical_Evidence_Pods.pdf`](paper/Hierarchical_Evidence_Pods.pdf).

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
| `evaluation/` | **Scripts that produced the paper's evidence:** freeze the inputs, run the main flow and the E1/E2/E4 experiments, compute the metrics, and write the results tables. Not needed to run the system | `metrics.py`, `build_pilot50.py`, `benchmarks/corpus20/pilot50_authored.py`, `runners/pilot_metrics.py`, `runners/pilot_freeze.py`, `runners/pilot_mainflow.py`, `runners/pilot_run.py`, `runners/pilot_deliverables.py` |
| `paper/` | **The paper as a PDF** | `Hierarchical_Evidence_Pods.pdf` |
| `research_evaluation_20261030/` | **Evidence for the paper,** from the evaluation of 30–31 October 2026. Nothing here is needed to run the system | See below |

Inside `research_evaluation_20261030/`:

| Folder | What it is |
|---|---|
| `frozen_20261030/` | The fixed inputs: 50 questions with gold evidence, the evidence texts, retrieval rankings, rendered pods, and the per-document graphs |
| `mainflow_20261030/` | Raw per-question logs of the full system on the 50 questions (`images_off` is the text-only run in the paper) |
| `runs/` | Raw logs of the smaller experiments: E1 evidence format, E2 graph expansion, E4 visual input |
| `documents/` | The test documents: the exact PDFs in the index. `pilot_20/` holds the 20 documents the questions ask about; `arxiv_180/` holds 180 arXiv papers that act as distractors. `DOCUMENT_INDEX.csv` maps all 201 indexed versions to a file, SHA-256 and source link (one version is a repeat upload of `DigitalTwinSoftware_en`) |
| `results/` | Final results: CSV and XLSX tables, grades, error cases and charts |

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
   docker compose up -d a3-inference web
   docker compose exec web python manage.py migrate
   docker compose exec web python manage.py ensure_demo_user
   docker compose up -d knowledge-worker entity-worker
   ```

4. **Open** http://127.0.0.1:8000/

   No login is needed: `ensure_demo_user` creates a local admin, and `AutoDemoLoginMiddleware` gives that account to every visitor, without a CSRF check. Compose publishes the site only on `127.0.0.1`. Do not expose it publicly in this state, because anyone who reaches it has admin rights, and a page open in the same browser could send it requests; remove the middleware from `config/settings.py` first.

| Page | URL |
|---|---|
| Dashboard: documents and versions | `/` |
| Extraction status | `/ai-extract/`, `/ai-extract/contracts/<id>/` |
| Page inspector: regions drawn on each page | `/ai-extract/contracts/<id>/pages/<n>/` |
| Questions and cited answers | `/newidea/` |

## Tests

```bash
docker compose exec web python manage.py test ai_extract     # also NewIdea, contracts, shared
python -X utf8 -m unittest discover -s evaluation/tests -t .
```

**Not all tests pass.** Result of 1 November 2026, one app at a time, on the project's existing image with an empty temporary PostgreSQL and no model services or gateway.

| Tests | Run | Passed | Failed | Errors | Skipped |
|---|---|---|---|---|---|
| `ai_extract/` | 560 | 534 | 11 | 11 | 4 |
| `NewIdea/` | 205 | 180 | 8 | 17 | 0 |
| `contracts/` | 70 | 47 | 4 | 18 | 1 |
| `shared/` | 4 | 4 | 0 | 0 | 0 |
| `evaluation/` (on the host) | 24 | 24 | 0 | 0 | 0 |

Most frequent causes among the 69 unsuccessful Django tests:

- 17 errors: `FOR UPDATE cannot be applied to the nullable side of an outer join`.
- 14 errors: `connection already closed`.
- 5 errors: no entity-service URL was set for the test (`MissingSchema`).
- 1 error: `openpyxl` is missing from the image.
- The rest are assertion failures and errors that have not been analysed. Some may be tests that no longer match the code; others may be real defects.

The evaluation in the paper ran the deployed pipeline directly and does not depend on this suite.

## Evidence for the paper

Everything is under `research_evaluation_20261030/`.

| Item | File |
|---|---|
| Protocol and grading rubric | `PILOT50_PROTOCOL.md` (with `protocol_sha256.txt`), `GRADING_RUBRIC.md` |
| Corpus list | `CORPUS_LIST.csv` (20 documents asked about; the index holds 201 document versions) |
| Questions and gold evidence | `frozen_20261030/gold_questions.jsonl`, `evidence.jsonl` |
| Frozen retrieval and pods | `frozen_20261030/retrieval.json`, `production_pods.json` |
| Main-flow answers per question | `mainflow_20261030/answers_images_off.jsonl`: sources sent, claims, citations, timing. The last row of each question is the reported run |
| E1, E2, E4 logs | `runs/E1`, `runs/E2`, `runs/E4`: prompts, selected evidence IDs, outputs, citations, tokens |
| Summary with mean ± SD | `results/SUMMARY_TABLES.xlsx`, `summary_mean_sd.csv` |
| Grades, error cases, comparison | `results/mainflow_manual_grades.csv`, `E4_manual_grades.csv`, `error_cases.csv`, `comparison_full.csv` |
| Retrieval and citation metrics | `results/mainflow_retrieval_citation.csv`, `mainflow_ranking_metrics.csv` |
| Worked examples, charts | `results/E1_examples.csv`, `E1_table_examples.csv`, `E2_examples.csv`, `E4_examples.csv`, `charts/` |

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

### What the codes in the files mean

The logs and tables use short codes. The definitions come from `runs/E1/manifest.json`, `runs/E2/manifest.json` and `runs/E4/manifest.json`.

**E1: how the same evidence is formatted for the generator** (column `format`)

| Code | Meaning |
|---|---|
| A | Flat text; each table row is one line, with cells written as `a ; b` |
| B | JSON with the evidence ID, text and table |
| C | Markdown tables |
| D | Markdown tables (C) plus the heading breadcrumb |
| E | Full Evidence Pods, as produced by the system |

**E2: how far retrieved sources are expanded through the graph** (column `mode`; the paper's names in brackets)

| Code | Meaning |
|---|---|
| A | No expansion (none) |
| B | Adds the previous and next blocks (prev/next) |
| C | B, plus the parent heading, sibling blocks, and figure captions and nearby or inner figure text (structural) |
| D | C, but only inside the heading scope of the retrieved source (scope-aware) |

**E4: what the generator receives** (column `input`)

| Code | Meaning | Name in the paper |
|---|---|---|
| F | Images of the full pages only | Full-page image |
| K | Image crops of the evidence regions only | Region crops |
| O | OCR text only (format A of E1) | OCR text only |
| S | OCR text and captions as Evidence Pods (format E of E1) | OCR + caption pods |

**Other values**

| Value | Meaning |
|---|---|
| `gold` / `retrieved` (columns `evidence`, `evidence_mode`) | The generator receives the human-marked gold evidence / the top 5 sources found by the system |
| `images = off` / `on`, `text_only` / `image_policy` | The generator receives text only (reported in the paper) / also images when the system's policy asks for them (kept for the record, not used) |
| `P001`–`P050` | Question IDs, as in `frozen_20261030/gold_questions.jsonl` |
| `correct`, `partial`, `wrong`, `no_answer` | Answer labels. Accuracy counts `partial` as 0.5 |

### Where each table and figure of the paper comes from

Figures 1 and 3 and Algorithms 1 and 2 describe the method and have no data file behind them, except that the pod excerpt in Figure 3 is taken from `production_pods.json`. Paths are relative to `research_evaluation_20261030/`.

| In the paper | File | How to read it |
|---|---|---|
| Fig. 2, extraction output | `frozen_20261030/graph_leaves.jsonl` | Stored regions of page 2 of the VS-050S2 datasheet (`VS050S2_E`) |
| Table 1, graph relations | `frozen_20261030/graphs.json` | The 20 pilot documents in `documents/DOCUMENT_INDEX.csv`, without version 1000020 (the repeat upload) |
| Table 2, main flow | `results/mainflow_ranking_metrics.csv`, `mainflow_retrieval_citation.csv`, `mainflow_manual_grades.csv`, `summary_mean_sd.csv` | Rows with `images = off` / `text_only` |
| Table 3 and Fig. 4, answers | `mainflow_20261030/answers_images_off.jsonl`, `frozen_20261030/production_pods.json` | Last row of each question |
| Table 4, questions | `frozen_20261030/gold_questions.jsonl` | Asked in Vietnamese; the paper gives an English translation |
| Table 5, E2 | `results/E2_examples.csv`, `runs/E2/` | |
| Table 6, E4 | `results/E4_examples.csv`, `E4_manual_grades.csv`, `e4_offline_latency.json`, `runs/E4/` | |
| Table 7, E1 | `results/E1_examples.csv`, `runs/E1/` | |
| E1 on table questions (text) | `results/E1_table_examples.csv` | P006, P008, P013, P015, P048 |
| ColPali-only sources (text) | `frozen_20261030/retrieval.json` | First ten ranked sources of each question where `streams` is only `colpali` |
| Visual bridge (text) | `mainflow_20261030/answers_images_off.jsonl` | Sources with `expansion_reason = visual_text_shared_entity_candidate` |
| Repeated configurations (text) | `runs/E1/`, `runs/E2/`, `runs/E4/` | Non-failed outputs of the same question, format and variant |

### How each number is computed

`evaluation/runners/pilot_metrics.py` computes every number below from the stored logs, without calling any model. It compares the result with the stored tables and lists any difference. `--write` saves the numbers, each with its numerator, denominator and source, to `results/paper_numbers.csv`.

```bash
python -X utf8 -m evaluation.runners.pilot_metrics research_evaluation_20261030 --write
python -X utf8 -m evaluation.runners.pilot_deliverables research_evaluation_20261030   # summary tables and charts
```

| Metric | Computation | Code |
|---|---|---|
| Hit@1, MRR | Position of the first gold region in the order the sources were sent: rank 1 for a hit; 1/rank for reciprocal rank, 0 if none was sent. Mean over 50 questions | `ranking_row` |
| Evidence recall@10 | Gold regions in the first 10 unique sources ÷ gold regions. Mean over questions | `metrics.evidence_metrics` |
| Evidence precision@10 | Gold regions in the first 10 unique sources ÷ those sources. The ceiling is 0.146 because a question has 1.46 gold regions on average | `metrics.evidence_metrics` |
| Required-evidence recall | Required regions among all sources sent ÷ required regions. Mean over questions | `metrics.evidence_metrics` |
| Citation precision, recall | Sources cited by the claims, compared with the gold regions. Mean over questions | `metrics.citation_id_overlap` |
| Key-value recall | Share of the reference numbers and codes found in the answer after normalization. Mean over the 24 questions that have such values | `key_value_recall`, with `key_values` and `normalize` from `pilot_run.py` |
| Answer accuracy, strict accuracy | (correct + 0.5 × partial) ÷ 50, and correct ÷ 50 | `paper_numbers` |
| Attribute leakage rate | Claims with a leak ÷ all claims, pooled: 5/98 | `paper_numbers` |
| Citation faithfulness | Claims supported by their citations ÷ all claims, pooled: 87/98 | `paper_numbers` |

Regions are matched by `content_hash`, because the same document can be indexed more than once.

The labels and the counts of leaked and unsupported claims are judgments, not calculations. They are inputs, stored in `results/mainflow_manual_grades.csv` and `E4_manual_grades.csv`. `summary_mean_sd.csv` gives per-question means, so its leakage (0.060) and faithfulness (0.890) differ from the pooled values in the paper.

### Test documents

The PDFs are publicly published documents and remain the property of their publishers and authors; each paper in `arxiv_180/` is under the licence shown on its arXiv page. They are included only so the evaluation can be checked. To request removal of a file, open an issue.

### Reading the evidence

- Gold evidence: 73 question–region pairs over 65 distinct regions, including 21 question–table pairs over 16 distinct tables.
- Five answers come from a second run: P047–P050 after a temporary server outage, and P007 after the output-language check that rejected its first answer was removed. Their first-run rows remain in `mainflow_20261030/answers_images_off.jsonl`, above the rows that are reported.
- Labels were assigned with LLM assistance and reviewed by the first author, without a second independent grader.
- The E1, E2 and E4 schedules were stopped part-way, so only questions that completed every configuration are compared.
- Evidence Pods gave no gain over row-wise text on the five E1 table questions. This negative result is kept.
- The image-input main-flow run (`answers_images_on.jsonl`) has 45 of 50 answers, was not rerun and is not used in the paper.
- The paper states that the figure-to-text bridge added 52 passages in 10 questions. That count predates the five second-run answers; the reported run gives 60 passages in 11 questions (`results/paper_numbers.csv`).
- `evidence_precision_at_10` of P047 was first stored as 0.1, using a denominator of 10 instead of its 9 unique sources, and was corrected to 0.111. The reported mean (0.126) and SD (0.060) do not change.
- `evaluation/build_pilot50.py` writes the question and evidence files, then stops because it expects the PDFs under a top-level `documents/` folder. The questions and reference answers it writes match all 50 in `gold_questions.jsonl`.
- `mainflow_20261030/judgments.jsonl` (automatic judge) stopped at 14 of 100 answers and is not used, so it is not included.

### Missing logs

- The raw output of P007's rejected first answer was not stored.
- Main-flow prompts are rebuilt by the production code and not stored per question. The `runs/` logs hold full prompts.

### Not in this repository

- Databases, extracted images and model weights.
- Gateway host names and e-mail addresses found in third-party text are redacted in the logs, so the hash chain of the `runs/` logs only verifies against the originals.
- An earlier copy of the paper source and a static results viewer are in commit [`43d5ff1`](https://github.com/Andydo21/MultiModelAndExtract/tree/43d5ff1).

### Removed tables

Summary tables that repeat other files, the tables of the first run, and a separate copy of the first-run log were removed to keep the folders short; they remain in the Git history. `results/SUMMARY_TABLES.xlsx` is the workbook to read first.

### Renamed files

`PILOT50_PROTOCOL.md` is sealed by `protocol_sha256.txt`, so it keeps the rubric's earlier name `RUBRIC_DANH_GIA.md`, which is now `GRADING_RUBRIC.md`. `results/` was earlier `KET_QUA_20261030/`, and `results/SUMMARY_TABLES.xlsx` was `BANG_TONG_HOP.xlsx`. The evidence commits before this rename used the earlier names, and their JSON logs under `mainflow_20261030/` and `results/` are not valid JSON because of a redaction error, fixed here.

## License

The code is under the [MIT License](LICENSE). It does not cover `paper/`, the PDFs in `research_evaluation_20261030/documents/`, or the models listed under Run, which keep their own terms.

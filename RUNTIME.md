> C?p nh?t 2026-10-14: Granite v? LayoutLM ?? ???c g? kh?i code runtime, endpoint v? script kh?i ??ng. Lu?ng ?ang d?ng Surya ? c?u tr?c ngu?n/GLiNER ? Neo4j ? BGE/ColPali. C?c ph?n b?n d??i nh?c ??n Granite/LayoutLM l? ghi ch?p l?ch s?, kh?ng ph?i h??ng d?n b?t model hi?n t?i.

# Chạy web hợp đồng và các dịch vụ

## Quan hệ bố cục cục bộ — 2026-10-14

`LEFT_OF`/`ABOVE` chỉ nối hàng xóm gần: khoảng cách mép tối đa 80 trên hệ
tọa độ 0–1000, phần chiếu chồng nhau tối thiểu 50% cạnh ngắn hơn, không có
block khác chắn trong hành lang giữa hai bbox. Mỗi block giữ một hàng xóm
gần nhất bên phải và một bên dưới. `NEAR` giữ tối đa hai hàng xóm gần được
cả hai đầu chọn, không xuyên block khác. Không tạo mọi cặp trước/sau trong trang.
`NEXT` vẫn là thứ tự đọc, không chứng minh hai block cùng đoạn hoặc cùng nghĩa.

Graph xem trên trang giữ các cạnh vị trí. A2 ưu tiên cây đề mục, nguồn và bảng
trước khi thêm tối đa 3 cạnh vị trí (8 với câu hỏi hình/bố cục), với vai trò
`supplementary`. Các cạnh được kiểm tra lại bbox/cùng trang/ảnh và dùng tối đa
20% ngân sách graph; khi có cạnh hợp lệ, dành 15% cho phần bổ sung. `NEXT` là
thứ tự đọc, không phải tiếp nối ngữ nghĩa. Prompt yêu cầu không suy diễn quan
hệ mô tả, sở hữu hoặc nguyên nhân từ vị trí. Ngữ cảnh vượt ngân sách báo partial.

Chú thích được chuẩn hóa thành `figure_caption`/`table_caption`. Chỉ tạo cạnh
nguồn hình/bảng → chú thích khi số hiệu có trong chính nguồn của hình/bảng,
khớp chú thích duy nhất và bbox cùng trang/ảnh, thẳng cột, khoảng cách ≤80.
Khi chưa có nhãn riêng thì dùng `RELATED`/`caption_candidate`, giữ unreviewed;
không mượn số hiệu của chú thích cho hình. Dẫn chiếu tường minh được giải quyết
tới chú thích có số hiệu duy nhất trong toàn phiên bản bằng `RELATED`/`references`.
Không gọi Qwen ở A3. Bộ bind gửi nguồn gốc, quote, trạng thái và vai trò cạnh;
ứng viên chú thích có ưu tiên thấp hơn quan hệ chính.

`scratch/normalize_a3_graphs.py` tính thử graph mới từ nguồn đã lưu; `--apply`
đồng bộ các graph mới nhất đang ready và kiểm tra số cạnh thực tế trong Neo4j.
Không chạy OCR/Granite/BGE/ColPali hoặc ghi lại vector. Không chạy khi có lượt
trích xuất mới đang hoạt động.
Thêm `--source-links` để cập nhật liên kết chú thích/dẫn chiếu từ cây đã lưu,
kiểm tra text/bbox/heading_path giữ nguyên. Kết quả nằm trong
`scratch/source-relations-live.json`. Kiểm tra đầu vào A2 với graph thật bằng
`scratch/check_source_relations_live.py`; script không gọi Qwen.

## A3 xử lý từng trang

Với PDF/ảnh, render ảnh trang rồi bắt đầu ColPali trong một tác vụ nền riêng.
Nhánh chính chạy Surya OCR → Granite phân tích cấu trúc → lưu nguồn và Neo4j.
ColPali không chờ OCR; mỗi trang chờ cả hai nhánh ghi xong (hoặc ghi nhận lỗi)
trước khi chuyển trang. Lỗi một nhánh không làm mất kết quả nhánh còn lại.
Surya, Granite, embedding và reranker vẫn suy luận bằng GPU. Runtime local
điều phối các bước bằng một khóa GPU; yêu cầu chồng lên nhau quá 5 giây báo bận,
không tích lũy tác vụ sau khi client timeout. ColPali chạy trên Kaggle nên không
chịu khóa GPU local. Endpoint health vẫn đọc được trong khi OCR đang chạy.
Model chưa dùng giữ trọng số trong RAM; khi đến lượt chuyển lại lên CUDA,
không tải checkpoint lại và không chuyển suy luận sang CPU. Chỉ model text
đang dùng chiếm CUDA, để tránh BGE/reranker/Granite cùng giữ VRAM với Surya.
Sau trang cuối, hoàn thiện cây/graph toàn tài liệu rồi BGE embedding các chunk
một lượt theo batch và ghi Qdrant. Không embedding từng trang trước bước này.
Office vẫn dùng parser hiện có; ảnh nhúng được gửi ColPali sau khi parser lưu nguồn.
Tác vụ nền dùng ThreadPoolExecutor có giới hạn vì ORM/client hiện tại đồng bộ,
không phải toàn bộ pipeline đã chuyển sang asyncio. Không chạy migration hoặc
tái lập chỉ mục dữ liệu cũ khi cập nhật code này.

### Log tiến trình A3

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\show-logs.ps1 -Service Web
```

### Surya ưu tiên tốc độ — 2026-10-13

Mặc định `SURYA_OCR_MODE=blocks`: nhận diện layout rồi OCR các crop, không
thử đọc toàn trang trước. Hai crop chạy đồng thời (`SURYA_CROP_CONCURRENCY=2`)
trên hai slot Surya GPU. Launcher giữ 16384 context mỗi slot, tổng 32768;
không tăng số trang chạy đồng thời. Kết quả crop vẫn trả theo thứ tự nguồn.
`SURYA_OCR_MODE=full_page` còn dùng được khi cần so sánh chất lượng hai chế độ.

Client nhận SSE, hủy sớm khi gặp nội dung lặp hoặc chuỗi dấu cách bất thường,
không đợi hết 12288 token rồi gửi lại cả trang tối đa 3 lần. Mỗi crop lỗi
chỉ được thử lại một lần: lỗi dấu cách thêm lề trắng bên phải ảnh crop,
giữ nguyên pixel nguồn và bbox; lỗi hết token tăng headroom; lỗi nội dung
lặp đổi cách lấy mẫu. Trích xuất thất bại sau lượt phục hồi sẽ báo lỗi,
không ghi/caching output dở dang như nguồn hoàn chỉnh.
Giới hạn mặc định: 80 giây mỗi request, 180 giây cho chuỗi OCR của trang;
được kiểm tra giữa các chunk stream. Đây không phải timeout toàn pipeline.

Phép đo trên ảnh trang 3 của run 15: lượt cũ kẹt ở Surya và timeout sau
600,19 giây; ColPali đã hoàn thành sau 7,6 giây. Nguyên nhân được tái hiện
ở vùng đầu: đọc `v0 := ld state[748]` rồi sinh 2048 token chủ yếu là dấu
cách mà không có EOS. Sau sửa, OCR tuần tự: 93,901 giây HTTP; hai crop
song song: 57,305 giây HTTP, trong đó model/pipeline OCR 53,779 giây.
Output có 9 vùng, 4856 ký tự, các mẫu Figure 3/4, TraceMonkey, mov ecx,
i=3, state[748], js_Array_set đều hiện diện. Đây là kiểm tra nguồn mẫu,
chưa phải CER/WER hay phép đo độ chính xác trên toàn bộ kho. Thời gian
Granite, Neo4j và embedding chưa nằm trong số 57,305 giây này.

Log Models `surya_runtime` ghi prompt, block_index, trạng thái, thời gian,
số token do server trả và số lượt phục hồi. Khi stream bị hủy trước usage,
output_tokens là null, không ước lượng số token từ ký tự. Log Web `surya`
ghi ocr_mode, model_elapsed_s, generation_calls và crop_retries.

Log bắt đầu bằng `A3` và có `run`, `stage`, `event`, `page`, `total_pages`,
`elapsed_s`, số block/chunk/point tùy bước. Các bước gồm `render`, `surya`,
`chunking`, `granite`, `structure`, `colpali`, `neo4j_page`, `wait_colpali`, `neo4j_final`,
`dense_index`, `bge_batch`, `qdrant_dense_write`, `pipeline`.
Mỗi bước chậm phát `waiting` mỗi 15 giây; đây là thời gian chờ thực tế,
không phải phần trăm hoàn thành hoặc thời gian dự đoán. `failed` ghi loại lỗi;
`status=FAILED` báo nhánh thất bại dù các trang đã trích xuất thành công.
Log không ghi nội dung tài liệu, ảnh base64 hoặc token. BGE/ColPali chạy lỗi
vẫn có trạng thái độc lập để phân biệt trích xuất xong với lập chỉ mục xong.
Code mới áp dụng cho các lần chạy tiếp theo; runtime local cần nạp lại khi
không còn tác vụ hoạt động để dùng điều phối GPU mới.

### Giới hạn và đo Granite

Granite dùng SDPA và KV cache. `GRANITE_GENERATION_TIMEOUT` mặc định 120 giây
giới hạn vòng sinh token; `GRANITE_MAX_NEW_TOKENS` vẫn là 8192. Output bị dừng
vì deadline hoặc hết token mà chưa có EOS không được lưu như DocTags hoàn chỉnh.
Deadline được kiểm tra giữa các bước decoder, không ngắt cứng một CUDA kernel.
Cache RAM tối đa 32 ảnh trang giữ kết quả parse thành công theo nội dung ảnh,
checkpoint và token budget; mất cache khi runtime restart. Lỗi không được cache.
Log Models có `granite_runtime`, số token và token/giây mỗi 15 giây. Log Web ở
bước `granite` ghi thêm `device`, `cache_hit`, `output_tokens`, `generation_s`
và `tokens_per_s`. `generation_s` trên cache hit là thời gian tạo kết quả gốc;
`elapsed_s` là thời gian phục vụ lần gọi hiện tại.

Đo thật ngày 2026-10-13 trên ảnh `pdf_page_contract_run_14_p1.png`: Granite
trả 21 phần tử, sinh 1501 token trên `cuda:0`; lần đầu 110,524 giây HTTP,
trong đó generation 99,557 giây. Gọi lại cùng trang trong runtime đó: 0,067
giây HTTP. Đây là một trang, không đại diện thời gian trích xuất toàn tài liệu
và không phải phép đo độ chính xác. Surya còn có thể mất vài phút ở trang dài
hoặc khi full-page OCR phải chạy nhánh dự phòng.

Vòng kiểm tra cuối sau sửa nạp text model trực tiếp trên CUDA: Granite 90,643
giây, cache 0,038 giây, embedding 0,560 giây (1024 chiều), rerank 0,431 giây
(2 ứng viên). Cả bốn endpoint trả HTTP 200; BGE/reranker mỗi checkpoint chỉ
nạp một lần. Bộ test cuối: 363 test, 361 qua, 2 live test tùy chọn bỏ qua.

## Bật nhanh và xem log

Từ `D:\Django_project\Contract_DL\Contract_DL`, khi Docker Desktop đã bật:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start-services.ps1 -Logs
```

Lệnh bật nhóm A2/A3, sử dụng lại container/image/checkpoint hiện có và mở log
live có nhãn Web, Worker, Models, Surya, Neo4j, Qdrant trong cùng terminal.
Launcher đọc inventory Compose một lần và bật/dừng container theo nhóm,
không build image hay chạy migration. Runtime đang nghe ở 8010 được dùng lại.
Startup no longer checks free RAM. `-AllowLowMemory` is retained only for compatibility with older commands.
Execution policy chỉ thay đổi cho tiến trình PowerShell này.

Chỉ xem log, không bật/dừng dịch vụ:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\show-logs.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\show-logs.ps1 -Service Worker
powershell -NoProfile -ExecutionPolicy Bypass -File .\show-logs.ps1 -Service Models
```

`-Service` nhận All/Web/Worker/Models/Surya/Neo4j/Qdrant. `-Tail 100` đổi số dòng
đầu được đọc; `-Once` chỉ in log hiện tại rồi thoát. Log model lấy file stdout/
stderr mới nhất trong `scratch/inference*-out.log` và `scratch/inference*-error.log`;
Surya dùng `scratch/surya-out.log`, `scratch/surya-error.log`. Script in đường dẫn
file đang theo dõi. Ctrl+C chỉ dừng các reader của script, dịch vụ vẫn chạy.
Nếu mới bật notebook Kaggle, cần bảo đảm URL hiện tại được lưu trong `.env`.
`-Seconds 10` theo dõi log trong 10 giây rồi tự dừng reader.

Chạy từ thư mục chứa `manage.py`, dùng cấu hình Neon trong `.env` của thư mục này. Compose truyền cùng `SECRET_KEY` cho web và các dịch vụ để xác thực JWT thống nhất. Các database hiện có là `contract_db`, `workflow_db`, `blockchain_db`; kết nối Neon dùng SSL.

Các image local đã được build trên máy này. Khi thay dependency, build lại bằng:

```powershell
docker compose -f docker-compose.yml -f docker-compose.web-local.yml build web ai-service ai-summary
```

Chế độ mặc định dành cho máy 16 GB là trích xuất/hỏi đáp. Chỉ chạy web, knowledge-worker, Qdrant, Neo4j và yêu cầu khởi động runtime local nếu đủ RAM:

```powershell
.\start-services.ps1
```

Web ở chế độ AI: http://localhost:8000/contract-intelligence/. Gateway cổng 80 chỉ chạy trong chế độ All.

Chuyển sang ký/ghi blockchain (tạm dừng knowledge-worker, Qdrant, Neo4j và runtime model local):

```powershell
.\start-services.ps1 -Mode Ledger
```

Chạy cả hai nhóm cùng proxy/gateway cũ khi máy có đủ tài nguyên:

```powershell
.\start-services.ps1 -Mode All
```

`-SkipLocalInference` chỉ khởi động nhóm container, không nạp model. Nếu RAM trống dưới 4 GB, launcher báo lỗi rõ ràng và không khởi động runtime local; container AI vẫn chạy. Không coi việc bật container là xác nhận model đã sẵn sàng. Các dịch vụ ký/Fabric và proxy cũ đặt restart policy `no`, chỉ bật khi chọn chế độ phù hợp. Script không build image, chạy migration, bootstrap lại ledger hoặc xóa volume; sử dụng lại các container đã có. Chế độ Ledger yêu cầu mạng Fabric đã bootstrap trước đó.

WSL được đặt giới hạn `memory=3GB` trong `%USERPROFILE%\.wslconfig` và `[experimental] autoMemoryReclaim=dropCache`. Script `scripts/apply-wsl-memory-limit.ps1` sao lưu cấu hình cũ nếu có và giữ các tùy chọn khác. Giới hạn áp dụng cho toàn bộ VM WSL 2; thay đổi cần khởi động lại WSL/Docker. Khi dữ liệu/graph tăng, đo lại RAM và OOM của container; 3 GB là cấu hình demo trên máy này, không phải sizing sản xuất. Dừng/bật service không giải quyết lỗi native đã ghi nhận khi đổi text model trong runtime local.

Kiểm tra trạng thái:

- AI: http://localhost:8001/health
- Summary: http://localhost:8004/health
- Workflow: http://localhost:8003/
- Blockchain: http://localhost:8002/health/
- Fabric Gateway: http://localhost:5000/health

AI và summary là proxy của notebook Kaggle. Nếu tunnel ngrok hết hạn, bật lại notebook, cập nhật `KAGGLE_AI_URL` và `KAGGLE_QWEN_SERVICE_URL` trong `.env`, rồi áp dụng biến môi trường mới:

```powershell
docker compose -f docker-compose.yml -f docker-compose.web-local.yml up -d --no-build web ai-service ai-summary workflow-service
```

## Fabric của project này

Mạng dùng container `contract_*`, channel `contracts-channel`, chaincode `ContractVerifyChaincode`, Org1MSP và TLS cho peer/orderer. Ledger được lưu trong named volumes `contract_dl_contract-peer-ledger` và `contract_dl_contract-orderer-ledger`. Đây là mạng demo riêng của Contract_DL; các container và ledger cũ từ thư mục RiskDL được giữ nguyên. Lịch sử trong Neon không được tự động chuyển vào ledger mới.

Bootstrap giữ lại chứng chỉ và channel block khi chạy lại. Không xóa `fabric/crypto-config`, channel block hoặc ledger volumes khi muốn giữ lịch sử. Không chạy `docker compose down -v` để khởi động lại thông thường.

Blockchain Django gọi `http://fabric-gateway:5000` qua mạng Docker. Health của Fabric thực sự evaluate chaincode; health của blockchain kiểm tra cả Neon và Fabric. Lỗi ghi Fabric trả lỗi, không tạo transaction/block giả với trạng thái CONFIRMED. Verify lấy transaction và block từ ledger, đồng thời so sánh hash ledger với hash proof lưu trong database.

Luồng mới tại http://localhost/contract-ledger/ dùng khóa RSA 3072 được quản lý khác duyệt, ký đúng manifest phiên bản và ghi qua outbox có retry/idempotency. Chaincode 2.0 kiểm tra chữ ký và binding công ty/người ký/khóa. API ghi/ký cũ đã ngừng nhận yêu cầu. Worker chạy cùng web qua `start-services.ps1`.

Smoke test ký RSA thật và phục hồi sau mất phản hồi đã thành công tại block 7: đọc lại đúng transaction đã commit, không gửi trùng. Đây là khóa được duyệt trong ứng dụng; chứng thư CA độc lập, HSM và hạ tầng nhiều tổ chức/HA vẫn chưa triển khai. Xem [LEDGER_INTEGRATION.md](LEDGER_INTEGRATION.md) để sử dụng, phân quyền và vận hành.

Theo [tài liệu Fabric 2.5](https://hyperledger-fabric.readthedocs.io/en/release-2.5/cc_service.html), chaincode chạy ngoài peer phải dùng CCID khớp package ID đã cài. Bootstrap lấy đúng một package ID bằng `calculatepackageid`, thay cho ghép tất cả package đã cài.

## A2/A3 and Qdrant

Qdrant starts with start-services.ps1; Windows port: 6335. RAG_INFERENCE_URL selects the gateway base URL for BGE and Surya routes. See [HACKATHON_DEMO.md](HACKATHON_DEMO.md) for configuration, A2 expert drafts and the labeled benchmark.

### Warm BGE/reranker runtime (2026-10-13)

`ai_extract/services/inference_runtime_service.py` owns the text-model cache;
HTTP routes only validate requests and run inference. `RAG_TEXT_DEVICE=cuda`
is the default and is set by the startup script: both embedding and reranker
stay on CUDA. CUDA fails if unavailable; it never silently offloads to CPU.
`auto` remains an explicit portability option (CUDA when available, otherwise
CPU); `cpu` is only used when deliberately selected outside the startup script.
CUDA weights use FP16; CPU weights use FP32. Input tensors follow the weights'
device and output vectors/scores are copied back to CPU before JSON encoding.

There is no free-VRAM estimate that preemptively moves models to CPU. Both
models stay cached on GPU. Inference uses `inference_mode`, embedding batches
of at most eight and rerank batches of four. Actual CUDA OOM reduces batch
size and retries the same inputs down to one, preserving order and token
limits. Persistent OOM returns HTTP 503 rather than switching to CPU or
fabricating vectors. Surya and Granite are not unloaded by this text policy.
Granite defaults to CUDA and its health endpoint distinguishes configured
device from actual loaded device. The existing extraction pipeline releases
Granite after structural extraction finishes, before final dense indexing.

`GET /rag/health` reports loaded devices, per-model load counts and load time.
Models load from the existing local path/cache with `local_files_only=True`
and safetensors; this optimization installs no dependencies and downloads no
checkpoints. Restart the inference process after changing its code; restart
`riskdl_web` and `contract_knowledge_worker` after changing Python web/worker
code (the web starts with `--noreload`).

### A2/A3 orchestration optimization (2026-10-13)

A2 searches one shared text collection containing A3 document leaves, saved
expert questions/answers, Qwen-extracted expert units and current approved
cases. Dense and ColPali corpus lookups run concurrently, once per vector space;
there is no repeated vector query for each document. BM25 is a single corpus
stream for exact terms and degraded inference. Fusion and reranking operate on
all source kinds together. Context expands only after this shared ranking.
Latest document versions are scoped by access without a 100-document shortlist.

Expert submissions index their text with A3's BGE indexer after semantic
extraction; attachments still use full A3. New case approvals enqueue indexing
in the existing knowledge worker. Expert draft/question provenance remains in
UI and Qwen inputs; a saved question is not proof that its premise is true.
Vector hits are hydrated from authoritative source records with company and
fingerprint checks. Deleted, outdated or unauthorized expert points are ignored.

Backfill saved expert text without rerunning Qwen or document extraction:

```powershell
docker exec riskdl_web python -u manage.py index_expert_knowledge --username admin
```

Use `--dry-run` to count records first. Stable point IDs make retries idempotent;
backfill does not delete document vectors. Search retains separate text (BGE)
and page-image (ColPali) vector spaces because their dimensions and encoders differ.

A3 persists only the new page's blocks and keeps source records in the extraction
session for cumulative structure analysis, avoiding a full database reread after
each page. Streamed PDF pages are chunked once by their page callback. PDF handles
close on both success and failure. Rendered images still feed Surya and ColPali;
native PDF text is never used. BGE indexes once after complete structure/graph
finalization; ColPali overlaps each page's OCR/structure processing.

Surya retries a timed-out or disconnected OCR crop once, preserving successful
sibling crops and the original pixels, source ID and token budget. It retains
the page deadline; SDK HTTP retries remain disabled. Runtime JSON logs include
`block_index`, `status`, `retries` and `retry_reason`. A second failure rejects
the page with the final failed block IDs/statuses; partial text is never cached
as successful OCR. Decoder-loop and truncated-output rescues share the same
one-retry limit rather than adding further transport retries.

The launcher sets `SURYA_MANAGED_BACKEND=1` and `SURYA_EXCLUSIVE_GPU=1`.
The inference process owns its local llama-server, reuses offline GGUF files,
checks readiness and restarts a crashed backend before retrying an interrupted
crop. It never terminates an external process. Lifecycle events (`starting`,
`ready`, `released`) appear in model logs; native backend output is appended to
`scratch/surya-managed.log`, included by `show-logs.ps1 -Service Surya`.
Surya releases GPU residency before Granite, BGE or reranking uses CUDA, then
starts again when OCR needs it. Inference remains on GPU. Granite loads weights
directly onto CUDA and retries one clean load after CUDA allocation failure.

If bounded recovery still cannot read a PDF page, the run records it under
`structure.processing.failed_pages`, logs `failed_continuing` and proceeds with
later pages. Successful source blocks and visual vectors remain persisted.
Final status is `FAILED` with explicit incomplete-page warnings, not a complete
document; the UI reloads the retained results without an unhandled HTTP 500.
Rerunning uses completed OCR cache and retries missing pages. This is recovery
from faults, not a guarantee that every damaged/unsupported input is readable.

These changes do not re-extract or migrate stored documents. Regression tests
verify concurrency, inference call counts, source order, cumulative page context
and database reads. Test duration is not a live model latency benchmark.

### Diagnose long ColPali waits

The outer `colpali` timer includes source loading and vector writes, not only
GPU inference. Separate stages now identify `colpali_sources`,
`colpali_schema`, `colpali_embed` and `qdrant_colpali_write`.
For rendered PDF pages the background branch receives the already saved page
ID, image path and document scope from the main thread; it does not reopen
Neon to load the same metadata. Office sources still load their embedded
visuals after persistence. Postgres `DB_CONNECT_TIMEOUT` defaults to 5 seconds
per resolved server address (libpq may try multiple addresses), Qdrant network
requests use 30 seconds, and ColPali uses 5-second connection and 180-second
read timeouts. These limits do not guarantee a whole-page runtime.
Restart web and knowledge-worker after updating code. A failed index remains
explicitly failed; health checks or a Docker restart do not repair old runs.

After all pages, graph reconciliation and dense indexing, A3 makes one final
ColPali pass over only pages whose visual index is `FAILED` or `PARTIAL`.
Successful and intentionally skipped pages are not sent again. This pass
does not repeat OCR, Granite, graph analysis or dense embedding. A rendered
page whose OCR failed can still recover its independent visual index.
Retries upsert the same deterministic point IDs; acknowledged points from
partial attempts are retained and counted once. A second failure is recorded
without further looping. Logs use `colpali_retry` and `colpali_retry_page`;
`structure.processing.colpali_missing_pages` lists pages still missing vectors,
and each page records `colpali_attempts`. These retries apply to new runs in
the current session, not a persisted resume of older failed runs.

### Full A2 answers and Kaggle context

The web defaults to `QWEN_ANSWER_MAX_TOKENS=8192`, up to 96 cited sentences,
and 24 selected sources. These are ceilings, not required answer lengths.
All selected source text is retained during generation, including expert
conditions and graph expansions; there is no second 1200-character crop.
Repeated text within one source document is deduplicated during selection.
Graph expansions have reserved source slots so relevant answers/conditions
are not displaced entirely by retrieval matches.

The updated Kaggle notebook starts Qwen with `--max-model-len 32768`, GPU 0,
one active sequence and chunked prefill. ColPali stays on GPU 1. Change
`QWEN_MAX_MODEL_LEN` in the notebook environment to choose another capacity;
changing the web `.env` cannot resize an already running vLLM engine.

For an already running notebook, paste
`scratch/kaggle_raise_qwen_context.py` into a new cell in the same kernel,
or run the final cell in the updated notebook. It stops only that kernel's
Qwen process tree, reuses cached weights in offline mode and keeps ColPali
and ngrok running. Check `max_model_len=32768` in `/v1/models` afterward.
If GPU KV-cache capacity is insufficient, startup reports the actual error;
it does not silently reduce context. Local web/worker changes require restart.

The context window includes input, image tokens and reserved output together.
Requests that do not fit return `context_limit_exceeded` with actual engine
capacity and requested output budget. The application never quietly shrinks
the requested output budget, discards source text to fit, or treats a response
ending with `finish_reason=length` as a complete answer. Citation and evidence
verification still run before a generated answer is accepted.

### Inspect actual Qwen JSON

New A2 responses expose `qwen_outputs`: each item has its task/stage and the
unchanged successful API response (`choices`, `usage`, `finish_reason`, model
and request ID). Outputs are retained for citation repairs, verification,
expert semantic extraction batches and whole-document review. Malformed or
truncated model content remains inspectable while the answer stays rejected.
Calls are captured within a request-local context, so concurrent requests
cannot share response lists. No prompt, image data or credentials are added.

On `/newidea/`, ask again and open **JSON output của Qwen** below the answer.
Each call shows the model content and expandable original API envelope;
**Tải JSON** exports all captured responses. Expert submissions show the same
panel. Job results persist the outputs under the existing access rules at
`/api/newidea/jobs/?id=<search_id>`; old jobs have no reconstructed raw output.
The fixture tests are not live JSON; `scratch/qwen-output-live.json` is written
only by the real worker/API acceptance script after actual gateway responses.

### OCR text inside figures (2026-10-14)

Surya 2 normally skips blocks labelled Picture/Image/Diagram. A3 renders
the PDF at 150 DPI for page layout and at 300 DPI for internal figure text.
Each page-level image region runs the same Surya LayoutPredictor followed by
the same RecognitionPredictor in block mode exactly once. Text regions are
OCRed as whole regions; inner image regions are ignored and never sent to Granite.
No second text detector, PDF text layer,
pixel-component proposal or Qwen is used. Every child retains its region ID,
parent ID and bbox mapped into original page coordinates.

Cache policy `surya-image-text-only-v12` promotes an exact-image legacy
body cache through `/rag/ocr_visual_regions`. It reads only the visual
crops and preserves cached body text. Each completed OCR region is retained
independently; failed siblings do not erase it. Font-height, character-capacity
and confidence thresholds no longer discard a region labelled text by Surya.
Repetition, transport failure and image descriptions are still rejected as
source transcription. There is no recursive image analysis. Identical full-image
labels do not create a duplicate source box. Every text-labelled region
remains its own chunk. Tiny image crops are placed in a white context frame
for layout inference; original pixels and mapped source boxes are preserved.

Generation deadlines are disabled by default: SURYA_PAGE_GENERATION_TIMEOUT=0,
SURYA_REQUEST_GENERATION_TIMEOUT=0 and GRANITE_GENERATION_TIMEOUT=0. Web OCR
read timeout SURYA_PAGE_READ_TIMEOUT=0 also means no total read deadline.
Positive values opt into a deadline. HTTP connection and stalled stream
timeouts, model token limits and repetition cancellation remain separate.

Historical read-only v8 acceptance on the previously incomplete EVP2 configuration text
crop returned one layout text region and one chunk, preserving all four lines:
No PC required; No visual controller required; No communication program required
for sending image processing results to the robot; A wide range of compatible
cameras. No extraction run, graph or vector write occurred. This is a crop
acceptance check, not whole-document OCR accuracy or latency measurement.
Evidence: scratch/recursive-surya-readonly-region.json. The earlier whole-figure
v8 probe still had the previous time limit and is not the final configuration.

New blocks store `surya_region_id`, `surya_parent_region_id`, `visual_depth`,
`ocr_status` and `chunk_policy=surya_layout_region`. One region is one chunk,
including multiline text; no character-budget split or bbox-centroid aggregation
is performed. Parent image chunks contain no concatenated child text. Source
image nodes CONTAINS child region nodes; each node HAS_EVIDENCE its own chunk.
Internal headings stay in the image scope and cannot alter document headings.
Text chunks follow normal final BGE/Qdrant indexing.
PDF text-layer extraction and Qwen are not used in this OCR branch. Runtime
logs identify `surya_visual_ocr` and mark model calls with `visual_ocr: true`.
The page inspector displays independent source boxes for internal regions.
Hover shows that region's text and parent; click selects that source block. Yellow
dashed source boxes mean incomplete Granite alignment, not an OCR failure.

Granite converts small source-aligned structural crops with closed-item
source-coverage stopping. Source lists, plain paragraphs and long adjacent body text stay
in the source tree without regeneration. The model input masks unrelated
graphics and text; the original stored PNG is unchanged. Timeouts remain
visible as analysis failures rather than being labelled successful output.

For new PDF runs, the already rendered 300-DPI image is saved as
`*_structure.png` and passed to Granite without whole-page downsampling.
Tables need extracted row/column evidence before DocTags table generation;
an empty source or a lone UI label such as “Search results” stays source-only.
Decoder stopping also detects identical closed items at identical coordinates.
Output budgets scale with source content, independently of wall-clock time.
If a terminal region conversion fails, successful sibling observations survive
and the page returns `partial` with `region_failures`, preserving Surya chunks.
Model/OCR errors are not silently presented as verified content.

### Windows paging failure and actual vector recovery (2026-10-14)

Run 27 completed both Granite pages (11/11 and 13/13 structural groups),
but BGE loading failed with Windows error 1455: the paging file could not
back the 2.27-GB safetensors mapping. `RAG_STREAM_CHECKPOINTS=1` constructs
the model on meta, allocates FP16 weights on CUDA, and reads existing tensors
in 8-MB blocks. This makes no converted checkpoint or new download. Inactive
text models are released rather than copied into the Windows paging pool;
Granite is fully unloaded before embedding/reranking in this mode.

Real acceptance used all 121 eligible source chunks from run 27, producing
121 normalized 1024-dimensional vectors. The BGE -> Granite -> BGE GPU
handoff completed with identical vectors before/after reload (max difference
0). A dense-only retry then stored 121 actual Qdrant points. Read-only audit
confirmed 748 Neo4j nodes and 1,924 edges for the unchanged run snapshot;
vector search returned the Runtime/controller/camera source passage. OCR
warnings remain, and ColPali is outside this recovery check.

Retry the existing text index without OCR, graph rebuilding or visual calls:
`docker exec riskdl_web python manage.py reindex_contract_versions --version-id 153 --dense-only`
The retry uses the application's indexer and updates only its index status;
source text/bboxes and Neo4j structure are not edited. Completed-run iteration
uses keyset queries, avoiding named-cursor failures with transaction-pooled
Postgres. Evidence tools: `scratch/verify_dense_embeddings_readonly.py`,
`scratch/verify_gpu_handoff_readonly.py`, `scratch/verify_a3_run_readonly.py`.

Historical v7 EVP2 live run 25 (contract 122/version 153) completed in 68.976 seconds,
reusing the v7 OCR cache created by live run 23. Granite completed both pages
on CUDA in 19.76 and 28.56 seconds, with no failed analysis pages. Neo4j was
ready and BGE/Qdrant stored 68 text points. The main page-1 figure retains 18
internal text regions, including Guidance/Runtime labels and search functions.
OCR remains partial for small rejected crops. ColPali still failed with HTTP
404, so the visual index is not ready. The 68.976-second total is a cache
rerun timing, not a cold OCR benchmark. Evidence: scratch/evp2-optimized-run.json.

Historical v5 live acceptance used the saved run 19, page 7 image through the production
extractor/cache path: two figures, 42.027 seconds visual inference and
45.918 seconds end to end. Body OCR was cached. Printed labels `Tree Call`,
`Nested Tree` and `Exit Guard` were recovered; some tiny index labels were
misread. One optional crop failed and one implausibly long output was
rejected. This read-only probe did not rewrite old blocks or index vectors.
Use **Chạy lại trích xuất** to persist the new OCR result and rebuild its
indexes. These figures are not a whole-document timing or OCR accuracy
benchmark. Evidence: `scratch/visual-ocr-live.json`.

### LayoutLMv3 provider and OCR v12 (2026-10-14)

`A3_STRUCTURE_PROVIDER=layoutlmv3` replaces Granite in the active web/worker
pipeline. The host uses the official `microsoft/layoutlmv3-base` checkpoint
on CUDA, with external Surya words and source-region bounding boxes.
No Tesseract, PDF text-layer extraction, or Qwen is added to A3.
The base encoder has no trained project classification/relation head:
`fine_tuned=false`, `task=base_features`. Its region features are not treated
as labels, relations, or extra BGE vectors. Source labels and hierarchy rules
remain authoritative. Long inputs use overlapping 512-token windows; all
source words must be covered. Logs use `stage=layoutlmv3` and `layoutlm_window`.

OCR v12 uses the existing 300-DPI rendered source for body block recognition,
maps boxes back to preview coordinates, and keeps each ListGroup as one chunk.
Blocks get at least 1024 output tokens; the one bounded retry gets at least
2048. Repetition cancellation remains enabled. Older body caches cannot
bypass this policy; successful body OCR is reusable even when image OCR is
partial. Images are inspected only once for inner text, never for inner images.
Duplicate text is removed only when equal normalized text and overlapping
source coordinates agree. Original source/graph/vector records are not edited.

Read-only live verification on the saved run 28:
- Failed ListGroup block 2820 recovered all four Runtime bullets in 2.288 s
  of model processing, with no failure or retry; source remained unchanged.
  Evidence: `scratch/layoutlm-list-readonly-result.json`.
- LayoutLMv3 encoded both saved pages on CUDA: 243 words/1 window and
  717 words/4 windows; no page failed. Warm timings were 0.414 s and 0.200 s,
  excluding initial model load and HTTP/image preparation.
- A following BGE request returned HTTP 200 and a 1024-dimensional vector.
  Reproduce with `docker exec riskdl_web python scratch/verify_layoutlm_run_readonly.py 28`.
These checks are not a full fresh extraction run or an OCR accuracy benchmark.

### Active Surya + GLiNER pipeline (2026-10-14)

`A3_STRUCTURE_PROVIDER=source`, `A3_ENTITY_PROVIDER=gliner` replace the
LayoutLMv3 stage. Surya labels and explicit source numbering build the document
tree. The trained multilingual NER model `urchade/gliner_multi-v2.1` then
extracts source-bound mentions on CUDA. GLiNER 0.2.29 was installed while
retaining torch 2.5.1+cu121 and transformers 5.16.1. Checkpoint provisioning:
`scratch/a3-runtime/Scripts/python.exe scratch/install_gliner_runtime.py`.
The 1,155,830,112-byte checkpoint is not smaller than Granite merely because
the task is faster; NER inference does not regenerate document text.

Every mention stores an allowlisted type, exact character offsets, original
text, model score and source leaf ID. Repeated spelling does not merge parties.
Projection creates `source -MENTIONS-> mention -HAS_EVIDENCE-> source`.
Mentions are unreviewed model labels; no semantic business edge is inferred
from co-occurrence. A missing runtime preserves source extraction, records
`entity_incomplete_pages` and surfaces warnings instead of claiming NER success.

Read-only acceptance through the actual GPU HTTP route:
- The explicitly synthetic Vietnamese payment sentence yielded two
  organizations, a money amount and a duration; no semantic relation.
  First call including model load: 19.383 s.
  Evidence: `scratch/gliner-contract-readonly-result.json`.
- Saved run 29 sources yielded 71 candidate mentions and 71 matching source
  evidence links in memory, with no extraction error, 2.444 s warm.
  A following BGE request returned HTTP 200 / 1024 dimensions.
  Reproduce: `docker exec riskdl_web python scratch/verify_gliner_graph_readonly.py 29`.
No old blocks, Neo4j snapshots or Qdrant points were rewritten by these probes.
Candidate counts and source-span validation are not an entity accuracy score.

For contract management, domain fact nodes (obligation, payment, deadline,
condition, penalty) must be separate from bare NER mentions. Predefine role
edges and require supporting source and applicability conditions. A duration
such as “30 days after acceptance” is not an absolute due date until the
acceptance event is known. Party identity requires verified identifiers/roles,
not shared spelling. GLiNER NER alone does not fill this business-fact layer.
# Shared entity registry (Surya + GLiNER)

The A3 source graph keeps each extracted mention, assertion and quotation in its
own tenant/version/run snapshot. `A3Entity` nodes form a separate identity
registry; occurrences point to them through `RESOLVES_TO`. Registry membership
is an unreviewed model inference, not independent evidence for contract facts.
Amounts, dates, obligations and conditions remain contract-local.

Completed graph snapshots queue identity work for `entity-worker` in
`docker-compose.web-local.yml`. This worker uses existing BGE embeddings to find
bounded candidates and the existing multilingual GLiNER RelEx checkpoint to
judge source pairs. It does not call Qwen. Names/similarity alone never merge
identities; unresolved or competing candidates remain separate. The current
checkpoint has not demonstrated reliable identity resolution in the real probe.

Run `python manage.py resolve_document_entities --watch` to watch the durable
queue, or `python manage.py resolve_document_entities --run-id RUN_ID` to requeue
a completed run through normal graph synchronization. Source content is not
rewritten. Jobs have a renewable lease and three attempts; requeue an exhausted
job after fixing its dependency. Source-hash-matched embeddings are reused on
retries. Old snapshot cleanup preserves registry entities with remaining source
occurrences in other contracts, and removes unreferenced entries.

Lexical lookup is indexed by tenant/type. The Neo4j ANN lookup currently takes
100 global candidates then filters by tenant/type; lexical lookup complements
it, but this bounded ANN pool is not a guarantee of complete multilingual recall
in a very large multi-tenant deployment. A2 follows shared identities only into
SQL-authorized current snapshots and reranks discovered passages before use.


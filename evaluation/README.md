# Evaluation & Audit Framework

Package `evaluation/` ở gốc dự án là điểm vào của bộ thực nghiệm. Code nghiệp vụ
production và log cũ không được sửa. Các artifact cũ ở `NewIdea/evaluation/artifacts`
được giữ nguyên. Framework hỗ trợ truy vết và phát hiện sai lệch; không tự bảo đảm
bài báo đúng, đạt chuẩn phản biện hoặc có ý nghĩa thống kê.

## Cấu trúc

- `schema.py`: Question, GroundTruth, RunResult; xác thực subject–attribute–value–condition.
- `provenance.py`: Git SHA/dirty state, hash mã và PDF, cấu hình, phần cứng máy cục bộ.
  GPU cục bộ không được nhận là GPU inference từ xa.
- `metrics.py`: reviewed ALR/accuracy/faithfulness; citation ID overlap và evidence recall riêng.
- `dataset_loader.py`: JSON/JSONL thật, review bắt buộc và kiểm tra tài liệu không trùng split.
- `ablation_adapters.py`: A–E, graph A–D và cầu nối renderer NewIdea thật qua lazy import.
- `reporters.py`: CSV, Markdown, LaTeX, paired tests và Holm correction.
- `runners/`: E0 đối soát/replay, E1 Pods, E2 graph, provider dùng chung.
- `benchmarks/pilot50_gold.jsonl`: **đang trống**, không có gold giả.
- `benchmarks/historical_audit.json`: selector file/ô Markdown; không nhập tay kết quả.
- `contracts/management/commands/run_experiment.py`: CLI Django, không cần query database.

## E0: làm trước

```powershell
.venv/Scripts/python.exe -B -X utf8 manage.py run_experiment --task E0 --reconcile
.venv/Scripts/python.exe -B -X utf8 manage.py run_experiment --export-report
```

Có thể thêm `--settings=config.intelligence_test_settings` để chạy command bằng
cấu hình SQLite cô lập. Command tắt system/migration checks và không ghi DB.
Standalone tương đương: `python -m evaluation --task E0 --reconcile`.

Mỗi lần chạy có folder UTC mới trong `evaluation/artifacts`, không đè artifact
cũ. `historical_raw.jsonl` lưu nguyên record archive và đường dẫn/hash/JSON pointer.
`reconciliation.csv/json` so từng trường giữa bản Markdown sửa đổi, bản cũ và raw
JSON. Thời gian được so bằng Decimal theo độ chính xác **được in**; token so nguyên.
`MATCH_VERIFIED` chỉ dành cho trường số khớp sau quy tắc làm tròn đã công bố.
Nó không xác minh đáp án, nguyên nhân cải thiện hoặc đủ provenance.

Trong dữ liệu hiện có: Q1/Q2 không theo thứ tự `number`; Markdown chỉ có Q2.
Không được gán số liệu J của Q1/Q3/Q4 cho cấu hình M. Bản sửa có các giá trị phù
hợp với log, nhưng bản cũ có số khác và những ca M không có log. Cần xem báo cáo
tự động thay vì coi mọi bản thảo đều “khớp 100%”.

Tổng các stage là tổng số học, chưa chứng minh không chồng lặp. Chênh lệch với
HTTP elapsed được ghi `unknown`, không tự gán cho mạng/queue. Token capture có thể
chưa gồm planning. Báo cáo lịch sử không sinh mean/SD/p-value từ một quan sát.

```powershell
.venv/Scripts/python.exe -B -X utf8 manage.py run_experiment --task E0 --repeat 5 --export-report
```

Lệnh trên **gửi nội dung evidence lưu trong log tới gateway Qwen cấu hình trong
`.env`**. Chưa chạy ở lần triển khai này; cho phép truyền dữ liệu tới gateway đang
chờ xác nhận từ bước trước. Đây là prompt mới trên evidence archive, không phải
tái lập nguyên trạng: prompt lịch sử đầy đủ, corpus/checkpoint/GPU cũ chưa biết.
Chỉ các ca đã có archive được lặp (4 J + 1 M); không tạo thêm ca M thiếu dữ liệu.
Kết quả marked exploratory, không tự chấm chất lượng hay tuyên bố causal speedup.

## Corpus và gold

```powershell
.venv/Scripts/python.exe -B -X utf8 manage.py run_experiment --prepare
```

Đọc PDF thật trong `documents/`, tạo snapshot/hash, corpus CSV, page CSV và các
vùng chữ có bbox. Đây là **lớp chữ PDF để chuẩn bị annotation**, không giả làm
OCR Surya/A3. Trang không có lớp chữ là `needs_ocr`. Annotation queue chứa text
thật với câu hỏi, đáp án và tuple trống. Không tự dựng graph từ thứ tự trang.

Pilot 50 cần câu hỏi/đáp án/evidence được duyệt độc lập, gồm bảng, đối chiếu thiết
bị, chú thích/điều kiện, nhiều nguồn và sơ đồ. Mỗi câu cần `split`, `categories`,
`reference_answer`, `ground_truth_tuples`, gold/required evidence IDs, vị trí
document/version/page/region, `reviewer`, `review_status=approved`.
Mỗi tuple phải có evidence_ids và condition (có thể null).
Q1–Q4 chỉ dùng regression. Tập test 200–300 câu cần tài liệu chưa dùng phát triển;
kiểm tra document ID không đủ để loại bản gần trùng/cùng họ/cùng phiên bản.
Lịch sử sử dụng và split theo họ tài liệu vẫn cần người đánh giá xác nhận.

## E1/E2

Điền revision checkpoint **đã kiểm chứng** và phần cứng inference thực vào config.
Sau khi đổi mã/corpus/model, tạo snapshot mới. Runner kiểm tra hash trước khi chạy.
Không dùng tên model mặc định làm chứng cứ checkpoint hay reranker thực tế.

```powershell
.venv/Scripts/python.exe -B -X utf8 manage.py run_experiment --task E1 --formats A,B,C,D,E --repeat 3 --dataset PATH_TO_GOLD.jsonl --evidence PATH_TO_EVIDENCE.json --snapshot PATH_TO_SNAPSHOT.json --mode gold --export-report
.venv/Scripts/python.exe -B -X utf8 manage.py run_experiment --task E1 --repeat 3 --dataset PATH_TO_GOLD.jsonl --evidence PATH_TO_EVIDENCE.json --snapshot PATH_TO_SNAPSHOT.json --mode retrieved --retrieved PATH_TO_RETRIEVAL.json --export-report
.venv/Scripts/python.exe -B -X utf8 manage.py run_experiment --task E2 --formats A,B,C,D --repeat 3 --dataset PATH_TO_GOLD.jsonl --evidence PATH_TO_EVIDENCE.json --snapshot PATH_TO_SNAPSHOT.json --mode retrieved --retrieved PATH_TO_RETRIEVAL.json --graph PATH_TO_GRAPH.json --export-report
```

PATH_TO biểu thị file thật do người dùng cung cấp, không phải sample fixtures.
Pilot trống sẽ bị từ chối. Không có fallback tạo câu hỏi/gold hoặc điểm giả.

E1 giữ cùng tập và thứ tự evidence, table cells, generator, seed theo repetition.
A là text phẳng; B là JSON content projection; C là Markdown; D thêm breadcrumb;
E thêm scope/metadata/links. A vẫn giữ **nội dung** tiêu đề cột dưới dạng text:
xóa thông tin tiêu đề sẽ đồng thời thay evidence, làm hỏng đối chứng định dạng.
B dùng content projection để không trộn hiệu ứng JSON với việc thêm metadata.
Token dùng `/tokenize` server thật, dưới cùng ceiling và tỷ lệ max/min mặc định
không quá 1,25. Vượt ngưỡng là lỗi được ghi, không cắt nguồn hay pad để khớp số.

`experiment.renderer=canonical` là adapter evaluator độc lập, không giả nhận là
toàn bộ production. Chọn `production` để E gọi đúng `answer_source_pods` và
`render_source_pods` NewIdea; evidence phải kèm `production_source` thực. Adapter
từ chối thay đổi thứ tự do grouping. Sự tương đương nội dung và token giữa renderer
production và canonical cần kiểm tra trên snapshot thật trước khi diễn giải E–D.

E2 modes: A none, B previous/next, C structural, D scope-aware. Graph input là
node/cạnh thực đã export và duyệt, không phải cạnh sinh từ annotation queue.
Nodes phải trùng evidence; cạnh có source/target/type. Structural dùng PARENT,
CHILD,TABLE_HEADER,FOOTNOTE,CONDITION,REFERENCE. Các mode chặn vượt document/version;
scope-aware còn chặn sai scope. Trace giữ lý do nhận/loại và độ sâu. Đây là
chính sách expansion của evaluator, không gọi đó là production graph ablation
nếu chưa ánh xạ cạnh/scope từ snapshot production đã xác minh.

Retrieval JSON phải có corpus_sha256, dataset_sha256, retriever, reranker_actual
(hoặc disabled có chủ đích); từng question có question_sha256 (hàm digest),
evidence_ids đúng thứ tự và latency_ms nếu đo được. BM25 không đổi tên thành BGE sparse.

Timing E1/E2 là **frozen-evidence replay**; không gồm OCR offline, retrieval gốc,
queue production. Graph latency E2 có thể gồm HTTP tokenization. Tokenization
cache làm end-to-end phụ thuộc warm state; dùng generation_http_ms cho phép đo
sinh riêng và ghi rõ cache policy khi diễn giải. Chưa có production end-to-end
benchmark ở đây. Không sinh visual tokens/GPU memory khi provider không đo.

## Scoring và báo cáo

Raw JSONL được append + fsync, có sequence và hash chain. Manifest lưu schedule,
config và hash input. Đây là cơ chế phát hiện chỉnh sửa, không phải kho lưu trữ
bất biến có chứng thực bên ngoài. Báo cáo không ghi đè raw log; export vào folder mới.

```powershell
.venv/Scripts/python.exe -B -X utf8 manage.py run_experiment --export-report --run-dir PATH_TO_RUN --reviews PATH_TO_REVIEWS.json --test wilcoxon
```

Review tham chiếu record_sha256, có reviewer và approved; accuracy là 0/1 theo
rubric. ALR là leaked_assertions/eligible_assertions, faithfulness là
supported_cited_claims/cited_claims. Mẫu số 0 hoặc chưa review là null.
Regex chỉ phát cảnh báo; câu phủ định/so sánh có thể match nhầm. Citation ID overlap
không chứng minh nguồn hỗ trợ nội dung. Required-evidence recall đo ID được chọn,
không chứng minh model hiểu đúng điều kiện; việc đó cần đánh giá semantic.

CSV/Markdown/LaTeX xuất từ raw log đã kiểm hash. Mean ± sample SD tính trên mean
của từng repetition đầy đủ. Báo cáo có số scheduled/attempted/failed/missing.
So E–D theo cặp câu và repetition, chỉ khi evidence hash giống nhau. Wilcoxon
hoặc paired t-test dùng **câu hỏi** làm đơn vị, lấy mean qua repetitions trong
mỗi câu; tối thiểu 5 câu đủ cặp. Không coi các lần lặp cùng câu là mẫu độc lập.
P-value không được tạo khi thiếu cặp hoặc phương sai không phù hợp; có Holm
correction cho nhóm metric được kiểm định. Pilot vẫn mang tính thăm dò.

E3/E4 chưa triển khai/chạy. Cần holdout đã duyệt, Text-only LLM thật, VLM và
đo visual tokens, offline/online latency, GPU memory thực trước khi báo cáo.

## Kiểm thử phần mềm

```powershell
.venv/Scripts/python.exe -B -X utf8 -m unittest discover -s evaluation/tests -v
```

Fixtures chỉ nằm trong test/tempdir; không được xuất vào benchmark hay artifact
thí nghiệm. Kết quả test phần mềm không phải độ chính xác hoặc latency model.

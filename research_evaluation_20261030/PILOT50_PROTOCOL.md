# Protocol pilot50: E1, E2, E4

File này được viết **trước khi chạy generation**, nhằm cố định cách chấm và tránh chỉnh tiêu chí sau khi đã thấy output. SHA-256 của file được ghi vào `protocol_sha256.txt` trước lượt gọi model đầu tiên.

## Dữ liệu

- `frozen_20261030/gold_questions.jsonl`: 50 câu P001–P050 trên 20 tài liệu (phiên bản 1000000–1000019). Đây là bộ dự thảo `pilot50_current_*`, được đánh dấu `approved` với `review_scope=light_text_check`. Cách duyệt: so câu hỏi, đáp án và tuple với text evidence. Ảnh trang không được xem lại. Người duyệt **không độc lập**: là trợ lý AI, được chủ dự án chấp nhận ngày 2026-10-30.
- Đây là bộ **pilot**: 20 tài liệu này đã dùng để phát triển, nên kết quả **không phải test độc lập**. Q1–Q4 cũ không có trong bộ này.
- Phân loại: condition 31, table 18, comparison 13, footnote 11, text 9, procedure 7, identifier 5, figure_text 3, causal 3, multi_evidence 3, multi_document 2. Câu về sơ đồ (figure_text) chỉ có 3, quá ít để kết luận riêng cho nhóm này.
- Retrieval là production thật: `search_knowledge(generate=False)` gồm BM25, BGE-M3 dense, ColPali, RRF và BGE reranker, chạy với user 1000000 trên toàn bộ 201 phiên bản. Lưu top-10 nguồn `query_match` theo đúng thứ tự xếp hạng.
- Graph cho E2 export trực tiếp từ Neo4j (node và cạnh của snapshot hiện tại), kèm toàn bộ leaf của các phiên bản liên quan.
- Hai vùng evidence được coi là cùng một nguồn khi chúng **ở hai phiên bản khác nhau** và có cùng `content_hash` (bản upload trùng của cùng PDF). Text lặp lại trong cùng phiên bản thì không tính là gold.

## Sinh câu trả lời

- Model: `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` trên gateway Kaggle (vLLM, `max_model_len=16384`). Revision tokenizer: `536a35794df8831aa814970ee8f89eff577e7718`. GPU Kaggle chưa xác minh.
- Tham số: temperature 0, top_p 1, max_tokens 2048, seed = 1729 + lần lặp. Mỗi cấu hình chạy 3 lần, theo lịch đã xáo trộn bằng seed 1729.
- Output là JSON schema strict gồm `answer` và `claims[]` (subject, attribute, value, condition, evidence_ids). `evidence_ids` chỉ được lấy từ các ID đã gửi cho model.
- Token đếm bằng tokenizer local `/rag/tokenize`. Runner kiểm tra số này phải bằng `usage.prompt_tokens` của API; lệch là lỗi. Nguồn không bao giờ bị cắt hay pad cho vừa ngân sách.

### E1: Evidence Pods

Giữ cố định tập evidence, model và **thứ tự nguồn**: dùng thứ tự gom nhóm của pods production cho cả năm định dạng.
- A: text phẳng; bảng có cấu trúc ô được ghi thành dòng `a ; b`.
- B: JSON `{evidence_id, text, table}`.
- C: bảng Markdown.
- D: C cộng breadcrumb tiêu đề.
- E: production `answer_source_pods` + `render_source_pods`, tức thêm scope, metadata và các link.

Một bảng chỉ được hiển thị thay cho text OCR khi chuỗi ô ghép lại khớp đúng text OCR, theo cùng tiêu chí production. Chạy hai chế độ evidence: `gold` và `retrieved` (top-5 production).

### E2: Mở rộng graph

Seed cố định là top-3 nguồn production. Mọi mode dùng chung một trần 6.000 prompt token, trình bày theo định dạng E canonical, mở rộng 1 bước cấu trúc.
- A: không mở rộng.
- B: các khối `NEXT` trước và sau.
- C: B cộng heading cha qua `CONTAINS`, các khối anh em (trừ khi cha là gốc tài liệu) và các link caption/near-text/figure-text.
- D: C nhưng chỉ nhận ứng viên cùng heading path với seed, hoặc là heading trực tiếp của seed.

Ứng viên được nhận theo thứ tự seed rồi theo khoảng cách cấu trúc. Trace ghi lý do nhận/loại: `already_selected`, `no_text`, `scope_boundary`, `token_budget`, `missing_leaf`.

### E4: Biểu diễn hình ảnh

Tập con gồm các câu có bảng hoặc hình, chỉ dùng evidence gold.
- F: ảnh cả trang.
- K: ảnh crop vùng evidence.
- O: chỉ text OCR (định dạng A).
- S: OCR + caption có cấu trúc (pods production).

Visual token = `prompt_tokens` của API trừ số token của cùng messages khi bỏ ảnh. GPU memory **không đo được** vì inference chạy từ xa, nên ghi null. Offline latency là wall-time của cả run A3 chia cho số trang, gồm toàn bộ pipeline chứ không riêng OCR.

## Chấm điểm (cố định trước khi chạy)

- **Judge tự động**: chính Qwen2.5-VL-7B (cùng model sinh), temperature 0, JSON schema strict. Đầu vào: câu hỏi, đáp án mẫu, tuple gold, text evidence gold, text của evidence được trích dẫn, câu trả lời và claims. Đây **không phải đánh giá của người**. Judge cùng model có thể thiên vị. Một mẫu ngẫu nhiên sẽ được kiểm tra thủ công và báo cáo tỷ lệ đồng ý.
- Nhãn: `correct` / `partial` / `wrong` / `no_answer`, theo `RUBRIC_DANH_GIA.md`.
  - **Answer Accuracy (chính)** = điểm từng phần: correct 1, partial 0,5, wrong và no_answer 0.
  - **Strict accuracy** = chỉ tính correct là 1.
- **Attribute Leakage Rate** = số claim gán sai chủ thể/thuộc tính/điều kiện ÷ số claim có giá trị.
- **Citation Faithfulness** = số claim có trích dẫn mà evidence được trích thực sự nói đúng nội dung đó ÷ số claim có trích dẫn.
- **Key-value recall** (tất định, không dùng model): tỷ lệ số liệu/mã trong tuple gold (dài ≥ 2 ký tự) xuất hiện trong câu trả lời sau khi chuẩn hóa khoảng trắng và dấu thập phân.
- **Evidence Recall@10, Precision@10, Required-Evidence Recall**: tính trên ID được chọn, sau ánh xạ bản trùng.
- **Thống kê**: mean ± SD lấy trên giá trị trung bình của từng lần lặp. Kiểm định Wilcoxon ghép cặp với đơn vị là câu hỏi (trung bình qua các lần lặp), hiệu chỉnh Holm. Không coi các lần lặp là mẫu độc lập. Ghi đủ số lượt failed, over_budget và chưa chấm.
- **So sánh chính**: E1 E so với D (cả gold và retrieved); E2 D so với A và D so với C.

## Ngoài phạm vi lần chạy này

E3 (tài liệu chưa thấy, 200–300 câu, LLM text-only thật, VLM khác) chưa chạy. Lý do: chưa có tập test độc lập đã duyệt, và chỉ có một model được phục vụ (Qwen2.5-VL). GPU local là RTX 3050 Ti 4 GB, không đủ chạy một LLM 7B text-only để so sánh công bằng.

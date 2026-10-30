# Kết quả thực nghiệm ngày 2026-10-30

## Đã chạy: luồng chính (full system), 50 câu pilot, mỗi chế độ 1 lần

Mỗi câu chạy qua `search_knowledge(generate=True)` của NewIdea: planner → BM25 + BGE-M3 + ColPali → RRF → reranker → mở rộng graph → evidence pods → sinh câu trả lời có kiểm tra citation. Hai chế độ:
- **ảnh tắt**: generator chỉ nhận text.
- **ảnh bật**: ảnh được gửi khi chính sách production yêu cầu.

Retrieval chạy một lần cho mỗi câu và dùng chung cho cả hai chế độ.

| | Ảnh tắt | Ảnh bật |
|---|---|---|
| Số câu có câu trả lời | 45/50 | 45/50 |
| Số câu có gửi ảnh | 0 | 27 |
| Key-value recall (20 câu có số liệu/mã) | 0,842 ± 0,322 | 0,775 ± 0,368 |
| Số câu đủ hết số liệu/mã | 15/20 | 13/20 |
| Nguồn gold có trong ngữ cảnh gửi model | 0,955 | 0,955 |
| Thời gian sinh (median) | 43,7 s | 44,0 s |
| Thời gian retrieval (median) | 67,2 s | (dùng chung) |

**Các câu không có câu trả lời:**
- P007: bị kiểm tra của production từ chối (`response_rejected`) ở cả hai chế độ.
- P047–P050: mất ở cả hai chế độ do gateway Kaggle/ngrok offline giữa lượt chạy (`gateway_unavailable`, `ERR_NGROK_3200`).

**Phạm vi diễn giải:**
- Key-value recall là so khớp chuỗi số/mã trong câu trả lời, không phải chấm ngữ nghĩa. Câu trả lời **chưa được chấm** Accuracy / Attribute Leakage / Citation Faithfulness: judge mới chạy 14/100 lượt thì gateway tắt, nên các lượt đó không được dùng.
- Đây là bộ pilot trên tài liệu đã dùng để phát triển, không phải test độc lập.

## Recall / precision (45 câu có câu trả lời)

| | Ảnh tắt | Ảnh bật |
|---|---|---|
| Evidence Recall@10 | 0,919 | 0,919 (retrieval dùng chung) |
| Evidence Precision@10 | 0,120 (trần 0,136, vì mỗi câu chỉ có 1,36 vùng gold) | 0,120 |
| Required-Evidence Recall (mọi nguồn gửi model) | 0,956 | 0,956 |
| Citation precision (nguồn được trích là gold) | **0,859** | 0,741 |
| Citation recall (vùng gold được trích) | **0,837** | 0,737 |

Gold được đối chiếu qua `content_hash`, nên bản upload trùng của cùng một PDF ở phiên bản khác vẫn được tính. File: `mainflow_retrieval_citation*.csv`, `recall_precision.xlsx`.

## Ví dụ thực tế E1 / E2 / E4 (mỗi bộ 4 câu, chạy 1 lần)

Chọn các câu đã chạy đủ mọi cấu hình, không chọn theo kết quả. File: `E1_examples.csv`, `E2_examples.csv`, `E4_examples.csv`.

- **E2** (P003, P007, P022, P027):
  - P007: không mở rộng thì recall 0,50; thêm khối trước/sau thì đủ 1,00.
  - Scope-aware so với structural: vẫn giữ đủ evidence, ít hơn 34–62% token, graph latency thấp hơn 3,5–8,6 lần.
- **E4** (P013, P028, P039, P040):
  - Ảnh cả trang: đúng 100% số liệu, nhưng tốn khoảng 2.760 visual token, tổng 2.934 token.
  - OCR + caption có cấu trúc: cũng đúng 100%, chỉ 616 token.
  - Ảnh crop: kém nhất (0,33).
- **E1** (P002, P018, P023, P029):
  - Đều là câu text không có bảng, nên các định dạng gần như không khác nhau. Mọi định dạng đều chỉ trích nguồn gold.
  - D (breadcrumb) có citation recall cao nhất khi dùng gold.

## File

- `mainflow_answers.csv`, `mainflow_results.xlsx`: từng câu × chế độ, gồm câu hỏi, đáp án mẫu, câu trả lời, ID evidence được trích, ID evidence gold, chỉ số và thời gian.
- `mainflow_summary.csv`: bảng trên.
- `mainflow_failures.csv`: 10 dòng không có câu trả lời, kèm lý do.
- `mainflow_answers_images_{on,off}.jsonl`: log đầy đủ: nguồn gửi model (ID + text), claims, planner, timing, chẩn đoán.

Dữ liệu đầu vào đóng băng ở `../frozen_20261030/`: gold, retrieval production, graph Neo4j, pods. Protocol nằm ở `../PILOT50_PROTOCOL.md`.

## Chưa hoàn thành

| Yêu cầu | Trạng thái |
|---|---|
| E1 (5 định dạng × gold/retrieved × ≥3 lần) | 276/1.500 lượt, chạy bằng prompt của runner đánh giá, không phải luồng chính. Đã dừng, chưa chấm. `../runs/E1/` |
| E2 (4 mode × ≥3 lần) | 111/600 lượt, đã dừng, chưa chấm. `../runs/E2/` |
| E4 (full-page / crop / OCR / OCR+caption) | 70/288 lượt, đã dừng, chưa chấm. Đã đo được: ảnh cả trang ≈ 2.700–2.800 visual token/trang. GPU memory không đo được vì inference chạy từ xa. `../runs/E4/` |
| E3 (text-only) | Dùng chế độ ảnh tắt của luồng chính: generator (Qwen2.5-VL) chỉ nhận text. Chưa chạy trên tài liệu chưa thấy |
| Bộ test 200–300 câu trên tài liệu chưa dùng | Chưa làm |

Các log trong `../runs/` lấy mẫu không đều giữa các ô (lịch chạy ban đầu xáo trộn toàn bộ), nên **không dùng để so sánh các định dạng**.

Các file `failed_concurrent_gpu_*` và `mixed_old_new_*` trong `../mainflow_20261030/` là các lượt hỏng (GPU bị tranh chấp, tiến trình cũ chạy chồng); chúng chỉ được giữ làm dấu vết.

# Kết quả thực nghiệm pilot (cập nhật 2026-10-31)

Toàn bộ số liệu dưới đây là số được báo cáo trong bài (`paper/main.tex`). Không có số nào được sửa tay; mọi chỉ số tính lại được bằng script trong `evaluation/runners/`.

## Luồng chính (full system), 50 câu pilot, text-only

Mỗi câu chạy qua `search_knowledge(generate=True)` của hệ thống thật: planner → BM25 + BGE-M3 + ColPali → RRF → reranker → mở rộng graph → Evidence Pods → sinh câu trả lời có kiểm tra citation. Generator (Qwen2.5-VL-7B-AWQ, zero-shot) chỉ nhận text.

**Đủ 50/50 câu có câu trả lời.** 5 câu lấy từ lượt chạy thứ hai (2026-10-31):
- P047–P050: lượt đầu mất do server sinh câu trả lời tạm ngừng (`gateway_unavailable`).
- P007: lượt đầu bị bước kiểm tra ngôn ngữ đầu ra từ chối (Qwen viết một từ tiếng Trung). Bước kiểm tra này đã được gỡ khỏi hệ thống, sau đó chạy lại.

45 câu còn lại giữ nguyên lượt đầu (đều đã qua bước kiểm tra ngay lần đầu). Log thô của cả hai lượt nằm ở `../mainflow_20261030/answers_images_off.jsonl` (dòng sau cùng của mỗi câu là lượt được báo cáo); bản trước khi chạy lại: `answers_images_off.before_rerun_20261031.jsonl`.

| Chỉ số | Giá trị | Mẫu số |
|---|---|---|
| Hit@1 | 0,800 (40/50) | 50 câu |
| MRR | 0,869 | 50 câu |
| Evidence Recall@10 | 0,905 | 50 câu |
| Evidence Precision@10 | 0,126 (trần 0,146 vì trung bình 1,46 vùng gold/câu) | 50 câu |
| Required-evidence recall | 0,955 | 50 câu |
| Citation precision / recall | 0,839 / 0,810 | 50 câu |
| Answer accuracy (partial = 0,5) | **0,800** = (33 + 0,5×14)/50 | 50 câu |
| Strict accuracy | 0,660 (33/50) | 50 câu |
| Đúng / một phần / sai / không trả lời | 33 / 14 / 2 / 1 | 50 câu |
| Attribute leakage rate | 0,051 (5/98) | 98 claim |
| Citation faithfulness | 0,888 (87/98) | 98 claim |
| Key-value recall | 0,826 (18/24 câu đủ) | 24 câu có số/mã |

Gold: 73 lượt câu–vùng trên 65 vùng khác nhau (7 vùng dùng chung cho nhiều câu); 21 lượt câu–bảng trên 16 bảng. Gold được đối chiếu qua `content_hash`, nên bản upload trùng của cùng một PDF ở phiên bản khác vẫn được tính.

**Chấm nhãn:** nhãn đúng/một phần/sai, leakage và faithfulness được gán trong một lượt chấm có hỗ trợ của LLM (Claude) theo rubric `../RUBRIC_DANH_GIA.md`, sau đó tác giả thứ nhất rà soát. Chưa có người chấm độc lập thứ hai.

**Chế độ bật ảnh** (generator nhận thêm ảnh khi chính sách production yêu cầu) vẫn được giữ trong các file nhưng **không dùng trong bài** và **không chạy lại**: còn 45/50 câu (P007, P047–P050 không có câu trả lời).

## Thí nghiệm nhỏ (worked examples), mỗi cấu hình chạy 1 lần

Chọn các câu đã chạy đủ mọi cấu hình, không chọn theo kết quả.

- **E2 mở rộng graph** (P003, P007, P022, P027), `E2_examples.csv`: scope-aware ít hơn 34–62% token so với structural mà vẫn giữ đủ evidence; nhưng prev/next còn rẻ hơn scope-aware ở cả 4 câu.
- **E4 đầu vào ảnh** (P013, P028, P039, P040), `E4_examples.csv`, `E4_manual_grades.csv`: OCR text 0,75 so với ảnh cả trang 0,63, chỉ khoảng 1/5 số token; OCR thô và OCR + pods cùng 0,75.
- **E1 định dạng bằng chứng**:
  - 4 câu text không có bảng (P002, P018, P023, P029), `E1_examples.csv`: các định dạng gần như không khác nhau.
  - 5 câu bảng có lưới (P006, P008, P013, P015, P048), `E1_table_examples.csv`: mọi định dạng đúng P006, P008, P013; ở P015 pods (E) trả lời cụt, thiếu biện pháp; ở P048 pods (D, E) chỉ liệt kê 2/4 mã, flat 3/4, JSON/Markdown 4/4. **Pods không tốt hơn** text theo hàng trên các bảng đơn giản này. Lưu ý: định dạng flat (A) đã ghi mỗi hàng bảng thành một dòng `ô ; ô`.

## File

- **`BANG_TONG_HOP.xlsx`** (file bàn giao chính): README, summary_mean_sd, comparison_full, error_cases, manual_grades, ranking, ví dụ E1/E2/E4. Tạo lại: `python -m evaluation.runners.pilot_deliverables research_evaluation_20261030`.
- `summary_mean_sd.csv`: mean ± SD theo câu hỏi cho mọi chỉ số và số câu đúng/một phần/sai/không trả lời.
- `comparison_full.csv`: so sánh E1 (5 định dạng × gold/retrieved), E2 (4 mode), E4 (4 kiểu input).
- `error_cases.csv`: các dòng lỗi, phân loại `missing_detail` / `leakage` / `unsupported_content` / `abstained` / `not_generated`.
- `charts/*.png`: chỉ số luồng chính với khoảng tin cậy bootstrap 95%, E2 (token, recall theo mode), E4 (accuracy so với token).
- `mainflow_answers.csv`, `mainflow_results.xlsx`: từng câu: câu hỏi, đáp án mẫu, câu trả lời, ID evidence được trích, ID gold, chỉ số, thời gian.
- `mainflow_answers_images_off.jsonl`: lượt được báo cáo của từng câu (50 dòng): nguồn gửi model (ID + text), claims, planner, timing, chẩn đoán. `mainflow_answers_images_on.jsonl`: chế độ bật ảnh (không dùng trong bài).
- `mainflow_manual_grades.csv`, `mainflow_manual_grades_summary.csv`: nhãn từng câu và bảng tổng hợp.
- `mainflow_ranking_metrics.csv`, `mainflow_retrieval_citation*.csv`, `recall_precision.xlsx`: Hit@1, RR, Recall@3/5/10, Precision@10, required recall, citation precision/recall.
- `mainflow_summary.csv`: số câu có câu trả lời, key-value recall, thời gian (retrieval median 66,9 s, sinh câu trả lời median 43,5 s).
- `mainflow_failures.csv`: các dòng không có câu trả lời (chỉ còn chế độ bật ảnh).
- `before_rerun_20261031/`: các CSV trước khi chạy lại 5 câu.

## Log thiếu hoặc không dùng (ghi rõ)

- Output thô của lượt đầu bị từ chối của P007 **không được lưu** (runner lúc đó chỉ lưu số token). Từ 2026-10-31 runner lưu toàn bộ `qwen_outputs`.
- Prompt của luồng chính do code production dựng lại mỗi lần và không lưu theo từng câu. Log ablation trong `../runs/` có prompt đầy đủ.
- `../mainflow_20261030/judgments.jsonl` và `report/`: bộ chấm Qwen tự động dừng ở 14/100 lượt khi gateway tắt. **Không dùng.**
- `failed_concurrent_gpu_*`, `mixed_old_new_*` trong `../mainflow_20261030/`: lượt hỏng (GPU tranh chấp, tiến trình cũ chạy chồng), chỉ giữ làm dấu vết, không đưa vào bản phát hành.
- `../runs/E1`, `E2`, `E4` là lịch chạy lớn đã dừng giữa chừng (E1 khoảng 300/1.500 lượt, E2 111/600, E4 70/288) và lấy mẫu không đều giữa các ô, nên **chỉ dùng các câu đã chạy đủ mọi cấu hình**, không dùng để so sánh toàn bộ.
- Chưa làm: bộ 200–300 câu trên tài liệu chưa dùng, người chấm độc lập thứ hai, lần chạy không có ColPali, LLM thuần text.

## Git

Bản phát hành: https://github.com/Andydo21/MultiModelAndExtract. Trong bản phát hành, tên máy chủ gateway và e-mail xuất hiện trong văn bản bên thứ ba đã được che (`<gateway-host>`, `<email>`), nên chuỗi hash của log ablation chỉ kiểm tra được trên log gốc.

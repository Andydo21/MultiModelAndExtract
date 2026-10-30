# NewIdea tìm kiếm và hỏi đáp tài liệu

Mở `/newidea/` bằng tài khoản web hợp đồng. Luồng chính đọc nguồn A3 đã lưu,
không tạo dữ liệu nhà máy hay đáp án mẫu khi thiếu nguồn hoặc model.

## Luồng hiện tại

1. View kiểm tra đăng nhập và nhận câu hỏi. `search_job_service` xếp hàng,
   worker kiểm tra lại quyền, cập nhật tiến độ và lưu kết quả.
2. `knowledge_search_service` tìm trên chỉ mục các phiên bản mới nhất được
   phép đọc. BM25, BGE-M3 và ColPali tạo ứng viên; RRF hợp nhất và BGE reranker
   xếp hạng chung. Chỉ tài liệu được chọn mới tải cây/graph.
3. Qwen tách ý hỏi, xác định phạm vi đọc và nhu cầu xem ảnh trong cùng lượt
   lập kế hoạch. Không dùng danh sách từ Việt/Anh để đoán các ý định này.
   Nếu model lập kế hoạch không chạy, tìm bằng câu hỏi gốc và ghi unavailable.
4. `search_evidence_service` chọn nguồn khởi đầu bao phủ các ý hỏi, giảm bản
   trùng và giữ ngữ cảnh của tài liệu đã chọn. Thuật toán truy xuất không tự
   tạo dữ kiện hoặc kết luận kỹ thuật.
5. `knowledge_answer_service` đưa nội dung nguồn trước, graph sau rồi gọi
   Qwen trả lời, giải thích và ghi phần nguồn chưa nói rõ. Không chặn sinh
   đáp án theo tỷ lệ trùng từ. Code kiểm tra mã nguồn và định dạng citation.
6. Nếu vượt context, giảm đầu ra rồi chia nhóm nguồn khi cần; lưu dấu vết
   nguồn đã đọc/chưa đọc. Đáp án, nguồn và ghi chú được lưu riêng.

Hỏi một phiên bản đi qua `contract_search_service`. Chế độ đọc toàn bộ do
người dùng chọn rõ hoặc do kế hoạch Qwen xác định. Tìm case chuyên gia dùng
BGE reranker đa ngôn ngữ trên case đã duyệt, còn đúng nguồn; không có điểm
trùng từ thay thế khi model lỗi. Nhánh này ghi unavailable khi không chạy.

## Tri thức chuyên gia

Case giữ tác giả, revision, reviewer, điều kiện áp dụng và snapshot nguồn.
Reviewer phải có quyền và khác tác giả. Đây là kiểm tra nghiệp vụ, không
giao cho model quyết định quyền đọc hay phê duyệt.

`expert_submission_service` dùng Qwen để đề xuất đơn vị nội dung và quan hệ
ngữ nghĩa. Code kiểm tra đoạn trích nguyên văn, loại quan hệ và đầu nối trước
khi lưu graph. Liên kết nguồn HAS_UNIT thể hiện đơn vị thuộc bản ghi nào;
code không tự gắn mọi đơn vị thành ASKS_ABOUT.

Bản ghi DEMO bị loại khỏi nguồn trả lời thật, kể cả khi câu hỏi nói đến demo
hoặc chuyên gia. Việc kiểm tra nhãn mẫu chỉ phục vụ nhận diện nguồn mẫu.
Model/migration prototype giữ cho lịch sử database; giao diện và API case
library prototype riêng đã gỡ. Server NewIdea riêng chuyển về web chính.
Không di chuyển hay sửa dữ liệu đã lưu trong lần dọn code này.

## Phân chia code

- `views/`: HTTP, đăng nhập, JSON và trả kết quả.
- `services/`: điều phối truy xuất, hỏi đáp, hàng đợi và chuyên gia.
- `repositories/`: ORM, truy vấn chỉ mục, lưu nguồn và graph.
- `case_rules.py`: trường case, điều kiện duyệt, quyền reviewer và xuất case.
- `evidence_service.py`: gắn nguồn và kiểm tra SHA-256 nguồn đã duyệt.

View không gọi repository trực tiếp. Repository không import service: nó chỉ
đọc/ghi nguồn, tệp, snapshot và graph. Service dựng cây, kiểm tra nghiệp vụ và
chuẩn bị bằng chứng rồi gọi repository lưu kết quả trong cùng giao dịch.
BM25 trên postings dùng hàm thuần trong `shared/retrieval.py`; không chứa ORM.

Không có Spec Drift trong luồng hiện tại. Không tự tìm Internet. Cache kết
quả qua nhiều lượt hỏi không bật mặc định. Quyền công ty, nguồn gốc, kiểu dữ
liệu và tính toàn vẹn graph vẫn được kiểm tra bằng code.

## Chạy và kiểm thử

Khởi động bằng `start-services.ps1` từ thư mục gốc. Worker chính chạy bằng
`python manage.py knowledge_worker`.

Ví dụ kiểm thử cô lập từ thư mục gốc:

```powershell
python manage.py test NewIdea.tests contracts.tests.test_intelligence shared.test_qwen_gateway --settings=config.intelligence_test_settings --noinput
```

Test dùng dữ liệu và model stub cô lập để kiểm tra giao thức, lỗi và phân
quyền. Stub chỉ nằm trong test, không được gọi trong luồng thật. Test phần
mềm không thay cho đo độ chính xác trên tài liệu thật hoặc thời gian Qwen.

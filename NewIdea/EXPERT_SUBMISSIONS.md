# Câu hỏi, câu trả lời và tài liệu của chuyên gia

Vào `/newidea/`, mở **Bổ sung và quản lý ý kiến chuyên gia**.
Nhập câu hỏi, câu trả lời nếu có, ghi chú và tối đa ba file đính kèm.
Nút **Hỏi kho kiến thức** tìm nhiều tài liệu qua luồng A2 hiện có; kết quả AI được hiển thị riêng, không tự ghi thành lời của chuyên gia.
Nút **Tách nội dung và lưu liên kết** tạo job bền vững, chạy trên `knowledge_worker`.

File mới được mã hóa và lưu thành phiên bản riêng. Worker dùng A3 để trích xuất và lập chỉ mục. Qwen nhận văn bản gốc của câu hỏi, câu trả lời, ghi chú và các block A3, tách thành unit, nhận diện mã máy, hành động, cách giải quyết. Mỗi unit và đối tượng phải có trích đoạn nguyên văn kiểm tra được. Không dùng dữ liệu giả khi model lỗi.

## Neo4j

- `A2ExpertQuestion`: câu hỏi gốc, công ty và tác giả.
- `A2ExpertAnswer`: câu trả lời nguyên văn, `ANSWERS` tới câu hỏi được nhập cùng yêu cầu. Đây là liên kết do người gửi chỉ định, không chứng nhận câu trả lời đúng.
- `A2SemanticUnit`: ý nhỏ có nguồn; câu hỏi và câu trả lời nối bằng `HAS_UNIT`.
- `A2Concept`: đối tượng chung trong cùng công ty. Unit nối bằng `ABOUT_MACHINE`, `HAS_ACTION`, `HAS_SOLUTION`.
- Câu hỏi và câu trả lời `HAS_DOCUMENT` tới snapshot A3 của tài liệu gửi kèm hoặc được chọn. Việc gửi kèm không tự chứng minh nội dung câu trả lời.
- Unit trích từ tài liệu `HAS_EVIDENCE` tới snapshot và block A3 chứa nguyên văn. Các unit ghi chú/câu trả lời không bị gán giả là xuất phát từ file.

Code tạo các loại cạnh cố định, Qwen chỉ trích nội dung và đối tượng. Neo4j dùng khóa có công ty và constraint duy nhất. Việc xử lý lại cùng job thay thế node của đúng job, giữ các unit của yêu cầu khác.

Đối tượng dùng chung chỉ khi cùng loại và cùng cụm từ nguồn sau chuẩn hóa hoa/thường, khoảng trắng. Không tự gộp mã máy khác dấu gạch, revision khác hoặc giải pháp diễn đạt khác. Các ý liên quan được tìm qua đối tượng chung và hiển thị kèm nguồn/trạng thái nháp; chung đối tượng không đồng nghĩa cùng điều kiện áp dụng.

## Phạm vi và kiểm tra

Mọi nội dung mới giữ trạng thái nháp, chưa tự trở thành case chuyên gia được duyệt trong câu trả lời toàn công ty. Luồng case/phê duyệt hiện có vẫn dùng riêng. Unit mới chưa có collection vector riêng; tìm ý chung ở đây là đi qua node đối tượng trong Neo4j.

Mỗi yêu cầu xử lý tối đa 12 nguồn/đoạn văn bản độc lập. Không trộn câu hỏi, câu trả lời và các block khác vào cùng lần gọi model. Tên đối tượng được code lấy từ trích đoạn đã kiểm tra, không dùng bản dịch do model tự sinh. Schema ràng buộc ID nguồn và loại đối tượng; nếu kiểm tra nguồn thất bại, thử lại một lần với phản hồi rồi báo partial/unavailable và giữ dữ liệu gốc. Graph báo partial nếu không nối được đủ snapshot hoặc bằng chứng A3; không báo đã có bằng chứng khi chỉ có file đính kèm.

Ngày 2026-10-12: kiểm thử tự động A2 và giao thức Neo4j thật đã kiểm tra liên kết nguồn, ghi lại không trùng, đối tượng chung và cách ly công ty. Qwen thật đã truy cập được. Dữ liệu mẫu tiếng Việt qua upload TXT → A3 → BGE → Qwen → Neo4j; hai yêu cầu tìm thấy nhau qua mã máy và hành động chung. ColPali bỏ qua trên TXT không có ảnh trang, nên lần thử này không đánh giá OCR hoặc tìm kiếm ảnh. Kết quả giữ nháp, không phải benchmark độ chính xác hay chứng nhận mọi phân loại ngữ nghĩa đúng.

90 kiểm thử tự động đạt. Job mẫu kèm tài liệu tạo 7 unit, 6 cạnh bằng chứng tới snapshot/leaf và không có unit tài liệu thiếu nguồn. Job trả lời thứ hai tạo 2 unit và tìm được các ý có đối tượng chung. Mẫu có nhãn dưới đây đã được Qwen thật trích đủ machine/action/solution, giữ điều kiện. Các lần thử trước có lỗi phân loại và bỏ sót: dạng nhập tự do chưa ổn định như mẫu có nhãn; vẫn cần duyệt.

## Mẫu nhập câu trả lời chuyên gia

Ví dụ kiểm thử, không phải hướng dẫn vận hành thiết bị thực tế:

```text
Mã máy: AX-203.
Vấn đề: báo lỗi E17.
Hành động: ngắt nguồn trước khi kiểm tra dây tín hiệu.
Cách giải quyết: thay dây tín hiệu bị hỏng.
Điều kiện áp dụng: chỉ thực hiện khi đã xác nhận dây hỏng; không áp dụng cho AX-204.
```

System prompt yêu cầu đọc duy nhất nguồn đang xử lý, tách ý theo ngữ nghĩa, giữ phủ định/điều kiện/số liệu và trích nguyên văn. Qwen chỉ chọn loại unit/đối tượng từ schema cố định. Code lấy tên đối tượng từ quote, kiểm tra nguồn rồi tạo quan hệ. Không yêu cầu Qwen tự tạo loại quan hệ hay tự hoàn thiện thông tin còn thiếu.

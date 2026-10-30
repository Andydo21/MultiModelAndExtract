# Trích dẫn nguyên văn trong NewIdea

- Qwen được yêu cầu chọn nguyên câu hỗ trợ mỗi ý. Với bảng không có câu hoàn chỉnh, chọn hàng hoặc ô liên quan nguyên văn và giữ tiêu đề khi có thể; không tự tạo câu từ các ô rời.
- Hệ thống chỉ giữ trích dẫn khớp với văn bản của đúng nguồn, bỏ trích dẫn không khớp và gộp các đoạn trùng theo nguồn. Kiểm tra này xác nhận vị trí văn bản, không chứng minh kết luận đúng về mặt ngữ nghĩa.
- Kết quả có `answer_quotes`, hiển thị ở phần nguồn trong giao diện NewIdea và lưu cùng lịch sử tìm kiếm. Nếu Qwen không trả được đoạn nguyên văn khớp nguồn, không tự dựng đoạn trích; văn bản nguồn vẫn có thể được xem để đối chiếu.
- Với hàng bảng được Qwen trích dưới dạng Markdown, đối chiếu nội dung các ô theo đúng thứ tự và trả về văn bản gốc khớp nguồn, không hiển thị dấu phân cách được model thêm vào.
- Đã chạy 33 kiểm tra `KnowledgeAnswerChecks` trong Docker. Chạy thật câu 35 qua NewIdea với Qwen: thu được nguyên câu chú thích về tải thử và hàng gốc về tải tối đa; kết quả khớp trang 2 PDF LPH. Đây là một lượt thử, không bảo đảm mọi câu hỏi đều trả được trích dẫn đầy đủ.

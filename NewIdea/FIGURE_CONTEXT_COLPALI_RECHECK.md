# Bổ sung ngữ cảnh figure khi tìm bằng ColPali

## Thay đổi

- Khi rerank figure, dùng thêm chữ OCR con có quan hệ parent do Surya cung cấp; text này không bị gán thành nguyên văn của figure.
- Khi figure được chọn, thêm các block chữ con cùng trang làm nguồn riêng; caption chưa có parent được đưa vào dưới dạng ứng viên bố cục lân cận, kèm ID figure, trang và bbox. Không ghi quan hệ ngữ nghĩa đã xác nhận vào graph.
- Khi có caption ứng viên và cần ảnh, ưu tiên ảnh crop figure và ảnh nguyên trang để model có thể xem ngữ cảnh; giới hạn số ảnh hiện hành vẫn áp dụng.
- Giữ ID nguồn Qwen viết trong câu trả lời, tránh câu bị khuyết sau khi bỏ ID. Nguồn text và nguồn ảnh vẫn có ID riêng.
- Không chạy lại OCR hay ColPali: thay đổi sử dụng các block và vector đã lưu.

## Kiểm tra

- 84 kiểm tra NewIdea/ColPali trong Docker đạt; các cảnh báo về Surya và ảnh không có trong fixture xuất phát từ test cô lập, không phải lượt chạy model thật.
- Với figure block 1001499 trên trang 1 DigitalTwinSoftware_en, truy vấn Qdrant sử dụng chính vector ảnh đã lưu trả figure này; context gồm block con `Rich asset library` và caption ứng viên `A complete collection of resources to improve digital twin environments.` Figure cạnh bên block 1001502 có block con `Physics` và caption ứng viên riêng. Đã đối chiếu bố cục trang PDF.
- Truy vấn Qdrant trên dùng vector của ảnh làm đầu vào để kiểm tra kết nối dữ liệu, không phải kiểm thử tìm kiếm bằng câu hỏi qua ColPali.
- Đã thử chạy lại câu hỏi 2 qua luồng NewIdea thật; gateway Qwen và ColPali trả HTTP 404 nên chưa thể kiểm tra câu trả lời, ảnh thực sự được gửi và so sánh chất lượng trước/sau với model thật.
- Benchmark cũ được tạm dừng khi gateway mất: 22 câu có câu trả lời, các kết quả dự phòng được giữ nguyên. Lượt mới dùng file `evaluation/colpali50_results_after_visual_context.jsonl` riêng để tránh trộn hai phiên bản code; chạy lại sau khi gateway hoạt động.

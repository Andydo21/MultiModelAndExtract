# A2 trong Contract Guard AI

Điểm tìm kiếm NewIdea là `/newidea/` trên web quản lý hợp đồng, dùng cùng phiên đăng nhập hiện tại. Dashboard, chi tiết hợp đồng và pipeline A3 có liên kết “NewIdea — Tìm tài liệu A3”. `/contract-intelligence/` vẫn dùng cùng pipeline tìm kiếm.

## Từ tài liệu A3 đến tìm kiếm NewIdea

1. Tải file vào hợp đồng và chạy trích xuất A3 cho phiên bản đó.
2. Từ pipeline, chọn “Tìm tài liệu này bằng NewIdea”; liên kết giữ version ID.
3. Nhập câu hỏi. NewIdea đọc nguồn của extraction run hoàn tất trong web, dùng BM25 + BGE-M3 + ColPali, fusion/reranker và mở rộng cấu trúc từ service A3 chung.
4. Kết quả giữ citation, trang, bbox và liên kết đối chiếu nguồn; đồng thời trả case A2 đã duyệt phù hợp. Có thể bật Qwen để tổng hợp từ nguồn.

Không sao chép file hoặc index sang SQLite/collection nhà máy. `NewIdea/views/contract_views.py` gọi `services/contract_search_service.py`; service dùng repository để đọc nguồn và `ai_extract/services/contract_pipeline_service.py` để chạy A3. Endpoint tra cứu của web chính dùng cùng service. API mới là `POST /api/newidea/contracts/<contract_id>/search/`; client gửi JSON gồm `query`, `version_id`, `generate` và `mode` như A3. Client gửi contract ID trong body dùng `POST /api/newidea/ask/`. Request cần phiên đăng nhập và CSRF token; công ty, hợp đồng và phiên bản được kiểm tra trước retrieval. Source mới chỉ xuất hiện khi trích xuất đã hoàn tất. Vector model lỗi được báo rõ; lexical vẫn tìm trên nguồn đã lưu.

## Khi chạy NewIdea độc lập

Mặc định `NEWIDEA_MODE=contracts`: trang chat chuyển sang `CONTRACT_WEB_URL/newidea/`, mặc định `http://localhost:8000/newidea/`. Tìm kiếm được thực hiện trên web chính để giữ session và quyền theo công ty. API hỏi prototype không quét index hợp đồng trong chế độ này.

Prototype case library chỉ chạy khi chủ động đặt `NEWIDEA_MODE=prototype` trong môi trường riêng. Không dùng index chứa hợp đồng khách hàng cho prototype. API hỏi/đóng góp/expert/xóa lịch sử cũ đã ngừng cung cấp; cấu hình mode không mở lại các đường ghi này.

`NewIdea/services/case_rules.py` cung cấp quy tắc nghiệp vụ A2 cho web chính. Dữ liệu được lưu trong `contracts.ContractKnowledgeCase`, không đưa hợp đồng khách hàng vào database mẫu nhà máy trong NewIdea.

Một case gồm vấn đề điều khoản, căn cứ của chuyên gia, cách xử lý, kết quả xác minh, điều kiện áp dụng và nguồn. Case gắn với ContractVersion, tác giả và công ty. MANAGER hoặc ADMIN cùng công ty, khác tác giả revision, phê duyệt. Nguồn là ID A3 thật trong phiên bản; phê duyệt lưu nội dung và SHA256. Case thiếu snapshot hoặc nguồn đã đổi không được gợi ý tái sử dụng. Revision mới giữ bản đã duyệt cho đến khi được duyệt thay thế; AuditLog lưu thao tác.

## Tìm toàn bộ kho và chuyển chuyên gia

`POST /api/newidea/search/` tìm trên phiên bản mới nhất của tất cả tài liệu tài khoản được phép xem, không thêm bộ lọc theo công ty đang gắn với admin. Admin tìm qua mọi công ty; tài khoản thường giữ quyền đọc hiện có. Dò BM25 và BGE trên kho, sau đó dùng A3 để lấy nguồn, mở rộng cấu trúc và xếp hạng lại. Tổng hợp chỉ dùng phiên bản có run A3 hoàn tất và chỉ mục sẵn sàng. Giới hạn baseline và phạm vi kết quả được mô tả trong [README](README.md); API trả cờ `partial` khi vượt giới hạn. Khi bằng chứng không đủ liên quan, A2 từ chối tổng hợp và giữ ứng viên trong phần nguồn để kiểm tra.

Trong màn hình tra cứu, “Gửi câu hỏi cho chuyên gia” tạo yêu cầu từ lịch sử truy vấn thật. Quản lý phân công người xử lý; người đó soạn case nháp trên đúng phiên bản. Phê duyệt bởi người khác tác giả mới chuyển yêu cầu thành `RESOLVED`.

UI gửi yêu cầu tổng hợp Qwen vào `/api/newidea/jobs/`. Worker database nhận job qua `knowledge_worker`, kiểm tra lại quyền trước retrieval và lưu đầy đủ kết quả trước công bố `COMPLETED`. Job lỗi không tự retry. Truy vấn nguồn thuần vẫn chạy trực tiếp. Migration `contracts.0013_knowledge_search_workflow` bổ sung snapshot nguồn, lịch sử tra cứu và yêu cầu chuyên gia.

Các màn hình và seed nhà máy cũ là prototype tham khảo. Không chạy seed đó để tạo dữ liệu hợp đồng. Thư viện hợp đồng chính dùng trường hợp thực tế do người dùng nhập, không dùng dữ liệu DENSO.

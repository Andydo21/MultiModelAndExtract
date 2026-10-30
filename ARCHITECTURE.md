# Cấu trúc code theo nghiệp vụ

Endpoint đi theo `urls → views → services → repositories → models/database`.

- View đọc request, xử lý đăng nhập/session/CSRF, gọi service và trả HTTP hoặc render template.
- Service nhận user, dữ liệu và ID; kiểm tra quyền, điều phối nghiệp vụ, giao dịch, AI và gateway. Service trả dữ liệu hoặc `ServiceResult`, không nhận HttpRequest hay tạo HttpResponse.
- Repository chứa truy vấn ORM, khóa bản ghi, SQL và ghi dữ liệu. Giao dịch bao quanh một nghiệp vụ nằm trong service; các thao tác repository giữ cùng transaction hiện có.
- Model, migration, admin, URL, template, serializer, tiện ích mã hóa và kiểm thử giữ vai trò Django tương ứng.

## Contracts

`views/` chia theo hợp đồng, phân tích, rủi ro, workflow, tài khoản, quản trị, tải tài liệu, tri thức và ledger.

`services/` chứa các nhóm nghiệp vụ tương ứng. `contract_service.py` cung cấp `ContractService`; `contract_operations_service.py` điều phối thao tác API trên hợp đồng. `ledger_service.py` xử lý chữ ký và outbox; `ledger_operations_service.py` xử lý đăng ký khóa, thao tác ledger và kiểm chứng.

`repositories/` chứa các repository model đã có cùng nhóm truy vấn nghiệp vụ. SQL workflow của chữ ký nằm trong `ledger_workflow_repository.py`.

## AI Extract

`views/extraction_views.py` nhận API tóm tắt, thực thể và điều khoản; `views/layout_views.py` nhận màn hình và API pipeline.

`services/` điều phối các thao tác qua `extraction_service.py`, `layout_service.py` và các service tóm tắt/thực thể/điều khoản. `repositories/` chọn đúng hợp đồng, phiên bản, trang, block và ghi kết quả.

`services/contract_pipeline_service.py` giữ thuật toán và điều phối A3 chung. Surya, parser, chunking, embedding, reranker và Qwen cùng nằm trong `services/`. Các route gateway inference nằm trong `views/inference_views.py`; notebook Kaggle giữ bản nhúng. Việc ghi extraction run, page, block và context nằm trong `repositories/layout_repository.py`.

Chỉ còn một ứng dụng xử lý AI: `ai_extract`. `a3/documents` và các API kho tài liệu độc lập đã được bỏ. `legacy_schema/` giữ nguyên model/migrations cũ với app label `documents` và bảng `documents_file` để đọc hoặc quản trị database lịch sử; schema này không nằm trong INSTALLED_APPS của web chính. Không áp dụng migration hay chuyển dữ liệu thật trong lần gom code này. Code đã ngừng dùng có bản ZIP kiểm tra SHA256 trong `a3/`.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe manage.py check
.\.venv\Scripts\python.exe manage.py test contracts.tests.test_layered_architecture contracts.tests.test_intelligence contracts.tests.test_ledger NewIdea.tests.test_enterprise NewIdea.tests.test_contract_search ai_extract.tests.test_a3_bridge ai_extract.tests.test_hybrid_retrieval ai_extract.tests.test_contract_structure ai_extract.tests.test_surya_ocr shared.test_qwen_gateway --settings=config.intelligence_test_settings --noinput
```

`test_layered_architecture.py` kiểm tra không truy vấn model trực tiếp ở view/service, không dùng HttpRequest/HttpResponse trong service, cùng hành vi endpoint và phân quyền công ty. Các kiểm thử dùng SQLite cô lập, không chạy migration trên Neon hoặc gửi hợp đồng thật đến AI.

Kiểm thử nằm trong `tests/` của từng app. Các nhóm route hợp đồng nằm trong `views/knowledge_urls.py` và `views/ledger_urls.py`. Mã hóa tài liệu và tạo PDF nằm trong `services/`. API blockchain cũ và trang explorer cũ đã được bỏ; dùng `/contract-ledger/` và `/api/ledger/`.

Các route debug `/developer-test/`, `/api/developer-test/run/` và API hỏi của prototype NewIdea đã được bỏ. API phiên bản và workflow kiểm tra đăng nhập và quyền công ty trước khi đọc dữ liệu hoặc gọi dịch vụ ngoài. Tài khoản chưa gắn công ty không được liệt kê toàn bộ hợp đồng.

Luồng web được giữ: tạo hợp đồng/phiên bản và lưu file → A3 trích xuất trang, block và cấu trúc điều/khoản → lập chỉ mục theo đúng công ty/phiên bản → NewIdea tìm và trả bằng chứng → chuyên gia tạo case, reviewer duyệt → workflow và ledger ký/kiểm chứng. Kiểm thử dùng gateway giả lập; kết quả không xác nhận Docker, Kaggle hay Fabric đang chạy thật.

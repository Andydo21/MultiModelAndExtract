# Runtime trích xuất A3 trong Docker

Trước đây web trong Docker gọi runtime Python trên Windows qua cổng 8010. Khi máy khởi động lại, runtime đó không tự chạy, khiến Surya báo lỗi kết nối.

Overlay `docker-compose.web-local.yml` cấu hình container `a3-inference` chạy model trên GPU, dùng lại checkpoint trong `scratch/a3-models` và cache Hugging Face trên ổ D. Container có chính sách tự khởi động lại; web đợi kiểm tra sức khỏe khi Compose khởi động. Qwen và ColPali ở gateway bên ngoài vẫn có vòng đời riêng.

`setup_a3_linux_runtime.sh` chuẩn bị Python Linux và llama.cpp CUDA trên ổ D; `start_a3_linux_runtime.sh` khởi chạy runtime. Cấu hình này chỉ được coi là hoạt động sau khi cài đặt hoàn tất và kiểm thử trích xuất thực tế.

Batch 180 PDF dừng khi tất cả trang của một tài liệu lỗi kết nối Surya, để tránh tiếp tục tạo hàng loạt kết quả thất bại.

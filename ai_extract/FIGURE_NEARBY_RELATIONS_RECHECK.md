# Quan hệ ảnh và ngữ cảnh trong A3

## Cách tạo và dùng quan hệ

- Luồng dựng graph tạo `HAS_FIGURE_TEXT` khi metadata Surya gắn block text với figure cha trên cùng trang. Cạnh giữ trạng thái chưa kiểm chứng ngữ nghĩa; không tự tạo liên kết giữa tài liệu.
- Các block text hoặc caption lân cận được lưu dưới dạng `FIGURE_NEAR_TEXT_CANDIDATE` hoặc `FIGURE_CAPTION_CANDIDATE`. Đây là ứng viên theo quan hệ vị trí, không kết luận text mô tả ảnh. Chỉ xét các block cùng trang; không tạo quan hệ này cho header/footer.
- Khi tìm trúng figure, graph có thể đi từ figure qua text con hoặc text lân cận, rồi qua `RELATED_PASSAGE` do BGE đã tạo ở bước trích xuất để lấy đoạn liên quan. Cạnh BGE vẫn chỉ là similarity candidate, không phải quan hệ tham chiếu đã xác nhận. Qwen nhận các nguồn, cạnh và trạng thái này để đánh giá khi hỏi đáp.
- Qwen được nhắc kiểm tra ảnh và nguồn; không gán caption hay đoạn khác là bằng chứng chắc chắn chỉ vì ở gần hoặc có similarity. Không dùng ColPali làm model xác minh quan hệ: ColPali phục vụ tìm ảnh theo truy vấn.
- Tất cả các cạnh được giới hạn trong snapshot của một tài liệu. Code chạy trong các service hiện có, không thêm service, không yêu cầu chạy Qwen ở bước trích xuất.

## Kiểm tra

- Đã cập nhật graph/snapshot của 20 run hoàn tất, không chạy lại OCR hoặc ColPali. Neo4j có 79 cạnh text con của 37 figure, 336 cạnh text lân cận ứng viên và 4 cạnh caption ứng viên; không thấy cạnh trùng cùng loại/đầu-cuối hoặc cạnh xuyên snapshot.
- Với figure DUALARM `block:1003637:1:0`, Neo4j trả text lân cận `block:1003636:1:0` và đoạn ứng viên `block:1003642:1:0` thông qua `RELATED_PASSAGE`. Đã kiểm tra source pods chứa cả cạnh ảnh–text lân cận và cạnh `RELATED_PASSAGE`, cùng nhãn candidate; chưa gọi Qwen thật cho lượt này.
- Với figure `block:1003665:1:0`, Neo4j trả block chữ con `RC8A` và đoạn khác chứa `RC8A` thông qua similarity. BGE kiểm tra relevance khi cần; similarity không chứng minh đoạn là tham chiếu xác nhận.
- 125 kiểm tra trong Docker đạt, bỏ qua một; các cảnh báo không có model Surya và ảnh trong test fixture là của test cô lập.

## Giới hạn

- Text/caption có thể không được chọn làm láng giềng vì nhận diện bố cục hoặc điều kiện vị trí; link `RELATED_PASSAGE` do BGE cũng có giới hạn và có thể bỏ sót hoặc nhiễu. Vì vậy không đảm bảo tìm được mọi chỗ nói về ảnh hoặc mọi tham chiếu kiểu “Hình 18”.
- Chưa xác nhận được một ví dụ “Hình 18” cụ thể trong 20 run để chứng minh trường hợp này. Cần một tài liệu có ví dụ đó và gateway Qwen hoạt động để kiểm thử đầy đủ.
- Gateway Qwen cũ trả HTTP 404, nên chưa xác nhận model thực sự nhận ảnh, đoạn và đánh giá đúng quan hệ trong lượt hỏi đáp mới. Không coi kiểm tra graph/source pods là kiểm thử end-to-end với Qwen.

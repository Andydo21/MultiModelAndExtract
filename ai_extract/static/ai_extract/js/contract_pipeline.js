const form = document.getElementById('extract-form');
const message = document.getElementById('pipeline-message');
if (form) form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const button = form.querySelector('button');
    const label = button.querySelector('.button-label');
    button.disabled = true;
    label.textContent = 'Đang trích xuất và phân tích cấu trúc…';
    message.hidden = false;
    message.classList.remove('error');
    message.textContent = 'Đang trích xuất nguồn, phân tích cấu trúc và lập chỉ mục BGE/Neo4j. Vector ảnh ColPali được bổ sung sau.';
    try {
        const csrf = form.querySelector('[name=csrfmiddlewaretoken]').value;
        const response = await fetch(form.dataset.endpoint, {method: 'POST', headers: {'X-CSRFToken': csrf}});
        if (!response.headers.get('content-type')?.includes('application/json')) {
            throw new Error('Phiên đăng nhập hết hạn hoặc máy chủ không trả dữ liệu. Hãy đăng nhập lại và thử lại.');
        }
        const data = await response.json();
        if (response.ok && data.partial) {
            window.location.hash = 'extraction-results';
            window.location.reload();
            return;
        }
        if (!response.ok || !data.success) throw new Error(data.error || 'Extraction failed');
        const indexFailed = [data.dense_index_status, data.colpali_index_status].includes('FAILED');
        message.classList.toggle('error', indexFailed);
        message.textContent = `Đã trích xuất ${data.total_pages} trang, ${data.total_blocks} khối nội dung. ${indexFailed ? 'Chỉ mục tìm kiếm có lỗi; xem chi tiết bên dưới.' : 'Đã cập nhật trạng thái chỉ mục tìm kiếm.'}`;
        window.location.hash = 'extraction-results';
        window.location.reload();
    } catch (error) {
        message.classList.add('error');
        message.textContent = error.message;
        button.disabled = false;
        label.textContent = 'Thử lại extraction';
    }
});

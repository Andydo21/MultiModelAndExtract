/* Persisted structure and real retrieval lineage; source text is never HTML. */
(() => {
  const element = (parent, tag, text, className) => {
    const item = document.createElement(tag);
    item.textContent = text || '';
    if (className) item.className = className;
    parent.append(item);
    return item;
  };
  window.renderDocumentTree = (parent, data) => {
    parent.replaceChildren();
    if (!data?.nodes?.length) { element(parent, 'p', 'Chưa có cây đã lưu. Hãy chạy phân tích cấu trúc.'); return; }
    element(parent, 'p', `Trạng thái: ${data.status}. Chọn đề mục để xem nhóm nội dung và mở nguồn.`, 'structure-note');
    if (data.visual_analysis) element(parent,'p',`Phân tích ảnh: ${data.visual_analysis.status} · ${data.visual_analysis.object_count} đối tượng đề xuất · ${data.visual_analysis.relation_count} quan hệ vị trí`, 'structure-note');
    if (data.reading_streams?.length) {
      const panel = element(parent, 'details');
      element(panel, 'summary', `Các luồng đọc (${data.reading_streams.length}) — tách theo trang và vùng nguồn`);
      const sources = new Map((data.leaves || []).map(leaf => [leaf.id, leaf]));
      const streams = new Map();
      data.reading_streams.forEach(stream => {
        const entry = document.createElement('details');
        element(entry, 'summary', `Trang ${stream.page_number} · ${stream.kind} · ${stream.status}`);
        if (stream.status === 'unresolved') element(entry, 'p', 'Thứ tự hoặc vùng cha chưa xác định; không dùng làm chuỗi đọc chắc chắn.');
        stream.items.forEach(item => {
          const row = element(entry, 'div', '', 'tree-source');
          element(row, 'span', `${item.source_id} · ${item.kind} `);
          item.leaf_ids.forEach(id => {
            const source = sources.get(id);
            if (source?.source_url?.startsWith('/')) element(row, 'a', 'Mở nguồn').href = source.source_url;
          });
        });
        streams.set(stream.id, entry);
      });
      data.reading_streams.forEach(stream => {
        const entry = streams.get(stream.id);
        const owner = stream.status !== 'unresolved' && streams.get(stream.parent_stream_id);
        (owner || panel).append(entry);
      });
    }
    if (data.graph_index || data.knowledge) {
      element(parent, 'p', `Neo4j: ${data.graph_index?.status || 'not_indexed'} · Phân tích quan hệ: ${data.knowledge?.status || 'not_analyzed'}`, 'structure-note');
      const sourceRelations = data.knowledge?.source_relations || [];
      if (sourceRelations.length) {
        const panel = element(parent,'details');
        element(panel,'summary',`Quan hệ giữa các đoạn (${sourceRelations.length}) — model đề xuất, chưa rà soát`);
        sourceRelations.forEach(relation => {
          const row = element(panel,'div','','tree-source');
          element(row,'strong',`${relation.from_id} → ${relation.kind} → ${relation.to_id}`);
          element(row,'p',relation.from_quote);
          element(row,'p',relation.to_quote);
          [relation.from_id, relation.to_id].forEach(id => {
            const source = (data.leaves || []).find(leaf => leaf.id === id);
            if (source?.source_url?.startsWith('/')) element(row, 'a', `Mở nguồn · trang ${source.page}`).href = source.source_url;
          });
        });
      }
      const relations = data.knowledge?.relations || [];
      if (relations.length) {
        const panel = element(parent, 'details');
        element(panel, 'summary', `Quan hệ model đề xuất (${data.knowledge.relation_count}) — cần rà soát ý nghĩa`);
        const sources = new Map((data.leaves || []).map(leaf => [leaf.id, leaf]));
        relations.forEach(relation => {
          const row = element(panel, 'div', '', 'tree-source');
          element(row, 'strong', `${relation.from_name} → ${relation.predicate} → ${relation.to_name}`);
          element(row, 'p', relation.quote);
          const source = sources.get(relation.source_id);
          if (source?.source_url?.startsWith('/')) {
            const link = element(row, 'a', `Mở nguồn · trang ${source.page}`);
            link.href = source.source_url;
          }
        });
      }
    }
    const controls = element(parent, 'div', '', 'structure-controls');
    const input = element(controls, 'input'); input.type = 'search'; input.placeholder = 'Tìm đề mục hoặc nội dung'; input.setAttribute('aria-label', input.placeholder);
    const expand = element(controls, 'button', 'Mở tất cả'); expand.type = 'button';
    const collapse = element(controls, 'button', 'Thu gọn'); collapse.type = 'button';
    const container = element(parent, 'div', '', 'document-tree');
    const entries = new Map();
    data.nodes.forEach(node => {
      const details = document.createElement('details'); details.className = 'tree-node';
      details.dataset.node = node.id;
      const summary = element(details, 'summary', `${node.title || node.kind || 'Nội dung'} · ${node.id}`);
      summary.title = node.origin || 'Nguồn chưa xác định';
      if (node.hierarchy_status === 'unresolved' || node.hierarchy_level_status === 'unspecified_by_model') {
        details.classList.add('uncertain');
        element(details, 'small', 'Quan hệ cha hoặc cấp đề mục chưa được xác định.');
      }
      if (node.origin === 'unclassified_source') { details.classList.add('uncertain'); element(details, 'p', 'Chưa đối chiếu/phân loại được', 'structure-note'); }
      element(details, 'small', node.origin || 'Node tài liệu');
      entries.set(node.id, {details, node, searchable: `${node.title || ''} ${node.kind || ''}`});
    });
    data.nodes.forEach(node => {
      const entry = entries.get(node.id);
      const owner = entries.get(node.parent_id);
      try { (owner && owner !== entry ? owner.details : container).append(entry.details); }
      catch (_) { container.append(entry.details); }
      entry.details.open = node.parent_id === null;
    });
    (data.leaves || []).forEach(leaf => {
      const entry = entries.get(leaf.node_id); if (!entry) return;
      entry.searchable += ' '+leaf.text;
      const row = element(entry.details, 'div', '', 'tree-source');
      element(row, 'p', leaf.text || 'Nguồn thị giác không có chữ');
      if (leaf.text_is_preview) element(row, 'small', 'Đang hiển thị đoạn xem trước');
      if (leaf.source_url?.startsWith('/')) {
        const link = element(row, 'a', `${leaf.source_id} · Trang ${leaf.page || '—'} · Mở nguồn`);
        link.href = leaf.source_url;
      }
    });
    expand.onclick = () => entries.forEach(e => { e.details.open = true; });
    collapse.onclick = () => entries.forEach(e => { e.details.open = false; });
    input.oninput = () => {
      const query = input.value.trim().toLocaleLowerCase('vi');
      const visible = new Set();
      entries.forEach((entry, id) => {
        if (!query || entry.searchable.toLocaleLowerCase('vi').includes(query)) {
          let current = id; const visited = new Set();
          while (entries.has(current) && !visited.has(current)) { visible.add(current); visited.add(current); current = entries.get(current).node.parent_id; }
        }
      });
      entries.forEach((entry, id) => { entry.details.hidden = !visible.has(id); if (query && visible.has(id)) entry.details.open = true; });
    };
  };
  window.renderRetrievalTrace = (parent, trace) => {
    parent.replaceChildren();
    if (!trace) { element(parent, 'p', 'Chưa có dữ liệu truy xuất.'); return; }
    element(parent, 'h3', '1. Các nhánh tìm kiếm');
    Object.entries(trace.streams || {}).forEach(([name, detail]) => element(parent, 'p', `${name}: ${detail.state} · ${detail.candidates ?? 0} ứng viên`));
    element(parent, 'h3', '2. Nguồn được chọn và mở rộng');
    const reasons = {query_match:'Kết quả tìm kiếm',parent_context:'Mở rộng nội dung cha',figure_or_table_context:'Hình/bảng liên quan',document_cross_reference:'Dẫn chiếu',cross_reference:'Dẫn chiếu điều khoản',definitions:'Định nghĩa',original_block_context_without_inferred_hierarchy:'Ngữ cảnh block gốc, chưa suy luận cây',review_batch:'Nguồn trong lô rà soát'};
    (trace.selected || []).forEach(source => element(parent, 'p', `[${source.evidence_id}] ${reasons[source.reason] || source.reason} → ${(source.heading_path || []).join(' → ')} · ${source.source_id || source.chunk_id}`));
    element(parent, 'h3', '3. Đầu vào tổng hợp Qwen');
    if (!trace.generation_calls?.length) element(parent, 'p', 'Chưa có lượt Qwen trả về kèm dữ liệu đầu vào. Có thể chưa bật tổng hợp hoặc lượt gọi đã lỗi.');
    (trace.generation_calls || []).forEach((call, index) => {
      element(parent, 'p', `Lượt ${index+1} · ${call.stage} · ${(call.evidence_ids || []).join(', ')} · ${call.delivery === 'confirmed_gateway_response' ? 'Gateway đã trả lời' : 'Thiếu metadata xác nhận đầu vào'}`);
      if (call.derived_ids?.length) element(parent, 'p', 'Tổng hợp từ nhận định đã trích nguồn: '+call.derived_ids.join(', '));
      (call.images || []).forEach(image => element(parent, 'small', `Ảnh nguồn [${image.evidence_id}] · ${image.image_path}`));
    });
    element(parent, 'p', `Kết quả cuối: ${trace.generation_status}. Nguồn được gửi không đồng nghĩa câu trả lời đã vượt kiểm tra.`, 'structure-note');
  };
  document.querySelectorAll('[data-structure-json]').forEach(parent => {
    const data = document.getElementById(parent.dataset.structureJson);
    if (data) window.renderDocumentTree(parent, JSON.parse(data.textContent));
  });
})();

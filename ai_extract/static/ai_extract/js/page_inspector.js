(() => {
  const script = document.getElementById('page-visualization'); if (!script) return;
  const data = JSON.parse(script.textContent);
  const layoutlmBase = data.structure_provider === 'layoutlmv3';
  const sourceRules = data.structure_provider === 'source';
  const surface = document.querySelector('.page-surface');
  const notes = document.getElementById('granite-observations');
  const add = (parent, tag, text) => { const item = document.createElement(tag); item.textContent = text; parent.append(item); return item; };
  const matches = new Map();
  const blocksById = new Map(data.blocks.map(block => [block.id, block]));
  const observationsById = new Map(data.observations.flatMap(page => page.items).map(item => [item.id, item]));
  function preview(ids, modelId, description, regionText) {
    const status = document.getElementById('selected-source');
    const content = document.getElementById('selected-content');
    if (status) status.textContent = description || (modelId ? 'Granite · ' + modelId : 'Block nguồn: ' + ids.join(', '));
    if (!content) return;
    const messages = ids.map(id => {
      const block = blocksById.get(id);
      if (!block) return '';
      const alignment = sourceRules ? 'Nhãn và chữ từ Surya; cấu trúc từ quy tắc nguồn. GLiNER tách thực thể trong chữ, không gán bố cục.' : layoutlmBase ? 'Nhãn từ Surya; cấu trúc từ quy tắc nguồn. LayoutLMv3 base chưa dự đoán nhãn.' : block.classified ? 'Đã đối chiếu cấu trúc.' : 'Chưa đối chiếu đầy đủ với Granite; vẫn là block nguồn đã lưu.';
      const ocr = {
        partial: 'OCR trong ảnh chỉ đọc được một phần; chữ không rõ đã được bỏ qua.',
        unreadable: 'OCR không đọc rõ chữ trong vùng ảnh này.',
        failed: 'OCR bên trong ảnh gặp lỗi; chưa có kết quả dùng được.',
        no_text: 'OCR không tìm thấy chữ đọc được bên trong ảnh.',
        unresolved: 'Surya chưa phân tách được thêm bố cục bên trong ảnh.',
        depth_limit: 'Đã đến giới hạn phân tích ảnh lồng nhau.',
        not_expanded: 'Ảnh con được lưu; không phân tích sâu thêm bên trong.',
        completed: 'OCR bên trong ảnh đã hoàn tất.'
      }[block.visual_ocr_status] || '';
      const containment = block.parent_block_id ? '\nVùng con của ảnh: ' + block.parent_block_id : '';
      const children = block.child_block_ids?.length ? '\nCó ' + block.child_block_ids.length + ' vùng con riêng: ' + block.child_block_ids.join(', ') : '';
      const empty = children ? 'Khung ảnh cha; chữ được lưu trong từng chunk con.' : 'Chưa có nội dung chữ được lưu cho vùng này.';
      const failed = block.ocr_status === 'failed' ? '\nOCR vùng văn bản này gặp lỗi.' : '';
      return `${id} · ${block.kind}\n${alignment}${ocr ? '\n' + ocr : ''}${containment}${children}${failed}\n${block.text || empty}`;
    }).filter(Boolean);
    if (!messages.length) {
      const observation = observationsById.get(modelId);
      messages.push(observation
        ? `Granite · ${observation.kind} · Chưa khớp block nguồn, chưa dùng làm bằng chứng.\n${observation.text || 'Quan sát này không có nội dung chữ.'}`
        : 'Đối tượng ảnh do model đề xuất; chưa có block chữ tương ứng.');
    }
    content.textContent = messages.join('\n\n');
    if (regionText !== undefined) content.textContent = 'Surya · Chữ OCR bên trong ảnh\n' + regionText;
    content.hidden = false;
    content.scrollTop = 0;
  }
  function bindPreview(box, ids, modelId, description, regionText) {
    box.addEventListener('pointerenter', () => preview(ids, modelId, description, regionText));
    box.addEventListener('focus', () => preview(ids, modelId, description, regionText));
  }
  data.blocks.forEach(block => {
    const card = document.getElementById(block.id);
    const box = Array.from(document.querySelectorAll('[data-layer="surya"]')).find(b => b.dataset.block === block.id);
    if (!block.classified) { card?.classList.add('uncertain'); box?.classList.add('uncertain'); }
    if (card) {
      add(card, 'strong', layoutlmBase || sourceRules ? 'Cấu trúc từ nguồn Surya' : block.classified ? 'Đã đối chiếu cấu trúc' : 'Chưa đối chiếu đầy đủ với Granite');
      block.units.forEach(unit => {
        add(card, 'p', `${unit.id} · ${unit.origin} · ${unit.headings.join(' → ')}`);
        if (unit.granite) {
          const detail = add(card, 'details', ''); add(detail, 'summary', 'Kết quả Granite dùng để đối chiếu');
          add(detail, 'p', `${unit.granite.kind} · cấp ${unit.granite.level || '—'} · ${unit.granite.text}`);
          if (!matches.has(unit.granite.id)) matches.set(unit.granite.id, []);
          matches.get(unit.granite.id).push(block.id);
        }
      });
    }
  });
  function select(ids, modelId, sourceDescription) {
    preview(ids, modelId, sourceDescription);
    document.querySelectorAll('.bbox,.block-card').forEach(item => item.classList.remove('active'));
    document.querySelectorAll('.bbox').forEach(box => {
      if (ids.includes(box.dataset.block) || (modelId && box.dataset.model === modelId) || (box.dataset.sources && JSON.parse(box.dataset.sources).some(id => ids.includes(id)))) box.classList.add('active');
    });
    ids.forEach(id => document.getElementById(id)?.classList.add('active'));
    document.getElementById(ids[0])?.scrollIntoView({behavior:'smooth', block:'nearest'});
  }
  let count = 0;
  data.observations.forEach(page => page.items.forEach(item => {
    count++;
    const ids = matches.get(item.id) || [];
    const button = add(notes, 'button', `${item.kind === 'table' ? 'BẢNG · Granite' : item.kind} · ${ids.length ? ids.join(', ') : 'Chưa khớp nguồn'}`); button.type = 'button';
    button.onclick = () => select(ids, item.id);
    bindPreview(button, ids, item.id, 'Granite · ' + item.kind);
    if (data.has_page_image && item.bbox?.length === 4) {
      const [x1,y1,x2,y2] = item.bbox;
      const box = add(surface,'button',''); box.type = 'button'; box.className = 'bbox granite-box';
      if (item.kind === 'table') box.classList.add('granite-table');
      const kindLabel = {table:'Bảng', section_header:'Tiêu đề', text:'Văn bản', picture:'Ảnh', list_item:'Mục danh sách'}[item.kind] || item.kind;
      add(box, 'span', 'Granite · ' + kindLabel).className = 'source-box-label';
      box.title = 'Granite · ' + kindLabel + ' · ' + item.id;
      box.dataset.layer = 'granite'; box.dataset.model = item.id; box.dataset.sources = JSON.stringify(ids);
      box.setAttribute('aria-label', `Granite ${item.kind}, ${ids.length ? 'đã khớp nguồn' : 'chưa khớp nguồn'}`);
      box.style.cssText = `left:${x1/10}%;top:${y1/10}%;width:${Math.max(0,x2-x1)/10}%;height:${Math.max(0,y2-y1)/10}%`;
      box.onclick = () => select(ids,item.id,box.title);
      bindPreview(box, ids, item.id, box.title);
    }
  }));
  if (!count) add(notes,'p',sourceRules ? 'Bố cục và OCR từ Surya. GLiNER chỉ tách thực thể trong chữ, không có khung ảnh model riêng.' : layoutlmBase ? 'LayoutLMv3 base lấy đặc trưng chữ–bố cục–ảnh, chưa fine-tune nên chưa có khung nhãn model riêng.' : 'Chưa có vùng Granite đã lưu cho trang này. Dữ liệu cũ cần phân tích lại; Office chưa render trang sẽ không có overlay Granite.');
  const visualNotes = document.getElementById('visual-relations');
  const visual = data.visual || {status:'not_analyzed',objects:[],relations:[],text_links:[]};
  add(visualNotes,'p',`Phân tích ảnh: ${visual.status}. Đối tượng và vị trí do model đề xuất, chưa rà soát.`);
  const visualNames = new Map(visual.objects.map(object => [object.key, object.name]));
  visual.objects.forEach(object => {
    const button = add(visualNotes,'button',`${object.name} · bbox ${object.bbox.join(', ')}${object.overlay_on_page ? '' : ' · tọa độ trong ảnh nhúng, không phải toàn trang'}`);
    button.type='button'; button.onclick=()=>select(object.block_ids,object.key);
    if (data.has_page_image && object.overlay_on_page) {
      const [x1,y1,x2,y2]=object.bbox;
      const box=add(surface,'button',''); box.type='button'; box.className='bbox visual-object-box';
      add(box, 'span', 'AI ảnh · ' + object.name).className='source-box-label';
      box.title='AI ảnh · ' + object.name;
      box.dataset.layer='visual'; box.dataset.model=object.key; box.dataset.sources=JSON.stringify(object.block_ids);
      box.style.cssText=`left:${x1/10}%;top:${y1/10}%;width:${(x2-x1)/10}%;height:${(y2-y1)/10}%`;
      box.setAttribute('aria-label',`Đối tượng chưa rà soát: ${object.name}`);
      box.onclick=()=>select(object.block_ids,object.key,box.title);
      bindPreview(box, object.block_ids, object.key, box.title);
    }
  });
  visual.relations.forEach(relation => add(visualNotes,'p',`${visualNames.get(relation.from_key)} → ${relation.kind} → ${visualNames.get(relation.to_key)} · tính trên bbox đề xuất${relation.threshold == null ? '' : ' · ngưỡng '+relation.threshold+'/1000'}`));
  visual.text_links.forEach(link => add(visualNotes,'p',`${link.source_id} → ${link.kind} → ${visualNames.get(link.object_key)} · nguồn: ${link.quote} · chưa rà soát`));
  document.querySelectorAll('[data-layer="surya"]').forEach(box => {
    box.onclick = () => select([box.dataset.block], null, box.title);
    bindPreview(box, [box.dataset.block], null, box.title);
  });
  if (data.has_page_image) data.blocks.forEach(block => {
    if (block.kind !== 'figure') return;
    (block.visual_text_regions || []).forEach(region => {
      const b = region.bbox;
      if (!Array.isArray(b) || b.length !== 4 || !b.every(Number.isFinite) || b[0] >= b[2] || b[1] >= b[3]) return;
      const box = add(surface, 'button', '');
      box.type = 'button'; box.className = 'bbox visual-text-box';
      box.dataset.layer = 'surya'; box.dataset.block = block.id;
      box.title = 'Surya · Chữ bên trong ảnh · ' + block.id;
      box.setAttribute('aria-label', box.title + ': ' + region.text);
      box.style.cssText = `left:${b[0]/10}%;top:${b[1]/10}%;width:${(b[2]-b[0])/10}%;height:${(b[3]-b[1])/10}%`;
      add(box, 'span', 'Surya · Chữ trong ảnh').className = 'source-box-label';
      bindPreview(box, [block.id], null, box.title, region.text);
      box.onclick = () => { select([block.id], null, box.title); preview([block.id], null, box.title, region.text); };
    });
  });
  data.blocks.forEach(block => {
    document.getElementById(block.id)?.addEventListener('click', event => { if (!event.target.closest('details,a,button')) select([block.id]); });
  });
  function groundSourceHash(){
    const id=window.location.hash.slice(1);
    if(!/^block-\d+$/.test(id)||!data.blocks.some(block=>block.id===id))return;
    const boxes=Array.from(document.querySelectorAll('[data-layer="surya"]')).filter(box=>box.dataset.block===id);
    document.getElementById('show-surya').checked=true;
    document.getElementById('only-uncertain').checked=false;
    visibility();
    boxes.forEach(box=>{box.classList.add('source-grounded-highlight');box.style.display='';});
    select([id],null,'Vùng block nguồn đã lưu; khung này không xác nhận OCR hoặc thông số là đúng.');
  }
  window.addEventListener('hashchange',()=>{document.querySelectorAll('.source-grounded-highlight').forEach(box=>box.classList.remove('source-grounded-highlight'));groundSourceHash();});
  groundSourceHash();
  function visibility() {
    const uncertain = document.getElementById('only-uncertain').checked;
    const unknown = new Set(data.blocks.filter(b => !b.classified).map(b => b.id));
    document.querySelectorAll('.bbox').forEach(box => {
      const enabled = document.getElementById('show-'+box.dataset.layer).checked;
      const relevant = box.dataset.layer === 'surya' ? unknown.has(box.dataset.block) : !JSON.parse(box.dataset.sources).length || JSON.parse(box.dataset.sources).some(id => unknown.has(id));
      box.hidden = !enabled || uncertain && !relevant;
    });
    data.blocks.forEach(block => { const card = document.getElementById(block.id); if (card) card.hidden = uncertain && block.classified; });
  }
  ['show-surya','show-granite','show-visual','only-uncertain'].forEach(id => document.getElementById(id).onchange = visibility);
  window.addEventListener('ai-extract-select-source',event=>{
    if(!data.blocks.some(block=>block.id===event.detail?.id))return;
    document.getElementById('show-surya').checked=true;
    document.getElementById('only-uncertain').checked=false;
    visibility();select([event.detail.id],null,event.detail.description);
    surface?.scrollIntoView({behavior:'smooth',block:'center'});
  });
  if (location.hash) document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView({block:'nearest'});
})();

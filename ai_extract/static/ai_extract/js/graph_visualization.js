(() => {
  'use strict';
  const dataTag = document.getElementById('neo4j-data'), svg = document.getElementById('neo4j-svg');
  if (!dataTag || !svg) return;
  const data = JSON.parse(dataTag.textContent), details = document.getElementById('graph-details');
  const ns = 'http://www.w3.org/2000/svg';
  const colors = {structure:'#a78bfa',evidence:'#34d399',layout_block:'#60a5fa',source_image:'#f472b6',table:'#fbbf24',table_row:'#fb923c',table_cell:'#facc15',entity:'#22d3ee',visual_object:'#f87171',image:'#f472b6'};
  const names = {structure:'Mục cấu trúc',evidence:'Đoạn nguồn',layout_block:'Block bố cục',source_image:'Ảnh trang',table:'Bảng',table_row:'Hàng',table_cell:'Ô',entity:'Thực thể',visual_object:'Đối tượng ảnh',image:'Ảnh'};
  function el(tag, attrs={}, text='') { const n=document.createElementNS(ns,tag); Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v)); if(text)n.textContent=text; return n; }
  const defs=el('defs'), marker=el('marker',{id:'graph-arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto-start-reverse'});
  marker.append(el('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'#a0aec8'})); defs.append(marker); svg.append(defs);
  const scene=el('g'); svg.append(scene);
  const edgeName = r => r.type==='RELATED' && r.predicate ? r.predicate : r.type;
  const nodes=new Map(), groups=new Map(), edges=[];
  data.nodes.forEach(n=>{if(!groups.has(n.category))groups.set(n.category,[]); groups.get(n.category).push(n);});
  let column=0, maxY=180;
  groups.forEach((items,category)=>{
    const color=colors[category] || '#cbd5e1';
    items.forEach((n,i)=>{n.x=100+column*215+Math.floor(i/16)*170;n.y=75+(i%16)*62;maxY=Math.max(maxY,n.y+70);nodes.set(n.key,n);});
    column+=Math.max(1,Math.ceil(items.length/16));
    const badge=document.createElement('span');badge.textContent=`${names[category]||category} (${items.length})`;badge.style.borderColor=color;document.getElementById('graph-legend').append(badge);
  });
  data.edges.forEach((r,i)=>{
    const a=nodes.get(r.from_key),b=nodes.get(r.to_key); if(!a||!b)return;
    const group=el('g',{class:'graph-edge',tabindex:0,role:'button','aria-label':`${edgeName(r)}: ${a.label} → ${b.label}`});
    const path=el('path',{'marker-end':'url(#graph-arrow)'}), hit=el('path',{style:'stroke:transparent;stroke-width:16px'}), label=el('text',{'text-anchor':'middle'},edgeName(r));
    group.append(path,hit,label,el('title',{},`${edgeName(r)}: ${a.label} → ${b.label}`));scene.append(group);
    const item={r,a,b,group,path,hit,label,i};edges.push(item);
    group.addEventListener('click',()=>selectEdge(item));group.addEventListener('keydown',e=>{if(e.key==='Enter')selectEdge(item);});
  });
  nodes.forEach(n=>{
    const group=el('g',{class:'graph-node',tabindex:0,role:'button','aria-label':`${names[n.category]||n.category}: ${n.label}`});
    const caption=el('text',{x:19,y:0},(n.display_label||n.label).replace(/\s+/g,' ').slice(0,34));
    if(n.preview)caption.append(el('tspan',{x:19,dy:16,style:'font-size:10px;fill:#bac8df'},n.preview.slice(0,30)+(n.preview.length>30?'…':'')));
    group.append(el('circle',{r:13,fill:colors[n.category]||'#cbd5e1'}),caption,el('title',{},n.label+'\n'+(n.text||'')));
    scene.append(group);n.group=group;
    group.addEventListener('click',()=>selectNode(n));group.addEventListener('keydown',e=>{if(e.key==='Enter')selectNode(n);});
    group.addEventListener('pointerdown',e=>{e.stopPropagation();selectNode(n);drag={node:n,startX:e.clientX,startY:e.clientY,x:n.x,y:n.y};svg.setPointerCapture(e.pointerId);});
  });
  let selected=null, view={x:0,y:0,w:Math.max(500,column*215+180),h:maxY}, drag=null;
  function updateView(){svg.setAttribute('viewBox',`${view.x} ${view.y} ${view.w} ${view.h}`);}
  function redraw(){
    nodes.forEach(n=>n.group.setAttribute('transform',`translate(${n.x} ${n.y})`));
    edges.forEach(e=>{const dx=e.b.x-e.a.x,dy=e.b.y-e.a.y,d=Math.hypot(dx,dy)||1;
      const x=e.b.x-dx*17/d,y=e.b.y-dy*17/d;
      const bend=(e.i%3-1)*20, cx=(e.a.x+x)/2-dy/d*bend,cy=(e.a.y+y)/2+dx/d*bend;
      const path=`M ${e.a.x} ${e.a.y} Q ${cx} ${cy} ${x} ${y}`;e.path.setAttribute('d',path);e.hit.setAttribute('d',path);e.label.setAttribute('x',cx);e.label.setAttribute('y',cy-5);
      e.label.style.display=document.getElementById('graph-labels').checked || selected===e || selected===e.a || selected===e.b ? '' : 'none';
    });
  }
  function field(label,value){if(value===undefined||value===null||value==='')return;const p=document.createElement('p'),b=document.createElement('strong');b.textContent=label+': ';p.append(b,document.createTextNode(typeof value==='object'?JSON.stringify(value):String(value)));details.append(p);}
  function link(url){if(!url || !url.startsWith('/ai-extract/contracts/'))return;const a=document.createElement('a');a.href=url;a.textContent='Mở trang nguồn →';details.append(a);}
  function highlight(node,edge){
    nodes.forEach(n=>{const related=!node&&!edge||n===node||edges.some(e=>(edge?e===edge:e.a===node||e.b===node)&&(e.a===n||e.b===n));n.group.classList.toggle('graph-muted',!related);n.group.classList.toggle('graph-selected',n===node);});
    edges.forEach(e=>{const active=edge?e===edge:node?e.a===node||e.b===node:true;e.group.classList.toggle('graph-muted',!active);e.group.classList.toggle('graph-selected',e===edge);});redraw();
  }
  function title(text){details.replaceChildren();const h=document.createElement('h3');h.textContent=text;details.append(h);}
  function selectNode(n){selected=n;highlight(n,null);title(n.display_label||names[n.category]||n.category);field('Nội dung',n.text||n.label);field('Block nguồn',n.source_id);field('Trang',n.page_number);field('Loại',n.kind);field('Nguồn tạo',n.origin);field('Duyệt',n.review_status);field('BBox',n.bbox_json);field('Hàng',n.row_index);field('Cột',n.column_index);link(n.source_url);
    const neighborhood=document.createElement('a'),target=new URL(window.location.href);target.searchParams.set('version_id',data.version_id);target.searchParams.set('graph_node',n.key);target.searchParams.set('graph_relation','all');target.searchParams.delete('graph_offset');target.hash='neo4j-graph';neighborhood.href=target.href;neighborhood.textContent='Xem các liên kết của node này →';neighborhood.style.display='block';details.append(neighborhood);
    if(n.source_id?.startsWith('block:') && document.getElementById('block-'+n.source_id.slice(6))){const button=document.createElement('button');button.type='button';button.textContent='Đánh dấu vùng trên ảnh';button.addEventListener('click',()=>window.dispatchEvent(new CustomEvent('ai-extract-select-source',{detail:{id:'block-'+n.source_id.slice(6),description:n.display_label||n.label}})));details.append(button);}
    field('ID graph',n.key);
    edges.filter(e=>e.a===n||e.b===n).forEach(e=>{const p=document.createElement('button');p.type='button';p.textContent=`${e.a===n?'→':'←'} ${e.r.type} · ${(e.a===n?e.b:e.a).label.slice(0,48)}`;p.style.cssText='display:block;margin-top:8px;text-align:left';p.addEventListener('click',()=>selectEdge(e));details.append(p);});
  }
  function selectEdge(e){selected=e;highlight(null,e);title(edgeName(e.r));field('Từ',e.a.label);field('Đến',e.b.label);field('Nguồn tạo',e.r.origin);field('Duyệt',e.r.review_status);field('Predicate',e.r.predicate);field('Độ tin cậy',e.r.confidence);field('Bằng chứng',e.r.quote);field('Ngưỡng bbox',e.r.threshold);link(e.r.source_url||e.a.source_url||e.b.source_url);}
  document.getElementById('graph-clear').addEventListener('click',()=>{selected=null;highlight(null,null);title('Chọn một node hoặc cạnh để xem chi tiết.');});
  document.getElementById('graph-labels').addEventListener('change',redraw);
  document.getElementById('graph-fit').addEventListener('click',()=>{view={x:0,y:0,w:Math.max(500,column*215+180),h:maxY};updateView();});
  svg.addEventListener('wheel',e=>{e.preventDefault();const rect=svg.getBoundingClientRect(),point=svg.createSVGPoint();point.x=e.clientX;point.y=e.clientY;const p=point.matrixTransform(svg.getScreenCTM().inverse());const factor=e.deltaY>0?1.12:0.88;if(view.w*factor<100||view.w*factor>15000)return;view.x=p.x-(p.x-view.x)*factor;view.y=p.y-(p.y-view.y)*factor;view.w*=factor;view.h*=factor;updateView();},{passive:false});
  svg.addEventListener('pointerdown',e=>{if(e.target.closest('.graph-node,.graph-edge'))return;drag={startX:e.clientX,startY:e.clientY,x:view.x,y:view.y};svg.setPointerCapture(e.pointerId);});
  svg.addEventListener('pointermove',e=>{if(!drag)return;const scale=Math.max(view.w/svg.clientWidth,view.h/svg.clientHeight),dx=(e.clientX-drag.startX)*scale,dy=(e.clientY-drag.startY)*scale;if(drag.node){drag.node.x=drag.x+dx;drag.node.y=drag.y+dy;redraw();}else{view.x=drag.x-dx;view.y=drag.y-dy;updateView();}});
  svg.addEventListener('pointerup',()=>{drag=null;});svg.addEventListener('pointercancel',()=>{drag=null;});
  if(!edges.length){title('Không có cạnh khớp bộ lọc.');}
  updateView();redraw();
  if(data.focus_key && nodes.has(data.focus_key))selectNode(nodes.get(data.focus_key));
})();

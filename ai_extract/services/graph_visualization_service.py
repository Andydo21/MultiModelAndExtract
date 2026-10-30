"""Prepare a source-linked view of the actual, authorized Neo4j snapshot."""
from django.urls import reverse
from ai_extract import SEMANTIC_EDGE_TYPES, RETRIEVAL_EDGE_TYPES, VISUAL_SOURCE_EDGE_TYPES
import logging
from ai_extract.repositories import document_graph_repository
from ai_extract.repositories import layout_repository
from .document_graph_service import graph_scope

logger = logging.getLogger(__name__)
RELATIONS = tuple(dict.fromkeys(('CONTAINS', 'HAS_EVIDENCE', 'HAS_VISUAL', 'APPEARS_IN', 'HAS_ROW', 'HAS_CELL',
             'NEXT', 'LEFT_OF', 'ABOVE', 'NEAR', 'MENTIONS', 'RELATED', 'EXPLAINS', 'HAS_EMBEDDING') + SEMANTIC_EDGE_TYPES + RETRIEVAL_EDGE_TYPES + VISUAL_SOURCE_EDGE_TYPES))
STRUCTURAL = ('HAS_EMBEDDING',) + RELATIONS[:7] + ('MENTIONS','RELATED') + SEMANTIC_EDGE_TYPES + VISUAL_SOURCE_EDGE_TYPES


def graph_payload(run, contract_id, version_id, page=None, relation='structure', offset=0, focus=None):
    if focus and (not isinstance(focus, str) or len(focus)>256):
        raise ValueError('Node graph không hợp lệ.')
    if relation not in ('structure', 'all', *RELATIONS):
        raise ValueError('Loại quan hệ không hợp lệ.')
    if page and (not str(page).isdecimal() or int(page) < 1):
        raise ValueError('Trang graph không hợp lệ.')
    if not str(offset).isdecimal() or int(offset) > 100000:
        raise ValueError('Vị trí graph không hợp lệ.')
    page = int(page) if page else None
    result = dict(status='not_indexed', nodes=[], edges=[], counts={}, total_nodes=0, total_edges=0,
                  matched_edges=0, offset=int(offset), limit=60)
    if run and (run.structure or {}).get('source_fingerprint') and (run.structure or {}).get('graph_index',{}).get('status')=='ready':
        try:
            kinds = STRUCTURAL if relation == 'structure' else RELATIONS if relation == 'all' else (relation,)
            result = document_graph_repository.visualization(graph_scope(run, run.structure), kinds, page, int(offset), focus=focus or None)
        except Exception:
            logger.exception('Unable to read A3 graph snapshot for run %s', run.pk)
            result['status'] = 'unavailable'
    leaves = {leaf['chunk_id']: leaf for leaf in (run.structure or {}).get('leaves', [])} if run else {}
    def source_url(page_number, source_id=''):
        if not page_number:
            return ''
        url = f'/ai-extract/contracts/{contract_id}/pages/{page_number}/?version_id={version_id}'
        return url + ('#block-'+source_id.split(':', 1)[1] if source_id.startswith('block:') else '')
    source_ids = {}
    for node in result['nodes']:
        source = node.get('source_id', '')
        if node.get('category') == 'layout_block' and ':block:' in node['key']:
            source = 'block:'+node['key'].rsplit(':block:', 1)[1]
        if source.startswith('block:') and source.split(':', 1)[1].isdecimal():
            source_ids[node['key']] = source.split(':', 1)[1]
    originals = layout_repository.graph_source_blocks(run, set(source_ids.values())) if run else {}
    kind_names = {'paragraph':'Văn bản', 'title':'Tiêu đề', 'table':'Bảng', 'figure':'Hình',
                  'figure_caption':'Chú thích hình', 'code':'Mã', 'formula':'Công thức',
                  'list_item':'Mục danh sách', 'page_header':'Đầu trang', 'page_footer':'Chân trang'}
    for node in result['nodes']:
        node['label'] = (node.get('title') or node.get('name') or node.get('text') or
                         ('Trang '+str(node['page_number']) if node.get('category') == 'source_image' else '') or
                         (f"Hàng {node['row_index']}" if node.get('category') == 'table_row' else '') or
                         node.get('kind') or node.get('category') or 'Node')
        node['source_url'] = source_url(node.get('page_number'), node.get('source_id', ''))
        node['text'] = node.get('text', '')[:1800]
        original = originals.get(source_ids.get(node['key']))
        if original:
            node['source_id'] = 'block:'+str(original['id'])
            node['block_order'] = original['block_order']
            node['page_number'] = original['page__page_number']
            if node['category'] == 'layout_block':
                node['text'] = original['text'][:1800]
            kind = kind_names.get(node.get('kind'), node.get('kind', '')) if node['category'] == 'layout_block' else 'Đoạn nguồn'
            node['display_label'] = f"P{node['page_number']} · #{node['block_order']} · {kind}"
            node['preview'] = ' '.join(node['text'].split())[:80] or 'Vùng không có chữ'
            node['label'] = node['display_label']+' — '+node['preview']
            node['source_url'] = source_url(node['page_number'], node['source_id'])
    for edge in result['edges']:
        leaf = leaves.get(edge.get('source_leaf_id'), {})
        edge['source_url'] = source_url(leaf.get('page_number'), leaf.get('source_chunk_id', ''))
    # Table keys encode the source page when the node has no page property.
    # This is a navigation link only, not a new model observation.
    for node in result['nodes']:
        if not node['source_url'] and ':layout-table:' in node['key']:
            stored_page = node['key'].split(':layout-table:', 1)[1].split(':', 1)[0]
            if stored_page.isdecimal():
                node['source_url'] = source_url(int(stored_page))
    result.update(page=page, relation=relation, focus_key=focus or '', run_id=run.pk if run else None,
                  version_id=version_id, relation_options=RELATIONS,
                  previous_offset=max(0, result['offset']-result['limit']),
                  next_offset=result['offset']+len(result['edges']),
                  has_next=result['offset']+len(result['edges']) < result['matched_edges'])
    return result


def tree_payload(tree, contract_id, version_id):
    nodes = [dict(id=n['id'], parent_id=n.get('parent_id'), kind=n.get('kind'),
                  title=n.get('title', ''), origin=n.get('structure_source', ''),
                  level=n.get('level'),hierarchy_status=n.get('hierarchy_status'),
                  hierarchy_level_status=n.get('hierarchy_level_status')) for n in tree.get('nodes', [])]
    leaves = []
    for leaf in tree.get('leaves', []):
        page = leaf.get('page_number')
        source = str(leaf.get('source_chunk_id', ''))
        url = reverse('ai_extract_page', args=[contract_id, page])+f'?version_id={version_id}' if page else ''
        if source.startswith('block:'):
            url += '#block-'+source.split(':', 1)[1]
        leaves.append(dict(id=leaf['chunk_id'], node_id=leaf['node_id'], source_id=source,
                           page=page, text=leaf.get('text', '')[:600], text_is_preview=len(leaf.get('text', '')) > 600,
                           source_url=url))
    knowledge = tree.get('knowledge_graph', {})
    entities = {e['key']:e for e in knowledge.get('entities', [])}
    relations = [dict(predicate=r['predicate'], quote=r['quote'], source_id=r['source_id'],
                      polarity=r.get('polarity', 'unspecified'), modality=r.get('modality', 'unspecified'),
                      condition_quotes=r.get('condition_quotes', []), review_status=r.get('review_status', 'unreviewed'),
                      from_name=entities.get(r['from_key'], {}).get('name', ''),
                      to_name=entities.get(r['to_key'], {}).get('name', '')) for r in knowledge.get('relations', [])[:100]]
    return dict(nodes=nodes, leaves=leaves, reading_streams=tree.get('reading_streams',[]),
                reading_policy=tree.get('reading_policy'),status=tree.get('structure_status', 'unavailable'),
                graph_index=tree.get('graph_index', {'status':'not_indexed'}),
                visual_analysis=dict(status=tree.get('visual_graph',{}).get('status','not_analyzed'),
                                     object_count=len(tree.get('visual_graph',{}).get('objects',[])),
                                     relation_count=len(tree.get('visual_graph',{}).get('relations',[]))),
                knowledge=dict(status=knowledge.get('status','not_analyzed'), relations=relations,
                               source_relations=knowledge.get('source_relations',[])[:100],
                               entity_count=len(entities), relation_count=len(knowledge.get('relations', []))),
                analysis={k: v for k, v in tree.get('analysis', {}).items() if k != 'visual_pages'})


def page_payload(run, page):
    tree = run.structure or {}
    nodes = {n['id']: n for n in tree.get('nodes', [])}
    blocks = []
    source_aligned_docling = tree.get('structure_method') == 'granite_observations_with_source_layout'
    page_blocks = list(page.blocks.all())
    region_blocks = {(b.structure_metadata or {}).get('surya_region_id'):f'block-{b.pk}' for b in page_blocks
                     if (b.structure_metadata or {}).get('surya_region_id')}
    for block in page_blocks:
        leaves = [leaf for leaf in tree.get('leaves', []) if leaf.get('source_chunk_id') == f'block:{block.pk}']
        metadata = block.structure_metadata or {}
        blocks.append(dict(id=f'block-{block.pk}', bbox=block.bbox,
                           text=block.text or '', kind=block.layout_type,
                           visual_ocr_status=metadata.get('visual_ocr_status', ''),
                           visual_text_regions=metadata.get('visual_text_regions', []),
                           parent_block_id=region_blocks.get(metadata.get('surya_parent_region_id')),
                           child_block_ids=[f'block-{b.pk}' for b in page_blocks if metadata.get('surya_region_id')
                               and (b.structure_metadata or {}).get('surya_parent_region_id')==metadata['surya_region_id']],
                           ocr_status=metadata.get('ocr_status',''),
                           classified=bool(leaves) and all(bool(l.get('structure_metadata', {}).get('granite_observation')) if source_aligned_docling else nodes.get(l['node_id'], {}).get('structure_source') not in {None, '', 'unclassified_source'} for l in leaves),
                           units=[dict(id=l['chunk_id'], headings=l.get('heading_path', []),
                                       origin=nodes.get(l['node_id'], {}).get('structure_source', 'unavailable'),
                                       granite=l.get('structure_metadata', {}).get('granite_observation') if source_aligned_docling else nodes.get(l['node_id'], {}).get('model_evidence')) for l in leaves]))
    observations = [p for p in tree.get('analysis', {}).get('visual_pages', []) if p.get('page_number') == page.page_number]
    visual=tree.get('visual_graph',{})
    page_leaves={l['chunk_id']:l for l in tree.get('leaves',[]) if l.get('page_number')==page.page_number}
    images=[i for i in visual.get('images',[]) if any(s in page_leaves for s in i['source_ids'])]
    image_ids={i['key'] for i in images}
    objects=[dict(o, overlay_on_page=next(i['image_path'] for i in images if i['key']==o['image_key'])==page.image_path,
                  block_ids=list(dict.fromkeys('block-'+page_leaves[s]['source_chunk_id'].split(':',1)[1]
                      for s in o['source_ids'] if s in page_leaves and str(page_leaves[s].get('source_chunk_id','')).startswith('block:'))))
             for o in visual.get('objects',[]) if o['image_key'] in image_ids]
    object_ids={o['key'] for o in objects}
    from .document_graph_service import visual_sources_fresh
    visual_fresh=visual_sources_fresh(tree)
    if not visual_fresh:
        for obj in objects: obj['overlay_on_page']=False
    return dict(blocks=blocks, observations=observations, has_page_image=bool(page.image_path),
                structure_provider=tree.get('analysis',{}).get('provider','source'),
                visual=dict(status=visual.get('status','not_analyzed') if visual_fresh else 'stale_visual_source', objects=objects,
                            relations=[r for r in visual.get('relations',[]) if r['from_key'] in object_ids and r['to_key'] in object_ids],
                            text_links=[l for l in visual.get('text_links',[]) if l['object_key'] in object_ids]),
                status=tree.get('structure_status', 'unavailable'))

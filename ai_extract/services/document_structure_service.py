"""Project model/parser source labels into document structure without heuristic inference.

No keyword, numbering, column, spacing, or punctuation rules decide hierarchy.
Unknown levels remain explicit; native/model provenance and source text survive.
"""
import json
import hashlib
import re

VISUAL = {'figure', 'picture', 'image', 'diagram', 'chart'}


def normalize_layout(label):
    key = re.sub(r'[-_\s]', '', str(label or '')).lower()
    return {'sectionheader': 'title', 'heading': 'title', 'text': 'paragraph',
            'picture': 'figure', 'image': 'figure', 'diagram': 'figure',
            'chart': 'figure', 'caption': 'figure_caption',
            'figurecaption': 'figure_caption', 'tablecaption': 'table_caption',
            'pageheader': 'page_header', 'pagefooter': 'page_footer',
            'listitem': 'list_item', 'listgroup': 'list_group', 'equation': 'formula'}.get(key, key or 'paragraph')


def _segments(line, maximum):
    """Split at word/sentence boundaries while retaining exact source offsets."""
    offset = 0
    while offset < len(line):
        end = min(offset + maximum, len(line))
        if end < len(line):
            candidates = list(re.finditer(r'(?<=[.!?;])\s+|\s+', line[offset:end]))
            if candidates and candidates[-1].end() >= maximum // 2:
                end = offset + candidates[-1].end()
        yield offset, line[offset:end]
        offset = end


def order_source_regions(records):
    """Sort native model ranks within each scope, then attach image text to its parent."""
    groups = {}
    for row in records:
        groups.setdefault((row.get('page_number',1),row.get('image_path','')),[]).append(row)
    ordered = []
    for _,rows in sorted(groups.items(),key=lambda pair:pair[0][0]):
        identities = {(r.get('structure_metadata') or {}).get('surya_region_id'):r for r in rows
                      if (r.get('structure_metadata') or {}).get('surya_region_id')}
        children = {}
        for region,row in identities.items():
            seen={region};parent=(row.get('structure_metadata') or {}).get('surya_parent_region_id')
            while parent:
                if parent not in identities or parent in seen:
                    row['structure_metadata'].update(source_parent_status='unresolved',reading_order_unresolved=True)
                    break
                seen.add(parent);parent=(identities[parent].get('structure_metadata') or {}).get('surya_parent_region_id')
        for row in rows:
            parent = (row.get('structure_metadata') or {}).get('surya_parent_region_id')
            children.setdefault(parent if parent in identities else None,[]).append(row)
        for siblings in children.values():
            metadata = [(r.get('structure_metadata') or {}) for r in siblings]
            if not any(m.get('reading_order_source') in {'surya_layout_model','surya_full_page_model'} for m in metadata):
                continue
            ranks = [m.get('reading_order') for m in metadata]
            valid = all(type(rank) is int and rank >= 0 for rank in ranks)
            valid = valid and len(set(ranks)) == len(ranks) and not any(m.get('reading_order_unresolved') for m in metadata)
            if valid:
                siblings.sort(key=lambda r:r['structure_metadata']['reading_order'])
            else:
                for m in metadata:
                    m.update(reading_order_unresolved=True,reading_order_status='unresolved')
        visited = set()
        def visit(row):
            identity = id(row)
            if identity in visited:return
            visited.add(identity); ordered.append(row)
            region = (row.get('structure_metadata') or {}).get('surya_region_id')
            for child in children.get(region,[]): visit(child)
        for row in children.get(None,[]):visit(row)
        for row in rows:visit(row)
    return ordered


def build_document_structure(records, title='Document', child_chars=900):
    """Use supplied labels/levels; preserve model order and source containment."""
    if child_chars < 64:
        raise ValueError('Chunk budget must be at least 64 characters.')
    identities=[r['chunk_id'] for r in records if 'chunk_id' in r]
    if len(set(identities))!=len(identities):raise ValueError('Duplicate structure source IDs')
    records = order_source_regions(annotate_page_content(records))
    nodes = [dict(id='root',parent_id=None,kind='document',title=title,ordinal=0,level=0)]
    stack = [(0,nodes[0])]
    leaves, region_owners = [], {}
    for source in records:
        metadata = {key:source[key] for key in ('heading_level','heading_source','image_path','visual_status') if key in source}
        metadata.update(source.get('structure_metadata') or {})
        kind = normalize_layout(source.get('layout_type'))
        content = source.get('text','') or ''
        if kind in VISUAL and metadata.get('ocr_status')=='skipped':
            metadata.update(display_only=True,display_text=content)
        if metadata.get('display_only') or metadata.get('text_source')=='visual_display_label':
            content=''
        atomic = metadata.get('chunk_policy') == 'surya_layout_region'
        unresolved = metadata.get('reading_order_unresolved',False)
        if unresolved:
            stack = [(0,nodes[0])]
        owner_stack = [(0,nodes[0])] if unresolved else stack
        if kind in {'page_header','page_footer','footnote'}:
            owner_stack = [(0,nodes[0])]
        region_key = (source.get('page_number',1),source.get('image_path',''),metadata.get('surya_region_id'))
        parent = metadata.get('surya_parent_region_id')
        if atomic and (parent or kind in VISUAL | {'table','page_header','page_footer','footnote'}):
            parent_key = (source.get('page_number',1),source.get('image_path',''),parent)
            ancestors = region_owners.get(parent_key) if parent and metadata.get('source_parent_status')!='unresolved' else owner_stack
            ancestors = list(ancestors or [(0,nodes[0])])
            node = dict(id=f'n{len(nodes)}',parent_id=ancestors[-1][1]['id'],kind=kind,
                        title=content[:120] if content else kind,level=ancestors[-1][0]+1,
                        ordinal=len(nodes),structure_source='surya_source_parent',
                        hierarchy_status=metadata.get('source_parent_status','source_supplied'))
            nodes.append(node)
            owner_stack = ancestors+[(node['level'],node)]
            region_owners[region_key] = owner_stack
        elif not unresolved and (kind == 'title' or metadata.get('heading_level') is not None):
            supplied = metadata.get('heading_level', metadata.get('section_level',source.get('section_level')))
            known = type(supplied) is int and 1 <= supplied <= 12
            if metadata.get('hierarchy_level_status') == 'unspecified_by_layout':
                known = False  # Older chunkers wrote a fabricated default level of 1.
            if known:
                while len(stack)>1 and stack[-1][0]>=supplied:
                    stack.pop()
                level = supplied
            else:
                stack = [(0,nodes[0])]
                level = 1  # Display depth under document, not an inferred heading level.
            node_kind = metadata.get('structure_kind','section')
            if node_kind not in {'section','article','chapter','clause','point','appendix'}:
                node_kind = 'section'
            node = dict(id=f'n{len(nodes)}',parent_id=stack[-1][1]['id'],kind=node_kind,
                        title=content.strip(),level=level,ordinal=len(nodes),
                        heading_level=supplied if known else None,
                        hierarchy_level_status=('model_supplied' if metadata.get('heading_source')=='surya_html' else 'source_supplied') if known else 'unspecified_by_model',
                        structure_source=metadata.get('heading_source') or metadata.get('layout_source') or 'source_layout_label')
            nodes.append(node)
            stack.append((level,node)); owner_stack = stack
        # Splitting controls input size only; it does not classify text or join source regions.
        parts = [content] if atomic else content.splitlines() or ([''] if kind in VISUAL or metadata.get('image_path') else [])
        for line_number,line in enumerate(parts,1):
            if not line.strip() and not (atomic or kind in VISUAL or metadata.get('image_path')):
                continue
            segments = [(0,line)] if atomic else list(_segments(line,child_chars)) if line else [(0,'')]
            for offset,text in segments:
                leaf = dict(source,text=text,chunk_id=f"{source.get('chunk_id','source')}:{line_number}:{offset}",
                            source_chunk_id=source.get('chunk_id'),node_id=owner_stack[-1][1]['id'],
                            ancestor_ids=[n['id'] for _,n in owner_stack],heading_path=[n['title'] for _,n in owner_stack],
                            source_line=line_number,source_offset=offset,
                            image_path=metadata.get('image_path') or source.get('image_path',''),
                            page_image_path=source.get('image_path',''),layout_type=kind,
                            content_hash=hashlib.sha256(text.encode()).hexdigest(),
                            retrieval_excluded=metadata.get('visual_ocr_status')=='failed'
                                or metadata.get('ocr_status') in {'skipped','failed','rejected','empty','unresolved'},
                            structure_metadata=metadata,related_chunk_ids=[])
                leaves.append(leaf)
    packed=[]
    for leaf in leaves:
        previous=packed[-1] if packed else None
        same_line=previous and previous.get('source_line_end',previous['source_line'])==leaf['source_line']
        separator='' if same_line else '\n'
        if (previous and previous['node_id']==leaf['node_id'] and previous['source_chunk_id']==leaf['source_chunk_id']
                and previous['layout_type'] not in VISUAL and len(previous['text'])+len(separator)+len(leaf['text'])<=child_chars):
            previous['text']+=separator+leaf['text']
            previous['source_line_end']=leaf['source_line']
            previous['content_hash']=hashlib.sha256(previous['text'].encode()).hexdigest()
        else:
            packed.append(leaf)
    tree=dict(schema='document-structure-v2',nodes=nodes,leaves=packed,
                structure_status='structured' if len(nodes)>1 else 'unstructured',
                structure_method='model_or_native_labels_and_supplied_levels',
                unsupported_heading_inference=False)
    return attach_reading_streams(tree)


def attach_reading_streams(tree):
    """Project source scopes into separate streams; never infer semantic continuation."""
    streams={};regions={};sources={}
    for leaf in tree.get('leaves',[]):
        meta=leaf.get('structure_metadata') or {}
        page=leaf.get('page_number',1)
        frame=leaf.get('page_image_path') or leaf.get('image_path','')
        parent=meta.get('surya_parent_region_id')
        scope=parent or meta.get('reading_order_scope') or 'page'
        kind=normalize_layout(leaf.get('layout_type'))
        channel=kind if kind in {'page_header','page_footer','footnote'} else 'content'
        signature=(page,frame,str(scope),channel)
        identity='reading:'+hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:20]
        stream=streams.setdefault(identity,dict(id=identity,page_number=page,frame=frame,scope=str(scope),
            kind='region' if parent else channel if channel!='content' else 'page',
            parent_stream_id=None,parent_source_id=None,items=[],status='source_order'))
        source=str(leaf.get('source_chunk_id') or leaf['chunk_id'])
        item=next((i for i in stream['items'] if i['source_id']==source),None)
        if item is None:
            item=dict(source_id=source,leaf_ids=[],node_id=leaf.get('node_id'),kind=kind,
                reading_order=meta.get('reading_order'),order_source=meta.get('reading_order_source',''),
                unresolved=bool(meta.get('reading_order_unresolved')),child_stream_ids=[])
            stream['items'].append(item)
        item['leaf_ids'].append(leaf['chunk_id'])
        sources[(identity,source)]=item
        if meta.get('surya_region_id'):
            regions[(page,frame,str(meta['surya_region_id']))]=(identity,source)
        leaf.update(reading_stream_id=identity,reading_stream_kind=stream['kind'])
    for stream in streams.values():
        items=stream['items']
        native=any(i['order_source'] in {'surya_layout_model','surya_full_page_model'} for i in items)
        ranks=[i['reading_order'] for i in items]
        valid=not any(i['unresolved'] for i in items)
        if native:valid &= all(type(r) is int and r>=0 for r in ranks) and len(set(ranks))==len(ranks)
        stream['status']='unresolved' if not valid else 'model_predicted' if native else 'source_order'
        if native and valid:items.sort(key=lambda i:i['reading_order'])
        if stream['kind']=='region':
            owner=regions.get((stream['page_number'],stream['frame'],stream['scope']))
            if owner and owner[0]!=stream['id']:
                stream['parent_stream_id'],stream['parent_source_id']=owner
                sources[owner]['child_stream_ids'].append(stream['id'])
            else:stream['status']='unresolved'
    # Reject cyclic parent scopes rather than manufacturing a reading chain.
    for stream in streams.values():
        seen=set();current=stream
        while current:
            if current['id'] in seen:
                for identity in seen:streams[identity]['status']='unresolved'
                break
            seen.add(current['id']);current=streams.get(current['parent_stream_id'])
    leaves={l['chunk_id']:l for l in tree.get('leaves',[])}
    visited=set();sequence=[]
    def visit(stream):
        if stream['id'] in visited:return
        visited.add(stream['id'])
        for position,item in enumerate(stream['items']):
            for identity in item['leaf_ids']:
                leaf=leaves[identity]
                leaf.update(reading_stream_position=position,reading_stream_status=stream['status'],
                    parent_reading_stream_id=stream['parent_stream_id'],
                    reading_document_position=len(sequence))
                sequence.append(identity)
            for child in item['child_stream_ids']:visit(streams[child])
    roots=[s for s in streams.values() if not s['parent_stream_id']]
    for stream in sorted(roots,key=lambda s:(s['page_number'],s['kind']!='page')):visit(stream)
    for stream in streams.values():visit(stream)
    tree.update(reading_structure_schema='source-reading-streams-v1',reading_streams=list(streams.values()),
        reading_sequence=sequence,reading_policy='scope_order_not_semantic_continuation')
    return tree


def annotate_page_content(records):
    """Copy source metadata without inferring columns, paragraph groups or order."""
    rows=[]
    for record in records:
        metadata=dict(record.get('structure_metadata') or {})
        for key in list(metadata):
            if key.startswith(('paragraph_group_','layout_column')) or key=='layout_region_source':
                metadata.pop(key)
        if metadata.get('reading_order_source')=='bbox_column_bands':
            metadata.update(reading_order=None,reading_order_unresolved=True,
                            reading_order_status='unresolved',reading_order_source='legacy_geometry_removed')
        rows.append(dict(record,structure_metadata=metadata))
    return rows


def source_fingerprint(records):
    fields = [dict(id=r['chunk_id'], text=r.get('text', ''), page=r.get('page_number'),
                   metadata=r.get('structure_metadata') or {}, image=r.get('image_path', ''),
                   layout=r.get('layout_type'), bbox=r.get('bbox') or []) for r in records]
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def structure_for_version(version, records, title):
    """Use the stored analysis in retrieval; never invoke a model on search."""
    from ai_extract.repositories import layout_repository
    from .document_structure_service import build_document_structure
    run = layout_repository.latest_run(version, completed=True)
    stored = run.structure if run else None
    if stored and stored.get('schema') in {'model-document-structure-v1', 'document-structure-v2'} and stored.get('source_fingerprint') == source_fingerprint(records):
        originals = {r['chunk_id']: r for r in records}
        leaves = [dict(originals[leaf['source_chunk_id']], **leaf) for leaf in stored['leaves']]
        # Stored leaves hold structural fields; current records add tenant,
        # document identity and URLs that are not persisted by ingestion.
        for leaf in leaves:
            for key in ('document_id', 'contract_id', 'version_id', 'version_number', 'original_name', 'source_url', 'source_location'):
                if key in originals[leaf['source_chunk_id']]:
                    leaf[key] = originals[leaf['source_chunk_id']][key]
        return attach_reading_streams(dict(stored, leaves=leaves))
    tree = build_document_structure(records, title)
    tree['structure_status'] = 'legacy_structure' if not stored else 'source_changed_requires_reanalysis'
    tree['analysis'] = dict(model_analysis_available=False)
    return tree


def analyze_document_structure(records, title):
    """Build the source tree; A3 never calls a generative structure model."""
    from django.conf import settings
    tree = build_document_structure(records, title)
    tree['source_fingerprint'] = source_fingerprint(records)
    tree['analysis'] = dict(provider='source', entity_provider=getattr(settings,'A3_ENTITY_PROVIDER','disabled'),
                            relation_provider='model_labels_and_explicit_source_metadata',
                            validation_scope='source_projection_without_heuristic_heading_inference')
    return tree


class HierarchicalChunker:
    @classmethod
    def process_hierarchical_chunks(cls, raw_chunks, doc_title="Document"):
        from .document_structure_service import normalize_layout
        chunks = [dict(record) for record in raw_chunks]
        for index, chunk in enumerate(chunks, 1):
            chunk['chunk_id'] = index
            chunk['original_layout_type'] = chunk.get('layout_type', '')
            chunk['layout_type'] = normalize_layout(chunk.get('layout_type'))
            # Keep page/region boundaries and original text. Hierarchy and leaf
            # budgets are handled once by the shared structure builder.
            if chunk['layout_type'] == 'title' and not chunk.get('heading_level'):
                chunk['heading_source'] = 'surya_layout'
                chunk['hierarchy_level_status'] = 'unspecified_by_layout'
        return chunks


def table_context(leaf):
    metadata = leaf.get('structure_metadata') or {}
    if metadata.get('table_structure_status') in {'unresolved','unavailable','rejected','failed'}:
        return {}
    headers = metadata.get('table_headers') or []
    values = metadata.get('table_values')
    if not isinstance(headers, list) or not headers:
        return {}
    if values is None and isinstance(metadata.get('table_rows'), list):
        rows = metadata['table_rows']
        row = leaf.get('table_source_row', leaf.get('source_line', 1) - 1)
        if isinstance(row, int) and 0 <= row < len(rows):
            values = rows[row]
    if not isinstance(values, list):
        return dict(headers=headers, status='row_mapping_unavailable')
    # Do not invent column headers for ragged rows or infer a unit from a
    # numeric value. Preserve header text (including explicit unit notation).
    if len(values) != len(headers):
        return dict(headers=headers, values=values, status='ambiguous_columns')
    return dict(status='source_aligned', row_number=metadata.get('table_row', leaf.get('table_source_row')),
                cells=[dict(column=i + 1, header=h, value=v) for i, (h, v) in enumerate(zip(headers, values))],
                row_context=values[0] if values else '', header_origin=metadata.get('table_header_origin', 'source_first_row_unverified'))

def context_text(leaf):
    import json
    context = table_context(leaf)
    if not context:
        return ''
    text = json.dumps(context, ensure_ascii=False)
    # Omit oversized context rather than clipping inside a cell or unit.
    return 'Table context (source metadata): ' + text if len(text) <= 2400 else 'Table context exceeds input budget; consult original table.'


from collections import defaultdict
READING_ANALYSIS_SCHEMA='surya-source-reading-v3'


def prepare_reading(run):
    """Keep supplied Surya streams and order without a semantic model review."""
    tree=run.structure
    tree.pop('reading_analysis',None)
    tree.pop('structure_review',None)
    attach_reading_streams(tree)
    tree['reading_analysis']=dict(schema=READING_ANALYSIS_SCHEMA,status='complete',
        source_fingerprint=tree.get('source_fingerprint'),provider='surya_source',
        semantic_review='disabled',
        source_ids=[l['chunk_id'] for l in tree['leaves']],
        source_order_preserved=True,new_reading_edges=0,
        policy='Preserve Surya source order and containment without GLiNER review')
    return tree

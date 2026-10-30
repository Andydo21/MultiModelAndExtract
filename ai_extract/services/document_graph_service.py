"""Project source-grounded structure and model-proposed knowledge into Neo4j."""
import math
from pathlib import Path
import hashlib
import json
import logging
import time
from django.conf import settings
from ai_extract import SEMANTIC_RELATIONS, SEMANTIC_EDGE_TYPES, RETRIEVAL_EDGE_TYPES, STRUCTURE_REVIEW_SCHEMA
from ai_extract.repositories import document_graph_repository as repository

SOURCE_RELATIONS={'NEXT','CONTAINS','HAS_EVIDENCE','HAS_ROW','HAS_CELL','HAS_VISUAL','APPEARS_IN','LEFT_OF','ABOVE'}
logger = logging.getLogger(__name__)


def infer_related_passages(tree,backend=None,*,candidate_limit=5,link_limit=2,min_logit=0.,reviewer=None):
    """Build source-to-source retrieval links during ingestion, without an LLM.

    Scores select navigation candidates. They do not establish a reference,
    semantic entailment, entity identity, or the truth of either passage.
    """
    import numpy as np
    from .contract_pipeline_service import HybridBackend
    from .gliner_entity_service import source_retrieval_eligible
    from .gliner_identity_classifier import request_passage_links,PASSAGE_REVIEW_SCHEMA,passage_link_supported,passage_pair_digest,MODEL_ID,REVISION
    if (type(candidate_limit) is not int or not 1<=candidate_limit<=20 or type(link_limit) is not int
            or not 1<=link_limit<=candidate_limit or type(min_logit) not in (int,float) or not math.isfinite(min_logit)):
        raise ValueError('Invalid related-passage retrieval policy')
    leaves=[l for l in tree.get('leaves',[]) if source_retrieval_eligible(l)]
    if len({l['chunk_id'] for l in leaves})!=len(leaves):raise ValueError('Duplicate source passage IDs')
    policy=dict(schema='bge-related-passages-v5-topic-reviewed',embedding_model='BAAI/bge-m3',
        topic_review_schema=PASSAGE_REVIEW_SCHEMA,topic_review_model=MODEL_ID,topic_review_revision=REVISION,
        scoring_input='source_text_only',
        duplicate_policy='reject_before_link_creation_and_reject_invalid_publication_batch',
        reranker_model='BAAI/bge-reranker-v2-m3',candidate_limit=candidate_limit,link_limit=link_limit,
        min_reranker_logit=min_logit,semantics='content_similarity_navigation_only',
        score_semantics='reranker_logit_not_reference_or_truth_probability')
    fingerprint=hashlib.sha256(json.dumps([policy,[(l['chunk_id'],l.get('source_chunk_id'),l.get('heading_path',[]),l['text'])
                                                 for l in leaves]],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    result=dict(policy,status='empty',fingerprint=fingerprint,source_count=len(leaves),candidate_pairs=0,links=[])
    if len(leaves)<2:return result
    backend=backend or HybridBackend()
    # Shared document headings can swamp short passages and create spurious links.
    # Score the quoted text itself; preserve headings on the original source nodes.
    texts=[l['text'] for l in leaves]
    vectors=np.asarray(backend.embedding(texts),dtype=np.float32)
    if vectors.shape!=(len(leaves),1024) or not np.isfinite(vectors).all():raise ValueError('Invalid related-passage embeddings')
    norms=np.linalg.norm(vectors,axis=1,keepdims=True)
    if not np.isfinite(norms).all() or (norms==0).any():raise ValueError('Empty related-passage embedding')
    vectors=vectors/norms;groups=[];bindings=[]
    for offset in range(0,len(leaves),128):
        similarities=vectors[offset:offset+128]@vectors.T
        for local,scores in enumerate(similarities):
            i=offset+local;source=leaves[i];source_block=source.get('source_chunk_id')
            choices=[j for j in np.argsort(-scores,kind='stable') if j!=i
                     and not (source_block and leaves[j].get('source_chunk_id')==source_block)][:candidate_limit]
            if not choices:continue
            candidates=[dict(leaves[j],heading_path=[],dense_similarity=float(scores[j])) for j in choices]
            groups.append((texts[i],candidates));bindings.append(source)
    ranked_groups=backend.rerank_many(groups) if groups else []
    if len(ranked_groups)!=len(groups):raise ValueError('Incomplete related-passage reranking')
    proposals=[]
    for source,(_,candidates),ranked in zip(bindings,groups,ranked_groups):
        originals={c['chunk_id']:c for c in candidates}
        if (len(ranked)!=len(candidates) or {c.get('chunk_id') for c in ranked}!=set(originals)
                or any(type(c.get('neural_logit')) not in (int,float) or not math.isfinite(c['neural_logit']) for c in ranked)):
            raise ValueError('Invalid related-passage ranking')
        selected=[c for c in sorted(ranked,key=lambda c:c['neural_logit'],reverse=True) if c['neural_logit']>=min_logit][:link_limit]
        for rank,item in enumerate(selected,1):
            target=originals[item['chunk_id']]
            proposals.append((source,target,rank,float(item['neural_logit'])))
    pending=[];pairs=set();targets=set()
    rejected=dict(self_pair=0,identical_text=0,repeated_target_text=0,reciprocal_or_duplicate=0)
    for source,target,rank,logit in sorted(proposals,key=lambda p:(-p[3],str(p[0]['chunk_id']),str(p[1]['chunk_id']))):
        source_id,target_id=source['chunk_id'],target['chunk_id']
        reason=repository.passage_pair_rejection(source_id,target_id,source['text'],target['text'],pairs,targets)
        if reason:
            rejected[reason]+=1;continue
        pairs.add(frozenset((source_id,target_id)));targets.add((source_id,' '.join(target['text'].split())))
        pending.append(dict(from_id=source_id,to_id=target_id,
            quote=source['text'],target_quote=target['text'],dense_similarity=target['dense_similarity'],
            rerank_logit=logit,retrieval_rank=rank,origin='bge_passage_retrieval',
            review_status='candidate',relation_semantics=policy['semantics'],score_semantics=policy['score_semantics']))
    links=[];reviews=[]
    review_pairs=[dict(id=str(i),source=l['quote'],target=l['target_quote']) for i,l in enumerate(pending)]
    for offset in range(0,len(review_pairs),128):
        batch=review_pairs[offset:offset+128]
        response=(reviewer or request_passage_links)(batch)
        decisions=response.get('decisions',[])
        if (response.get('schema')!=PASSAGE_REVIEW_SCHEMA or response.get('model')!=MODEL_ID or response.get('revision')!=REVISION
                or len(decisions)!=len(batch) or {d.get('id') for d in decisions}!={p['id'] for p in batch}):
            raise ValueError('Incomplete passage topic review')
        by_id={p['id']:p for p in batch}
        for decision in decisions:
            if (decision.get('pair_digest')!=passage_pair_digest(by_id[decision['id']])
                    or type(decision.get('accepted')) is not bool
                    or decision['accepted']!=passage_link_supported(decision)):
                raise ValueError('Ungrounded or inconsistent passage topic review')
            reviews.append(decision)
            if decision['accepted']:
                links.append(dict(pending[int(decision['id'])],relation_review_json=json.dumps(dict(
                    schema=PASSAGE_REVIEW_SCHEMA,model=MODEL_ID,revision=REVISION,**decision),ensure_ascii=False)))
    repository.validate_passage_links(links)
    return dict(result,status='complete',candidate_pairs=sum(len(g[1]) for g in groups),links=links,
                rejected_duplicate_candidates=rejected,topic_review=dict(status='complete',decisions=reviews,
                    accepted=len(links),withheld=len(pending)-len(links)))


def relation_review_key(edge):
    return hashlib.sha256(json.dumps([edge.get(k,'') for k in
        ('from_key','to_key','type','origin','source_leaf_id')],ensure_ascii=False).encode()).hexdigest()


def structure_review_fingerprint(tree,edges):
    from .gliner_entity_service import structure_review_policy
    return hashlib.sha256(json.dumps([STRUCTURE_REVIEW_SCHEMA,structure_review_policy(),tree.get('source_fingerprint'),
        sorted((n['id'],n.get('parent_id'),n.get('kind'),n.get('title')) for n in tree.get('nodes',[])),
        sorted(relation_review_key(e) for e in edges if e['type'] in SOURCE_RELATIONS)],sort_keys=True).encode()).hexdigest()


def review_surya_relations(run,request_model=None):
    """Account for every structural edge; semantic model abstention is explicit.

    Text-only GLiNER cannot certify geometry or source ownership. Those receive
    integrity checks. Reading/section proposals with source text on both sides
    receive a separate model review, never an invented fallback verdict.
    """
    from .gliner_entity_service import request_structure_pairs
    from .gliner_entity_service import source_text_eligible,structure_review_policy
    tree=run.structure
    _,nodes,all_edges=projection(run,tree,include_review=False)
    edges=[e for e in all_edges if e['type'] in SOURCE_RELATIONS]
    fingerprint=structure_review_fingerprint(tree,edges)
    previous=tree.get('structure_review',{})
    if previous.get('fingerprint')==fingerprint and previous.get('status')=='complete':return previous
    node_map={n['key']:n for n in nodes};sources={l['chunk_id']:l for l in tree['leaves']}
    eligible={i for i,l in sources.items() if source_text_eligible(l)}
    texts={n['key']:n['text'] for n in nodes if n.get('category')=='evidence' and n.get('leaf_id') in eligible}
    own_quotes={}
    for edge in all_edges:
        if edge['type']=='HAS_EVIDENCE' and edge['to_key'] in texts:
            own_quotes.setdefault(edge['from_key'],{})[edge['to_key']]=texts[edge['to_key']]
    for key,quotes in own_quotes.items():
        if node_map[key].get('category')!='structure':texts[key]='\n'.join(quotes.values())
    for node in nodes:
        if node.get('category')=='structure' and node.get('title'):
            title=node['title']
            if any(title in quote for quote in own_quotes.get(node['key'],{}).values()):
                texts[node['key']]=title
    reviews={};pairs={};pair_edges={}
    cached=previous.get('relations',{}) if previous.get('fingerprint')==fingerprint else {}
    for edge in edges:
        key=relation_review_key(edge)
        valid=(edge['from_key'] in node_map and edge['to_key'] in node_map
               and (not edge.get('source_leaf_id') or edge['source_leaf_id'] in sources))
        if valid and edge['type']=='HAS_EVIDENCE':
            valid=node_map[edge['to_key']].get('leaf_id') in sources
        if valid and edge['type'] in {'LEFT_OF','ABOVE'}:
            try:
                boxes=[json.loads(node_map[edge[k]].get('bbox_json','null')) for k in ('from_key','to_key')]
                valid=all(isinstance(b,list) and len(b)==4 and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in b)
                          and b[0]<b[2] and b[1]<b[3] for b in boxes)
                if valid:
                    a,b=boxes
                    valid=a[2]<=b[0] if edge['type']=='LEFT_OF' else a[3]<=b[1]
            except (ValueError,TypeError):valid=False
        review=dict(type=edge['type'],from_key=edge['from_key'],to_key=edge['to_key'],
                    integrity='valid' if valid else 'invalid',status='integrity_checked',origin='source_integrity')
        if not valid:review['status']='invalid'
        elif edge['type'] in {'NEXT','CONTAINS'}:
            left,right=texts.get(edge['from_key'],''),texts.get(edge['to_key'],'')
            if left.strip() and right.strip():
                review.update(status='pending',origin='gliner_structure_review')
                limit=structure_review_policy()['source_character_limit']
                if max(len(left),len(right))>limit:
                    review.update(status='unresolved',origin='review_input_validation',
                        reason='source_exceeds_review_budget',left_characters=len(left),right_characters=len(right))
                    reviews[key]=review
                    continue
                if cached.get(key,{}).get('status') in {'supported','disputed','unresolved'}:
                    review=cached[key]
                else:
                    signature=(edge['type'],left,right)
                    identity=hashlib.sha256(json.dumps(signature,ensure_ascii=False).encode()).hexdigest()
                    pairs.setdefault(identity,dict(id=identity,kind=edge['type'],left=signature[1],right=signature[2]))
                    pair_edges.setdefault(identity,[]).append(key)
                    review['review_input_id']=identity
            else:review.update(status='not_applicable',origin='source_integrity',reason='text_model_has_no_two_sided_source')
        reviews[key]=review
    pending=list(pairs.values());errors=[]
    enabled=request_model is not None or (getattr(settings,'A3_KNOWLEDGE_ENABLED',False)
        and getattr(settings,'A3_ENTITY_PROVIDER','disabled')=='gliner')
    if pending and not enabled:errors.append('gliner_disabled')
    for offset in range(0,len(pending) if enabled else 0,16):
        batch=pending[offset:offset+16]
        try:
            result=(request_model or request_structure_pairs)(batch)
            if len(result['decisions'])!=len(batch) or {r['id'] for r in result['decisions']}!={p['id'] for p in batch}:
                raise ValueError('Structure review pair coverage mismatch')
            for verdict in result['decisions']:
                if verdict.get('status') not in {'supported','disputed','unresolved'}:raise ValueError('Invalid structure verdict')
                for key in pair_edges[verdict['id']]:
                    reviews[key].update({k:v for k,v in verdict.items() if k!='id'})
        except Exception as error:
            errors.append(type(error).__name__);break
    counts={status:sum(r['status']==status for r in reviews.values()) for status in
            ('integrity_checked','not_applicable','supported','disputed','unresolved','pending','invalid')}
    tree['structure_review']=dict(schema=STRUCTURE_REVIEW_SCHEMA,fingerprint=fingerprint,
        status='partial' if errors else 'complete',relations=reviews,counts=counts,errors=errors,
        relations_total=len(edges),model_pairs=len(pairs),provider='gliner',
        inference_policy=structure_review_policy(),model_inputs=list(pairs.values()),
        policy='review_is_not_verified_structure; preserve_source_and_expose_uncertainty')
    return tree['structure_review']


def infer_relations(tree, request_model=None):
    """Use the single GLiNER extraction/validation path for semantic proposals."""
    if not getattr(settings, 'A3_KNOWLEDGE_ENABLED', False):
        return dict(status='disabled', entities=[], relations=[])
    if request_model is None and getattr(settings,'A3_ENTITY_PROVIDER','disabled')!='gliner':
        return dict(status='disabled', provider='source_layout', entities=[], relations=[],
                    reason='Structural and geometric graph only; no semantic model configured')
    from .gliner_entity_service import extract_graph_entities
    return extract_graph_entities(tree,request_model)


def graph_scope(run, tree):
    contract = run.version.contract
    scope = dict(company_id=contract.company_id, contract_id=contract.pk, version_id=run.version_id,
                 run_id=run.pk, fingerprint=tree.get('source_fingerprint', ''))
    if not scope['company_id'] or not scope['fingerprint']:
        raise ValueError('Graph requires tenant and source fingerprint')
    scope['key'] = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()
    return scope


def projection(run, tree, include_review=True):
    from .gliner_entity_service import source_layout_excluded
    scope = graph_scope(run, tree)
    nodes, edges, seen_edges = [], [], set()
    def key(logical):
        return scope['key']+':'+logical
    def node(logical, **props):
        nodes.append(dict(key=key(logical), snapshot_key=scope['key'], **props))
    def edge(a, b, kind, predicate='', quote='', source='', origin='source_structure', **metadata):
        record=dict(from_key=key(a), to_key=key(b), type=kind, predicate=predicate,
                    quote=quote, source_leaf_id=source, origin=origin, **metadata)
        identity=json.dumps(record,sort_keys=True,ensure_ascii=False)
        if identity not in seen_edges:
            edges.append(record);seen_edges.add(identity)
    node_ids = {n['id'] for n in tree['nodes']}
    for n in tree['nodes']:
        node('node:'+n['id'], category='structure', title=n.get('title',''), kind=n['kind'],
             origin=n.get('structure_source','source_structure'))
        if n.get('parent_id') in node_ids:
            model_structure=n.get('structure_source','').startswith('surya')
            edge('node:'+n['parent_id'], 'node:'+n['id'], 'CONTAINS',
                 origin='surya_model_structure' if model_structure else 'source_structure',
                 review_status='unreviewed' if model_structure else 'source_extracted')
    leaf_ids = {l['chunk_id'] for l in tree['leaves']}
    for leaf in tree['leaves']:
        meta = leaf.get('structure_metadata') or {}
        leaf_props = {k:leaf[k] for k in ('reading_stream_id','reading_stream_kind','reading_stream_status',
                      'reading_stream_position','parent_reading_stream_id','reading_document_position',
                      'source_reading_stream_id','source_reading_stream_position','source_parent_reading_stream_id',
                      'layout_type') if k in leaf}
        if meta.get('caption_text'): leaf_props['caption_text'] = meta['caption_text']
        if meta.get('in_image_text'): leaf_props['in_image_text_json'] = json.dumps(meta['in_image_text'], ensure_ascii=False)
        if meta.get('summary_description'): leaf_props['summary_description'] = meta['summary_description']
        if meta.get('table_structure_status'):
            leaf_props['table_structure_status']=meta['table_structure_status']
            warnings=(meta.get('table_structure_review') or {}).get('structure_warnings') or []
            if warnings:leaf_props['table_structure_warnings']=warnings
            if meta['table_structure_status']=='unresolved':
                leaf_props['table_structure_reason']=meta.get('table_structure_review',{}).get('reason','')
        node('leaf:'+leaf['chunk_id'], category='evidence', leaf_id=leaf['chunk_id'],
             source_id=leaf.get('source_chunk_id',''), text=leaf['text'],
             page_number=leaf.get('page_number',1), bbox_json=json.dumps(leaf.get('bbox',[])),
             image_path=leaf.get('image_path',''),
             **leaf_props)
        # Ancestors reach this source through CONTAINS; bind only its nearest
        # owner instead of repeating the same evidence edge at every level.
        owner = next((a for a in reversed(leaf.get('ancestor_ids', [])) if a in node_ids), None)
        if owner is not None:
            predicted=(leaf.get('structure_metadata') or {}).get('layout_source')=='surya' and owner!='root'
            edge('node:'+owner, 'leaf:'+leaf['chunk_id'], 'HAS_EVIDENCE',
                 origin='surya_heading_sequence' if predicted else 'source_structure',
                 review_status='unreviewed' if predicted else 'source_extracted')
    # Surya-supplied parent bindings are structural evidence, not semantic
    # assertions. Do not infer an association for captions without a parent.
    figures={}
    for leaf in tree['leaves']:
        meta=leaf.get('structure_metadata') or {}
        if leaf.get('layout_type')=='figure' and meta.get('surya_region_id'):
            figures.setdefault((leaf.get('page_number'),meta['surya_region_id']),[]).append(leaf)
    for child in tree['leaves']:
        meta=child.get('structure_metadata') or {}
        parent=meta.get('surya_parent_region_id')
        if not parent or meta.get('source_parent_status')=='unresolved' or not child.get('text','').strip() or child.get('retrieval_excluded'):
            continue
        for figure in figures.get((child.get('page_number'),parent),[]):
            edge('leaf:'+figure['chunk_id'],'leaf:'+child['chunk_id'],'HAS_FIGURE_TEXT',
                 'surya_parent',child['text'],figure['chunk_id'],'surya_source_parent',
                 target_source_leaf_id=child['chunk_id'],review_status='unreviewed',
                 relation_semantics='source_parent_not_verified_reference')
    for figure,caption,spatial in figure_caption_candidates(tree.get('leaves',[])):
        is_caption=caption.get('layout_type')=='figure_caption'
        edge('leaf:'+figure['chunk_id'],'leaf:'+caption['chunk_id'],
             'FIGURE_CAPTION_CANDIDATE' if is_caption else 'FIGURE_NEAR_TEXT_CANDIDATE',
             'nearby_caption_candidate' if is_caption else 'nearby_text_candidate',
             caption['text'],figure['chunk_id'],'source_bbox_neighbor_candidate',
             target_source_leaf_id=caption['chunk_id'],review_status='candidate',
             relation_semantics='layout_proximity_only_not_verified_ownership')
    originals = {l['chunk_id']:l for l in tree['leaves']}
    for rel in tree.get('relations', []):
        if rel.get('from_id') in leaf_ids and rel.get('to_id') in leaf_ids:
            edge('leaf:'+rel['from_id'], 'leaf:'+rel['to_id'], 'RELATED',
                 str(rel.get('kind', 'reference')), originals[rel['from_id']]['text'],
                 rel['from_id'], 'structure_model_relation_not_semantic_verification')
    knowledge = tree.get('knowledge_graph', {})
    entities = {e['key']:e for e in knowledge.get('entities', [])}
    entity_ids = entities.keys()
    entity_nodes={};entity_groups={}
    for entity in entities.values():
        logical='entity:'+str(entity.get('canonical_key') or entity['key'])
        entity_nodes[entity['key']]=logical
        entity_groups.setdefault(logical,[]).append(entity)
    for logical,occurrences in entity_groups.items():
        first=occurrences[0]
        node(logical,category='entity',kind='entity',name=first['name'],
             occurrence_count=len(occurrences),
             occurrences_json=json.dumps([{k:e.get(k) for k in ('key','name','source_ids','start','end','confidence')}
                 for e in occurrences],ensure_ascii=False),
             origin='gliner_source_span',review_status='unreviewed')
        for entity in occurrences:
            for source in entity['source_ids']:
                if source in leaf_ids and not source_layout_excluded(originals[source]):
                    edge('leaf:'+source,logical,'MENTIONS',quote=entity['name'],source=source,
                         origin=entity.get('origin','source_structure'),review_status='unreviewed',
                         occurrence_key=entity['key'],from_char_start=entity.get('start'),
                         from_char_end=entity.get('end'),confidence=entity.get('confidence'))
    for rel in knowledge.get('relations', []):
        if rel['from_key'] in entity_ids and rel['to_key'] in entity_ids and rel['source_id'] in leaf_ids:
            if rel['predicate']=='references' and rel.get('target_source_id',rel['source_id'])!=rel['source_id']:
                continue  # Store this once as a source-to-source REFERENCES edge below.
            kind=SEMANTIC_RELATIONS.get(rel['predicate'],'RELATED') if rel.get('origin')=='gliner_relex_source_span' else 'RELATED'
            edge(entity_nodes[rel['from_key']],entity_nodes[rel['to_key']],kind,rel['predicate'],
                 rel['quote'], rel['source_id'], rel['origin'], review_status='unreviewed',confidence=rel.get('confidence'),
                 polarity=rel.get('polarity', 'unspecified'), modality=rel.get('modality', 'unspecified'),
                 condition_quotes=rel.get('condition_quotes', []), scope_quote=rel.get('scope_quote', rel['quote']),
                 qualification_quote=rel.get('qualification_quote', ''),
                 relation_review_json=rel.get('relation_review_json',''),
                 from_char_start=entities[rel['from_key']].get('start'),from_char_end=entities[rel['from_key']].get('end'),
                 to_char_start=entities[rel['to_key']].get('start'),to_char_end=entities[rel['to_key']].get('end'),
                 target_source_leaf_id=rel.get('target_source_id'),target_quote=rel.get('target_quote',''),
                 qualification_validation=rel.get('qualification_validation', 'legacy_unqualified'))
    for link in knowledge.get('source_relations',[]):
        if link['from_id'] in leaf_ids and link['to_id'] in leaf_ids:
            typed=link.get('origin') in {'gliner_relex_source_span','gliner2_described_relation'}
            kind=SEMANTIC_RELATIONS.get(link['kind'],'RELATED') if typed else 'RELATED'
            a,b=entities.get(link.get('from_key'),{}),entities.get(link.get('to_key'),{})
            edge('leaf:'+link['from_id'],'leaf:'+link['to_id'],kind,link['kind'],link['from_quote'],
                 link['from_id'],link['origin'], target_quote=link['to_quote'],review_status='unreviewed',
                 target_source_leaf_id=link['to_id'],confidence=link.get('confidence'),
                 reference_type=link.get('reference_type',''),
                 endpoint_confidences_json=json.dumps(link.get('endpoint_confidences',{})),
                 score_semantics=link.get('score_semantics','legacy_model_score'),
                 from_char_start=a.get('start',link.get('from_char_start')),from_char_end=a.get('end',link.get('from_char_end')),
                 to_char_start=b.get('start',link.get('to_char_start')),to_char_end=b.get('end',link.get('to_char_end')))
    leaf_map={leaf['chunk_id']:leaf for leaf in tree.get('leaves',[])}
    passage_links=tree.get('related_passages',{}).get('links',[])
    for link in passage_links:
        source,target=leaf_map.get(link.get('from_id')),leaf_map.get(link.get('to_id'))
        if (not source or not target or source['chunk_id']==target['chunk_id']
                or link.get('quote')!=source['text'] or link.get('target_quote')!=target['text']
                or any(type(link.get(k)) not in (int,float) or not math.isfinite(link[k]) for k in ('dense_similarity','rerank_logit'))):
            raise ValueError('Related-passage edge must bind both original sources and finite relevance scores')
    repository.validate_passage_links(passage_links)
    for link in passage_links:
        source,target=leaf_map[link['from_id']],leaf_map[link['to_id']]
        edge('leaf:'+source['chunk_id'],'leaf:'+target['chunk_id'],'RELATED_PASSAGE','content_related',
             source['text'],source['chunk_id'],'bge_passage_retrieval',target_quote=target['text'],
             target_source_leaf_id=target['chunk_id'],review_status='candidate',
             dense_similarity=link['dense_similarity'],rerank_logit=link['rerank_logit'],retrieval_rank=link['retrieval_rank'],
             relation_review_json=link.get('relation_review_json'),
             relation_semantics='content_similarity_navigation_only',score_semantics='reranker_logit_not_reference_or_truth_probability')
    visual=tree.get('visual_graph',{})
    image_ids={i['key'] for i in visual.get('images',[])}
    object_ids={o['key'] for o in visual.get('objects',[])}
    for image in visual.get('images',[]):
        node('image:'+image['key'],category='image',image_path=image['image_path'],sha256=image['sha256'],
             width=image['width'],height=image['height'],coordinate_frame='image_normalized_0_1000')
        for source in image['source_ids']:
            if source in leaf_ids: edge('leaf:'+source,'image:'+image['key'],'HAS_VISUAL',source=source)
    for obj in visual.get('objects',[]):
        node('object:'+obj['key'],category='visual_object',name=obj['name'],bbox_json=json.dumps(obj['bbox']),
             image_key=obj['image_key'],origin=obj['origin'],review_status='unreviewed')
        if obj['image_key'] in image_ids:
            edge('object:'+obj['key'],'image:'+obj['image_key'],'APPEARS_IN',origin=obj['origin'],review_status='unreviewed')
    for rel in visual.get('relations',[]):
        if rel['from_key'] in object_ids and rel['to_key'] in object_ids:
            edge('object:'+rel['from_key'],'object:'+rel['to_key'],rel['kind'],origin=rel['origin'],
                 threshold=rel.get('threshold'),review_status='unreviewed')
    for link in visual.get('text_links',[]):
        if link['source_id'] in leaf_ids and link['object_key'] in object_ids:
            edge('leaf:'+link['source_id'],'object:'+link['object_key'],'EXPLAINS',link['kind'],
                 link['quote'],link['source_id'],link['origin'],review_status='unreviewed')
    append_layout_edges(tree, node, edge)
    from .layout_extractor import colpali_point_id
    from .contract_pipeline_service import COLPALI_COLLECTION
    acknowledged = set(tree.get('colpali_point_ids', []))
    if acknowledged:
        for page in run.pages.prefetch_related('blocks').order_by('page_number'):
            for block in page.blocks.all():
                point_id = colpali_point_id(run.pk, page.pk, block.pk)
                logical = f'layout-block:{page.page_number}:block:{block.pk}'
                if (point_id not in acknowledged or block.layout_type != 'figure'
                        or not any(n['key'] == key(logical) for n in nodes)):
                    continue
                node('embedding:'+point_id, category='visual_embedding', page_number=page.page_number,
                     point_id=point_id,image_id=point_id, collection=COLPALI_COLLECTION, model='vidore/colpali-v1.2',
                     storage='qdrant', status='indexed')
                edge(logical, 'embedding:'+point_id, 'HAS_EMBEDDING', origin='qdrant_acknowledged_write')
    edges=compact_relations(edges)
    review=tree.get('structure_review',{})
    if include_review and review.get('fingerprint')==structure_review_fingerprint(tree,edges):
        for edge in edges:
            judgment=review.get('relations',{}).get(relation_review_key(edge))
            if judgment:
                if judgment['origin']=='gliner_structure_review':
                    edge.update(source_review_status=edge.get('review_status','unreviewed'),review_status='model_'+judgment['status'])
                edge.update(structure_review_status=judgment['status'],review_origin=judgment['origin'],
                            review_evidence_json=json.dumps(judgment.get('evidence',[]),ensure_ascii=False))
    return scope, nodes, [e for e in edges if e.get('structure_review_status')!='invalid']


def compact_relations(edges):
    """Remove redundant graph paths without inferring new semantic facts.

    Directional geometry and bounding boxes remain available. A generic
    association is redundant only when its full evidence and qualifiers match
    a specific assertion; sharing endpoints alone is not sufficient.
    """
    def assertion(edge):
        return json.dumps({k:v for k,v in edge.items() if k != 'type'},
                          sort_keys=True, ensure_ascii=False)
    specific = {assertion(e) for e in edges if e['type'] in SEMANTIC_EDGE_TYPES}
    rows = {}
    cells = {}
    for edge in edges:
        if edge['type'] == 'HAS_ROW':
            rows.setdefault(edge['from_key'], set()).add(edge['to_key'])
        elif edge['type'] == 'HAS_CELL':
            cells.setdefault(edge['from_key'], []).append(edge)
    def redundant_cell(edge):
        # Keep a direct binding if the row path has different provenance.
        for row in rows.get(edge['from_key'], ()):
            for child in cells.get(row, ()):
                if child['to_key'] == edge['to_key'] and all(
                        child.get(k) == v for k,v in edge.items() if k != 'from_key'):
                    return True
        return False
    return [e for e in edges if e['type'] != 'NEAR'
            and not (e['type'] == 'RELATED' and assertion(e) in specific)
            and not (e['type'] == 'HAS_CELL' and redundant_cell(e))]


def sync_graph(run):
    tree = run.structure
    if not getattr(settings, 'A3_GRAPH_ENABLED', False):
        return dict(status='disabled')
    if not getattr(settings, 'NEO4J_URI', '') or not getattr(settings, 'NEO4J_PASSWORD', ''):
        return dict(status='not_configured')
    try:
        if getattr(settings,'A3_KNOWLEDGE_ENABLED',False):
            validate_prepared(tree)
        if not visual_sources_fresh(tree):
            return dict(status='stale_visual_source')
        scope, nodes, edges = projection(run, tree)
        if tree.get('pregraph_analysis'):
            identity=tree['identity_resolution']
            scope.update(identity_prepared=True,entity_compacted=True,entity_resolution_result=json.dumps(
                {k:identity[k] for k in ('schema','status','model','revision','mentions','linked',
                    'unresolved','canonical_entities')},ensure_ascii=False))
        from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError
        for attempt in range(3):
            try:
                repository.replace_snapshot(scope, nodes, edges)
                break
            except (ServiceUnavailable, SessionExpired, TransientError):
                if attempt == 2:
                    raise
                logger.warning('Neo4j graph publication failed temporarily; retry %s/2', attempt + 1, exc_info=True)
                time.sleep(2 ** attempt)
        return dict(status='ready', fingerprint=scope['fingerprint'], nodes=len(nodes), edges=len(edges),
            entity_nodes=sum(n.get('category')=='entity' for n in nodes),
            entity_occurrences=len(tree.get('knowledge_graph',{}).get('entities',[])),
            structure_review=tree.get('structure_review',{}).get('status','not_requested'))
    except Exception as exc:
        logger.exception('Neo4j graph publication failed for run %s', getattr(run, 'id', None))
        return dict(status='unavailable', error_type=type(exc).__name__)


def cleanup_old_run_graphs(run):
    """Only a persisted completed run with a ready graph may retire history."""
    if run.status != 'COMPLETED' or run.structure.get('graph_index',{}).get('status') != 'ready':
        return dict(status='skipped', deleted_snapshots=0)
    try:
        scope=graph_scope(run,run.structure)
        if run.structure['graph_index'].get('fingerprint') != scope['fingerprint']:
            return dict(status='skipped_stale_graph',deleted_snapshots=0)
        return dict(status='completed',deleted_snapshots=repository.prune_old_runs(scope))
    except Exception as error:
        # Cleanup failure must not turn a successfully persisted extraction
        # into FAILED. A later successful run can retry the scoped deletion.
        return dict(status='failed',error_type=type(error).__name__)


def resolve_shared_entities(scope, embedding=None, request_model=None):
    """Reject the retired graph-first identity path, including manual callers."""
    raise RuntimeError('Graph-first entity resolution is retired; use whole-document pregraph preparation')


def enrich_and_sync(run):
    from .layout_service import prepare_document
    from .inference_runtime_service import PipelineProgress
    prepare_document(run,PipelineProgress(run.pk))
    run.structure['graph_index'] = sync_graph(run)


def read_context_graph(run, tree, selected, seed_ids=None):
    """Hydrate stored edge identities from current A3 data, never Neo4j text."""
    empty = dict(state='not_indexed', nodes=[], edges=[], partial=False)
    if not getattr(settings, 'A3_GRAPH_ENABLED', False) or tree.get('graph_index', {}).get('status') != 'ready':
        return empty
    try:
        scope = graph_scope(run, tree)
        if tree['graph_index'].get('fingerprint') != scope['fingerprint'] or not visual_sources_fresh(tree):
            return dict(empty, state='stale')
        selected_ids = {leaf['chunk_id'] for leaf in selected}
        if not selected_ids:
            return dict(empty, state='empty')
        stored = repository.context_edges(scope, sorted(selected_ids),
            seed_ids=sorted(selected_ids if seed_ids is None else selected_ids & set(seed_ids)))
        _, nodes, edges = projection(run, tree)
        identity = lambda edge: (edge['from_key'], edge['to_key'], edge['type'],
                                 edge.get('predicate', ''), edge.get('source_leaf_id', ''))
        current_edges = {}
        for edge in edges:current_edges.setdefault(identity(edge),[]).append(edge)
        accepted=[];seen=set()
        for stored_edge in stored['edges']:
            for edge in current_edges.get(identity(stored_edge),[]):
                if stored_edge.get('occurrence_key') and stored_edge['occurrence_key']!=edge.get('occurrence_key'):continue
                signature=json.dumps(edge,sort_keys=True,ensure_ascii=False)
                if signature not in seen:accepted.append(edge);seen.add(signature)
        node_map = {node['key']: node for node in nodes}
        bindings = {node['key']: {node['leaf_id']} & selected_ids for node in nodes if node.get('category') == 'evidence'}
        # Bind headings, table cells and image/object context to actual source
        # leaves. CONTAINS propagates up only; root membership never binds all
        # children to the same paragraph.
        while True:
            changed = False
            for edge in edges:
                a, b, kind = edge['from_key'], edge['to_key'], edge['type']
                pairs = [(a,b)] if kind in {'HAS_EVIDENCE','CONTAINS'} else [(b,a)] if kind in {'MENTIONS','HAS_VISUAL','EXPLAINS','HAS_ROW','HAS_CELL','HAS_EMBEDDING'} else [(a,b)] if kind == 'APPEARS_IN' and node_map[a].get('category') == 'visual_object' else []
                for target, origin in pairs:
                    before = len(bindings.setdefault(target,set()))
                    bindings[target].update(bindings.get(origin,set()))
                    changed |= len(bindings[target]) != before
            if not changed:
                break
        keys = sorted({edge[side] for edge in accepted for side in ('from_key','to_key')
                       if bindings.get(edge[side])} | {key for key,node in node_map.items()
                       if node.get('category')=='evidence' and bindings.get(key)})
        aliases = {key: f'v{run.version_id}:n{i}' for i,key in enumerate(keys,1)}
        result_nodes = []
        for key in keys:
            node = node_map[key]
            item = dict(id=aliases[key], kind=node.get('kind') or node['category'],
                        label=node.get('title') or node.get('name') or node.get('text','') or node.get('source_id') or node['category'],
                        source_leaf_ids=sorted(bindings[key]), origin=node.get('origin','source_structure'))
            if node.get('bbox_json'):
                item['bbox'] = json.loads(node['bbox_json'])
            for field in ('page_number','image_path','reading_stream_id','reading_stream_kind',
                          'reading_stream_status','reading_stream_position','parent_reading_stream_id','reading_document_position',
                          'table_structure_status','table_structure_reason'):
                if field in node: item[field] = node[field]
            if node.get('row_index') is not None: item['row'] = node['row_index']
            if node.get('column_index') is not None: item['column'] = node['column_index']
            for field in ('rowspan','colspan','header'):
                if node.get(field) is not None:item[field]=node[field]
            result_nodes.append(item)
        result_edges = []
        for edge in accepted:
            if edge['from_key'] not in aliases or edge['to_key'] not in aliases:
                continue
            item = dict(from_id=aliases[edge['from_key']], to_id=aliases[edge['to_key']],
                        type=edge['type'], predicate=edge.get('predicate',''),
                        origin=edge.get('origin','source_structure'),
                        review_status=edge.get('review_status','unreviewed' if edge['type'] in ('RELATED','EXPLAINS',*SEMANTIC_EDGE_TYPES) else 'source_extracted'),
                        source_leaf_ids=[edge['source_leaf_id']] if edge.get('source_leaf_id') in selected_ids else [])
            target=edge.get('target_source_leaf_id')
            if target in selected_ids and target not in item['source_leaf_ids']:item['source_leaf_ids'].append(target)
            for field in ('quote','target_quote','confidence','polarity','modality','condition_quotes','scope_quote',
                          'structure_review_status','source_review_status','review_origin','review_evidence_json',
                          'qualification_quote','relation_review_json','reading_stream_id','occurrence_key','from_char_start','from_char_end','to_char_start','to_char_end',
                          'dense_similarity','rerank_logit','retrieval_rank','relation_semantics','score_semantics'):
                if edge.get(field) or field in {'from_char_start','from_char_end','to_char_start','to_char_end',
                                                'dense_similarity','rerank_logit','retrieval_rank'} and edge.get(field) is not None:
                    item[field] = edge[field]
            result_edges.append(item)
        return dict(state='ready', version_id=run.version_id, run_id=run.pk,
                    retrieved_edges=len(stored['edges']),delivered_edges=len(result_edges),delivered_nodes=len(result_nodes),
                    seed_leaf_ids=sorted(selected_ids if seed_ids is None else selected_ids & set(seed_ids)),
                    context_scope='direct_seed_relations_and_source_paths', nodes=result_nodes, edges=result_edges,
                    omitted_edges=len(stored['edges'])-len(result_edges),
                    partial=stored['truncated'] or len(result_edges)<len(stored['edges']),
                    policy='source_bound_graph_context_not_verified_semantic_truth')
    except Exception as error:
        return dict(empty, state='unavailable', error_type=type(error).__name__)


def expand_graph(run, tree, seeds):
    """Rehydrate current authorized leaves; never trust graph text as evidence."""
    status = tree.get('graph_index', {}).get('status', 'not_indexed')
    if not getattr(settings, 'A3_GRAPH_ENABLED', False) or status != 'ready' or not seeds:
        return tree, dict(state=status, candidates=0)
    try:
        if not visual_sources_fresh(tree):
            return tree, dict(state='stale_visual_source',candidates=0)
        scope = graph_scope(run, tree)
        if tree['graph_index'].get('fingerprint') != scope['fingerprint']:
            return tree, dict(state='stale', candidates=0)
        rows = repository.neighbors(scope, [s['chunk_id'] for s in seeds])
        allowed = {leaf['chunk_id'] for leaf in tree['leaves']}
        seed_ids = {s['chunk_id'] for s in seeds}
        by_seed = {seed:[] for seed in seed_ids};reasons={};seed_reasons={}
        for row in rows:
            seed,target = row.get('seed_id'),row.get('leaf_id')
            if seed in by_seed and target in allowed and target not in by_seed[seed]:
                by_seed[seed].append(target)
            if seed in by_seed and target in allowed:
                reasons.setdefault(target,set()).add(row.get('reason','source_relation'))
                seed_reasons.setdefault(seed,{}).setdefault(target,set()).add(row.get('reason','source_relation'))
        similarity_only={target for target,why in reasons.items() if why=={'retrieval_similarity'}}
        related = {target for targets in by_seed.values() for target in targets}
        leaves = [dict(leaf, graph_related_chunk_ids=by_seed[leaf['chunk_id']],
                       graph_relation_reasons={target:sorted(why) for target,why in seed_reasons.get(leaf['chunk_id'],{}).items()},
                       graph_similarity_only_chunk_ids=sorted(similarity_only & set(by_seed[leaf['chunk_id']])))
                  if leaf['chunk_id'] in by_seed else leaf for leaf in tree['leaves']]
        return dict(tree, leaves=leaves), dict(state='ready', candidates=len(related),
            seed_neighbors=by_seed, unreviewed_relations_retrieval_only=True)
    except Exception as exc:
        return tree, dict(state='unavailable', candidates=0, error_type=type(exc).__name__)


def valid_bbox(box):
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError('Expected an image-normalized bounding box')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1000 for v in box):
        raise ValueError('Invalid bounding box coordinates')
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError('Empty or reversed bounding box')
    return box


def spatial_relations(objects, near_distance=80):
    """Local, aligned, unobstructed neighbors; never semantic connectivity.

    Coordinates are normalized to 0..1000. Keep one nearest right/below
    neighbor per box, and at most two mutual NEAR neighbors per box.
    """
    if not math.isfinite(near_distance) or not 0 <= near_distance <= 300:
        raise ValueError('Invalid adjacency threshold')
    boxes = [valid_bbox(obj['bbox']) for obj in objects]
    nearest, nearby = {}, {i: [] for i in range(len(objects))}

    def blocked(source, target, corridor):
        return any(k not in (source,target)
                   and min(c[2],corridor[2]) > max(c[0],corridor[0])
                   and min(c[3],corridor[3]) > max(c[1],corridor[1])
                   for k,c in enumerate(boxes))

    def directional(source, target, kind):
        a, b = boxes[source], boxes[target]
        if kind == 'LEFT_OF':
            gap = b[0]-a[2]
            lo, hi = max(a[1],b[1]), min(a[3],b[3])
            overlap = (hi-lo)/min(a[3]-a[1],b[3]-b[1])
            corridor = [a[2],lo,b[0],hi]
        else:
            gap = b[1]-a[3]
            lo, hi = max(a[0],b[0]), min(a[2],b[2])
            overlap = (hi-lo)/min(a[2]-a[0],b[2]-b[0])
            corridor = [lo,a[3],hi,b[1]]
        if not 0 <= gap <= near_distance or overlap < .5:
            return
        # Another block in the shared corridor separates these endpoints.
        if blocked(source,target,corridor):
            return
        rank = (gap, -overlap, str(objects[target]['key']))
        key = (source,kind)
        if key not in nearest or rank < nearest[key][0]:
            nearest[key] = (rank,target)

    for i, a in enumerate(boxes):
        for j in range(i+1,len(boxes)):
            b = boxes[j]
            if a[2] <= b[0]: directional(i,j,'LEFT_OF')
            elif b[2] <= a[0]: directional(j,i,'LEFT_OF')
            if a[3] <= b[1]: directional(i,j,'ABOVE')
            elif b[3] <= a[1]: directional(j,i,'ABOVE')
            dx, dy = max(a[0]-b[2],b[0]-a[2],0), max(a[1]-b[3],b[1]-a[3],0)
            overlaps = min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1])
            distance = math.hypot(dx,dy)
            if not overlaps and distance <= near_distance:
                x = sorted((min(a[2],b[2]),max(a[0],b[0])))
                y = sorted((min(a[3],b[3]),max(a[1],b[1])))
                if not blocked(i,j,[x[0],y[0],x[1],y[1]]):
                    nearby[i].append((distance,str(objects[j]['key']),j))
                    nearby[j].append((distance,str(objects[i]['key']),i))
    close = {i:{row[2] for row in sorted(rows)[:2]} for i,rows in nearby.items()}
    pairs = [(i,target,kind) for (i,kind),(_,target) in nearest.items()]
    directional_pairs = {frozenset((i,j)) for i,j,_ in pairs}
    pairs.extend((i,j,'NEAR') for i,targets in close.items() for j in targets
                 if str(objects[i]['key']) < str(objects[j]['key']) and i in close[j]
                 and frozenset((i,j)) not in directional_pairs)
    return [dict(from_key=objects[i]['key'],to_key=objects[j]['key'],kind=kind,
                 origin='local_bbox_neighbors_not_semantics',review_status='unreviewed',
                 threshold=near_distance)
            for i,j,kind in sorted(pairs,key=lambda p:(str(objects[p[0]]['key']),str(objects[p[1]]['key']),p[2]))]


def figure_caption_candidates(leaves):
    """Offer nearby text as unverified layout candidates, never as ownership."""
    pages={}
    for leaf in leaves:
        if leaf.get('layout_type') not in {'figure','figure_caption','paragraph','title','list_item','list_group','footnote','code','formula'}:
            continue
        box=leaf.get('bbox') or []
        if (len(box)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in box)
                or not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000)):
            continue
        pages.setdefault(leaf.get('page_number'),[]).append(leaf)
    result=[]
    for page in pages.values():
        by_id={leaf['chunk_id']:leaf for leaf in page}
        figure_regions={(leaf.get('structure_metadata') or {}).get('surya_region_id') for leaf in page if leaf.get('layout_type')=='figure'}
        figure_regions.discard(None)
        for relation in spatial_relations([dict(key=leaf['chunk_id'],bbox=leaf['bbox']) for leaf in page]):
            a,b=by_id[relation['from_key']],by_id[relation['to_key']]
            if (a.get('layout_type')=='figure') == (b.get('layout_type')=='figure'):
                continue
            figure,caption=(a,b) if a['layout_type']=='figure' else (b,a)
            meta=caption.get('structure_metadata') or {}
            if (caption.get('text','').strip() and not caption.get('retrieval_excluded')
                    and meta.get('surya_parent_region_id') not in figure_regions):
                result.append((figure,caption,relation))
    return result


def visual_sources_fresh(tree):
    root=Path(settings.MEDIA_ROOT).resolve()
    for image in tree.get('visual_graph',{}).get('images',[]):
        try:
            path=(root/image['image_path']).resolve()
            if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size>20*1024*1024: return False
            if hashlib.sha256(path.read_bytes()).hexdigest()!=image['sha256']: return False
        except (ValueError,OSError): return False
    return True


def append_layout_edges(tree, node, edge):
    pages, spatial_pages, blocks, tables, created, contained = {}, {}, {}, {}, set(), set()
    row_sources = []
    table_sources = {}
    structure_nodes={n['id']:n for n in tree.get('nodes',[])}
    owner_sources={}
    for leaf in tree.get('leaves',[]):
        owner_sources.setdefault(leaf.get('node_id'),set()).add(leaf.get('source_chunk_id') or leaf['chunk_id'])
    def add(identity, **props):
        if identity not in created:
            node(identity, **props); created.add(identity)
    for leaf in tree.get('leaves', []):
        page = leaf.get('page_number', 1)
        image = leaf.get('page_image_path') or leaf.get('image_path')
        if image:
            image_id = 'layout-image:'+hashlib.sha256(image.encode()).hexdigest()
            add(image_id, category='source_image', image_path=image, page_number=page)
            edge('leaf:'+leaf['chunk_id'], image_id, 'HAS_VISUAL', origin='source_image_path')
        source = str(leaf.get('source_chunk_id') or leaf['chunk_id'])
        identity = f'layout-block:{page}:{source}'
        box = leaf.get('bbox') or []
        if identity not in blocks and len(box) == 4 and box[0] < box[2] and box[1] < box[3] and all(0 <= x <= 1000 for x in box):
            source_metadata = leaf.get('structure_metadata') or {}
            blocks[identity] = dict(key=identity, bbox=box,
                                   stream_id=leaf.get('reading_stream_id'),
                                   order_unresolved=source_metadata.get('reading_order_unresolved',False) or leaf.get('reading_stream_status')=='unresolved',
                                   order_source=source_metadata.get('reading_order_source',''),
                                   rank=source_metadata.get('reading_order'))
            add(identity, category='layout_block', kind=leaf.get('layout_type','paragraph'), bbox_json=json.dumps(box), page_number=page, image_path=image or '',
                reading_order=source_metadata.get('reading_order'),
                reading_order_source=source_metadata.get('reading_order_source',''),
                reading_order_scope=source_metadata.get('reading_order_scope','page'),
                reading_order_unresolved=blocks[identity]['order_unresolved'],
                **{k:leaf[k] for k in ('reading_stream_id','reading_stream_kind','reading_stream_status',
                    'reading_stream_position','parent_reading_stream_id','reading_document_position') if k in leaf})
            if image:
                edge(identity, image_id, 'APPEARS_IN', origin='source_page_coordinates')
            # Page reading and reading inside each picture are separate sequences.
            scope = source_metadata.get('reading_order_scope') or source_metadata.get('surya_parent_region_id') or 'page'
            pages.setdefault((page, image or '', leaf.get('reading_stream_id') or scope), []).append(blocks[identity])
            spatial_pages.setdefault((page, image or ''), []).append(blocks[identity])
        if identity in blocks:
            edge(identity, 'leaf:'+leaf['chunk_id'], 'HAS_EVIDENCE', origin='source_block_id')
        metadata = leaf.get('structure_metadata') or {}
        if metadata.get('ocr_status') in {'failed','rejected','empty','unresolved'}:
            continue
        if metadata.get('table_structure_status') in {'unresolved','unavailable','rejected','failed'}:
            continue
        rows = metadata.get('table_rows')
        if not rows and isinstance(metadata.get('table_values'), list):
            rows = [metadata['table_values']]
        if not isinstance(rows, list) or not rows:
            continue
        owner=structure_nodes.get(leaf.get('node_id'),{})
        native_owner=owner.get('kind')=='table' and owner_sources.get(owner['id'])=={source}
        table_id = 'node:'+owner['id'] if native_owner else f'layout-table:{page}:'+str(metadata.get('table_id') or source)
        model_cells = metadata.get('table_header_origin') in {'granite_first_row_unverified','surya_html_unreviewed','surya_cell_source_headers_unreviewed'}
        cell_origin = 'surya_cell_ocr' if metadata.get('table_structure_status')=='cell_aware' else 'surya_html' if metadata.get('table_header_origin')=='surya_html_unreviewed' else 'granite_table_values_source_aligned' if model_cells else 'source_table_cells'
        cell_review = 'unreviewed' if model_cells else 'source_extracted'
        if not native_owner:add(table_id, category='table', origin=cell_origin, review_status=cell_review)
        if not native_owner:edge(table_id, 'leaf:'+leaf['chunk_id'], 'HAS_EVIDENCE', origin=cell_origin)
        table_sources.setdefault(table_id, set()).add(leaf['chunk_id'])
        if identity in blocks and (identity, table_id) not in contained:
            contained.add((identity, table_id))
            edge(identity, table_id, 'CONTAINS', origin='source_table_block')
        for i, values in enumerate(rows):
            if not isinstance(values, list):
                continue
            row_index = metadata.get('table_row', i+1) if len(rows)==1 else i+1
            row_id = f'{table_id}:row:{row_index}'
            if row_id in tables:
                continue
            tables[row_id] = True
            add(row_id, category='table_row', row_index=row_index, origin=cell_origin, review_status=cell_review)
            edge(table_id, row_id, 'HAS_ROW', origin=cell_origin, review_status=cell_review)
            row_sources.append((table_id, row_id, leaf['chunk_id']))
            explicit = metadata.get('table_cells')
            cells = [c for c in explicit if c['row']==row_index] if isinstance(explicit,list) else [
                dict(column=col,text=value) for col,value in enumerate(values,1)]
            for cell in cells:
                col, value = cell['column'], cell['text']
                cell_id = f'{row_id}:cell:{col}'
                add(cell_id, category='table_cell', row_index=row_index, column_index=col,
                    bbox_json=json.dumps(cell['bbox']) if metadata.get('table_cell_bbox_scope')=='page_normalized_1000' and cell.get('bbox') else None,
                    page_number=page, image_path=image or '',
                    header=cell.get('header') if type(cell.get('header')) is bool else None,
                    rowspan=cell.get('rowspan',1),colspan=cell.get('colspan',1),
                    text=str(value or ''), origin=cell_origin, review_status=cell_review)
                edge(row_id, cell_id, 'HAS_CELL', origin=cell_origin, review_status=cell_review)
    # A one-source table already identifies every row's source. With multiple
    # source fragments retain row bindings to avoid mixing their provenance.
    for table_id, row_id, source_id in row_sources:
        if len(table_sources[table_id]) > 1:
            edge(row_id, 'leaf:'+source_id, 'HAS_EVIDENCE')
    for group in pages.values():
        native = any(b['order_source'] in {'surya_layout_model','surya_full_page_model','gliner_reading'} for b in group)
        if native:
            ranks = [b['rank'] for b in group]
            valid = all(type(rank) is int and rank >= 0 for rank in ranks)
            if (not valid or len(set(ranks)) != len(ranks)
                    or any(b['order_unresolved'] for b in group)):
                continue
            group = sorted(group,key=lambda b:b['rank'])
        # The chosen source/model stream is separate from spatial relations.
        # Ambiguous order must not produce NEXT edges.
        for a,b in zip(group, group[1:]):
            if a['order_unresolved'] or b['order_unresolved']:
                continue
            provider = a['order_source'] if a['order_source'] == b['order_source'] and a['order_source'] else 'observed_source_order'
            edge(a['key'], b['key'], 'NEXT', origin=provider,reading_stream_id=a.get('stream_id'))
    for group in spatial_pages.values():
        for relation in spatial_relations(group):
            edge(relation['from_key'], relation['to_key'], relation['kind'],
                 origin='source_bbox_geometry_not_semantics', threshold=relation.get('threshold'))




PREGRAPH_PIPELINE_SCHEMA='specific-entity-relations-with-related-passages-v5'


def validate_prepared(tree):
    proof=tree.get('pregraph_analysis',{})
    fingerprint=tree.get('source_fingerprint')
    if (proof.get('schema')!=PREGRAPH_PIPELINE_SCHEMA or proof.get('status')!='complete'
            or not fingerprint or proof.get('source_fingerprint')!=fingerprint):
        raise ValueError('Whole-document model analysis must finish before graph publication')
    for key in ('reading_analysis','identity_resolution'):
        stage=tree.get(key,{})
        if stage.get('status')!='complete' or stage.get('source_fingerprint')!=fingerprint:
            raise ValueError('Missing or stale pregraph stage: '+key)
    from .gliner_identity_classifier import identity_name_key
    grouped_names={}
    for entity in tree.get('knowledge_graph',{}).get('entities',[]):
        if entity.get('entity_type')!='ENTITY':
            raise ValueError('Graph entities must not carry semantic subtypes')
        if not entity.get('canonical_key'):
            raise ValueError('An identity-bearing mention lacks a precomputed canonical ID')
        grouped_names.setdefault(entity['canonical_key'],set()).add(identity_name_key(entity['name']))
    if any(len(names)>1 for names in grouped_names.values()):
        raise ValueError('Different entity names cannot share a compacted graph node')

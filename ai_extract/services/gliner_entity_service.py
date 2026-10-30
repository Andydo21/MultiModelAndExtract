"""Source-region NER; historical relation helpers stay outside the ingestion path."""
import gc
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from ai_extract import STRUCTURE_REVIEW_SCHEMA

MODEL_ID='knowledgator/gliner-relex-multi-v1.0'
ENTITY_SCHEMA='gliner-source-entities-only-v15-specific-names'
ENTITY_SOURCE_BATCH_SIZE=64  # GPU windows stay bounded independently inside infer_entities.
CONTEXT_SEPARATOR='\n\n[Source boundary]\n\n'
ENTITY_LABELS={
    'entity':'ENTITY',
    'organization':'ENTITY','person':'ENTITY','product':'ENTITY','location':'ENTITY',
    'component':'ENTITY','attribute':'ENTITY','value':'ENTITY',
    'organization name':'ENTITY','person name':'ENTITY','product or model name':'ENTITY',
    'location name':'ENTITY','physical component':'ENTITY','technical property':'ENTITY',
    'measurement with unit':'ENTITY','communication interface':'ENTITY',
    'full name of a person':'ENTITY','specific product or model name':'ENTITY',
    'specific named technical component or tool':'ENTITY',
    'standard or communication protocol name':'ENTITY',
}
# These are decoder search cues, not output classes. Their model labels are
# discarded in validate_mentions; neither merging nor graph storage uses them.
ENTITY_DETECTION_LABELS=['organization name','full name of a person','specific product or model name',
    'specific named technical component or tool','standard or communication protocol name',
    'generic reference or nonspecific term']


def source_layout_excluded(leaf):
    """Suppress MENTIONS only; page headers/footers remain NER and embedding sources."""
    from .document_structure_service import normalize_layout
    return normalize_layout(leaf.get('layout_type')) in {'page_header','page_footer'}


def source_text_eligible(leaf):
    """Select real source text by provenance, never by its words or name shape."""
    meta=leaf.get('structure_metadata') or {}
    return (bool(leaf.get('text','').strip())
        and meta.get('ocr_status') not in {'skipped','failed','rejected','empty','unresolved'}
        and meta.get('text_source')!='visual_display_label'
        and not meta.get('display_only')
        and meta.get('visual_ocr_status')!='failed')


def source_retrieval_eligible(leaf):
    """Page furniture stays searchable, including older trees with its legacy exclusion flag."""
    return source_text_eligible(leaf) and (not leaf.get('retrieval_excluded') or source_layout_excluded(leaf))

_runtime=None
RELATION_LABELS={'part of':'part_of','located in':'located_in','designed by':'designed_by',
    'supports':'supports','integrates with':'integrates_with'}


def relation_review_claim(rel,entities,sources,identity):
    head,tail=entities.get(rel['from_key']),entities.get(rel['to_key'])
    source=rel.get('source_id');target=rel.get('target_source_id',source)
    if (not head or not tail or source not in sources or target not in sources
            or head['source_id']!=source or tail['source_id']!=target
            or rel.get('predicate') not in RELATION_LABELS.values()):
        raise ValueError('Relation review requires grounded entity endpoints')
    return dict(id=str(identity),head=head['name'],tail=tail['name'],
        predicate=rel['predicate'].replace('_',' '),source=sources[source],
        target_source=sources[target] if target!=source else None)


def review_entity_relations(result,units,reviewer=None):
    """Keep NER intact; classify each proposed relation's source support/polarity.

    A failed or ambiguous review withholds the relation, not its entities.
    No keyword rules infer polarity or replace a model verdict.
    """
    from .gliner_identity_classifier import infer_relation_claims,RELATION_REVIEW_SCHEMA,relation_claim_verdict,source_entailment_supported,direct_relation_supported
    result=dict(result);proposals=list(result.get('relations',[]));entities={mention_key(e):e for e in result['entities']}
    sources={u['id']:u['text'] for u in units};claims=[]
    for i,rel in enumerate(proposals):
        claims.append(relation_review_claim(rel,entities,sources,i))
    result['relations']=[];result['relation_proposals']=proposals
    if not claims:
        result['relation_review']=dict(schema=RELATION_REVIEW_SCHEMA,status='empty',decisions=[])
        return result
    try:
        response=(reviewer or infer_relation_claims)(claims)
        decisions=response.get('decisions',[])
        if (response.get('schema')!=RELATION_REVIEW_SCHEMA or len(decisions)!=len(claims)
                or {d['id'] for d in decisions}!={c['id'] for c in claims}):
            raise ValueError('Incomplete entity relation review')
        accepted=[]
        for decision in decisions:
            if decision['status'] not in {'affirmed','negated','conditional','unresolved'}:
                raise ValueError('Invalid entity relation review status')
            if decision['status'] in {'affirmed','negated'}:
                verdict,_=relation_claim_verdict(decision.get('scores'))
                if verdict!=decision['status']:raise ValueError('Relation verdict contradicts its support scores')
                if not source_entailment_supported(decision.get('source_entailment'),decision['status'],claims[int(decision['id'])]):
                    raise ValueError('Relation lacks source-bound NLI entailment evidence')
                if not direct_relation_supported(decision,claims[int(decision['id'])]):
                    raise ValueError('Relation lacks source-bound direct relationship review')
                rel=dict(proposals[int(decision['id'])],polarity=decision['status'],modality='assertion',
                    review_status='model_reviewed',qualification_validation='gliner2_source_support_with_nli',
                    relation_review_json=json.dumps(decision,ensure_ascii=False))
                accepted.append(rel)
        result['relations']=accepted
        result['relation_review']=dict(response,status='complete',accepted=len(accepted),
                                       withheld=len(proposals)-len(accepted))
    except Exception as error:
        result['relation_review']=dict(schema=RELATION_REVIEW_SCHEMA,status='unavailable',decisions=[],
            accepted=0,withheld=len(proposals),error_type=type(error).__name__,reason=str(error)[:200])
    return result


def valid_score(value, threshold):
    return (isinstance(value,(int,float)) and not isinstance(value,bool)
            and math.isfinite(value) and threshold<=value<=1)


def mention_key(item):
    return hashlib.sha256(json.dumps([item['source_id'],item['start'],item['end'],item['entity_type']],ensure_ascii=False).encode()).hexdigest()


def unload():
    global _runtime
    if _runtime is None:return dict(model=MODEL_ID,loaded=False)
    _runtime=None;gc.collect()
    import torch
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return dict(model=MODEL_ID,loaded=False)


def runtime():
    global _runtime
    if _runtime is None:
        import torch
        from gliner import GLiNER
        if not torch.cuda.is_available():raise RuntimeError('GLiNER requires configured CUDA GPU')
        path=Path(os.getenv('GLINER_MODEL_PATH','scratch/a3-models/gliner-relex-multi-v1.0'))
        if not (path/'gliner_config.json').is_file() or not (path/'model.safetensors').is_file():
            raise RuntimeError('GLiNER checkpoint not installed')
        _runtime=GLiNER.from_pretrained(str(path),
            cache_dir=os.getenv('GLINER_ENCODER_CACHE','scratch/a3-models/gliner-encoder-cache'),
            local_files_only=True,map_location='cuda',strict=True,
            dtype=torch.float16,low_cpu_mem_usage=True).eval()
        print(json.dumps(dict(event='gliner_load',model=MODEL_ID,device='cuda',task='entity_extraction')),flush=True)
    return _runtime


def text_windows(text, words_per_window=160, overlap=24):
    """Keep exact character offsets; overlap protects entities at window boundaries."""
    if not 0<=overlap<words_per_window:raise ValueError('Invalid entity window overlap')
    tokens=list(re.finditer(r'\S+',text))
    start=0
    while start<len(tokens):
        stop=min(len(tokens),start+words_per_window)
        a,b=tokens[start].start(),tokens[stop-1].end()
        yield a,text[a:b]
        if stop==len(tokens):break
        start=stop-overlap


def validate_mentions(text,mentions,threshold=.65):
    """A label/score never substitutes for an exact source span."""
    accepted=[]
    for mention in mentions:
        start,end=mention.get('start'),mention.get('end')
        label=mention.get('label');score=mention.get('score')
        if (not isinstance(start,int) or isinstance(start,bool) or not isinstance(end,int) or isinstance(end,bool)
            or not 0<=start<end<=len(text) or label not in ENTITY_LABELS
            or not valid_score(score,threshold)
            or mention.get('text')!=text[start:end] or not text[start:end].strip()):
            continue
        accepted.append(dict(start=start,end=end,text=text[start:end],entity_type=ENTITY_LABELS[label],score=float(score)))
    return accepted


def flat_source_entities(items):
    """Apply GLiNER's confidence-first flat decoding across window boundaries.

    Positions are original half-open character spans. Repeated names at
    different positions/sources remain separate occurrences.
    """
    grouped={}
    for item in items:grouped.setdefault(item['source_id'],[]).append(item)
    result=[]
    for source,mentions in grouped.items():
        selected=[]
        for item in sorted(mentions,key=lambda e:(-e['confidence'],e['start'],e['end'])):
            if any(item['start']<other['end'] and other['start']<item['end'] for other in selected):continue
            selected.append(item)
        result.extend(sorted(selected,key=lambda e:(e['start'],e['end'])))
    return result


def infer_entities(units,predictor=None,*,include_relations=False):
    started=time.perf_counter();threshold=float(os.getenv('GLINER_ENTITY_THRESHOLD','.5'))
    relation_threshold=float(os.getenv('GLINER_RELATION_THRESHOLD','.8'))
    if not 0<=threshold<=1 or not 0<=relation_threshold<=1:raise ValueError('Invalid extraction threshold')
    pending=[]
    for unit in units:
        if 'context_windows' in unit:
            for window in unit['context_windows']:
                start,text=window['offset'],window['text']
                if unit['text'][start:start+len(text)]!=text:raise ValueError('Invalid target context offset')
                for context in window['contexts']:
                    if context['id']==unit['id'] or not context['text']:continue
                    boundary=len(text)+len(CONTEXT_SEPARATOR)
                    combined=text+CONTEXT_SEPARATOR+context['text']
                    spans=[(0,len(text),unit['id'],start),
                           (boundary,len(combined),context['id'],context['offset'])]
                    pending.append((unit['id'],combined,spans))
        else:
            for offset,text in text_windows(unit['text']):
                pending.append((unit['id'],text,[(0,len(text),unit['id'],offset)]))
    entities={};relations={}
    if pending:
        network=predictor or runtime()
        for index in range(0,len(pending),8):
            batch=pending[index:index+8]
            output=network.inference(texts=[r[1] for r in batch],labels=ENTITY_DETECTION_LABELS,
                relations=list(RELATION_LABELS) if include_relations else [],threshold=threshold,adjacency_threshold=.6,
                relation_threshold=relation_threshold,return_relations=include_relations,
                flat_ner=not include_relations,multi_label=False,batch_size=2)
            predictions,links=output if include_relations else (output,[[] for _ in batch])
            if len(predictions)!=len(batch) or len(links)!=len(batch):raise ValueError('GLiNER batch source coverage mismatch')
            for (target,text,spans),mentions,proposals in zip(batch,predictions,links):
                local={}
                for mention in validate_mentions(text,mentions,threshold):
                    span=next((s for s in spans if s[0]<=mention['start']<mention['end']<=s[1]),None)
                    if span is None:continue
                    low,high,source,offset=span
                    start,end=offset+mention['start']-low,offset+mention['end']-low
                    key=(source,start,end,mention['entity_type'])
                    entity=dict(source_id=source,start=start,end=end,name=mention['text'],
                                entity_type=mention['entity_type'],confidence=mention['score'],
                                origin='gliner_source_span',review_status='unreviewed')
                    if key not in entities or entity['confidence']>entities[key]['confidence']:entities[key]=entity
                    local[(mention['start'],mention['end'],mention['entity_type'])]=entity
                for link in proposals if include_relations else []:
                    score=link.get('score');label=link.get('relation')
                    if label not in RELATION_LABELS or not valid_score(score,relation_threshold):continue
                    endpoints=[]
                    for side in ('head','tail'):
                        endpoint=link.get(side,{})
                        if not isinstance(endpoint,dict):continue
                        if any(not isinstance(endpoint.get(k),int) or isinstance(endpoint[k],bool) for k in ('start','end')):continue
                        item=local.get((endpoint.get('start'),endpoint.get('end'),ENTITY_LABELS.get(endpoint.get('type'))))
                        if item and endpoint.get('text')==item['name']:endpoints.append(item)
                    if len(endpoints)!=2:continue
                    a,b=[mention_key(item) for item in endpoints]
                    if a==b:continue
                    source=endpoints[0]['source_id'];destination=endpoints[1]['source_id']
                    own=next(s for s in spans if s[2]==source)
                    other=next(s for s in spans if s[2]==destination)
                    key=(source,a,b,RELATION_LABELS[label])
                    if key not in relations or score>relations[key]['confidence']:
                        relations[key]=dict(source_id=source,from_key=a,to_key=b,predicate=RELATION_LABELS[label],
                            quote=text[own[0]:own[1]],target_source_id=destination,
                            target_quote=text[other[0]:other[1]],confidence=float(score),
                            origin='gliner_relex_source_span',review_status='unreviewed')
            print(json.dumps(dict(event='gliner_batch',windows_completed=min(index+8,len(pending)),windows_total=len(pending),
                                  entities=len(entities),relations=len(relations))),flush=True)
    candidates=len(entities)
    if not include_relations:
        entities={mention_key(e):e for e in flat_source_entities(entities.values())}
    metrics=dict(windows=len(pending),sources=len(units),entities=len(entities),relations=len(relations),
                             flat_ner=not include_relations,window_overlap_withheld=candidates-len(entities),
                             elapsed_s=round(time.perf_counter()-started,3),threshold=threshold,
                             device=str(getattr(predictor or _runtime,'device','cuda')) if pending else None)
    if predictor is None and pending:
        import torch
        metrics.update(cuda_allocated_gb=round(torch.cuda.memory_allocated()/1e9,3),
                       cuda_peak_allocated_gb=round(torch.cuda.max_memory_allocated()/1e9,3))
    return dict(model=MODEL_ID,provider='gliner',entity_schema=ENTITY_SCHEMA,
                extraction_mode='entities_and_relations' if include_relations else 'entities_only',entities=list(entities.values()),
                relations=list(relations.values()),metrics=metrics)


def _request_runtime(route,payload,timeout=600,*,expected_model=MODEL_ID):
    import requests
    from django.conf import settings
    from .contract_pipeline_service import NGROK_HEADERS
    response=requests.post(settings.RAG_INFERENCE_URL.rstrip('/')+'/rag/'+route,
                           json=payload,headers=NGROK_HEADERS,timeout=timeout)
    response.raise_for_status();result=response.json()
    if result.get('model')!=expected_model:raise ValueError('Invalid GLiNER runtime model')
    return result


def request_entities(units):
    result=_request_runtime('gliner_entities',dict(units=units),timeout=(5,None))
    if result.get('model')!=MODEL_ID or result.get('provider')!='gliner':raise ValueError('Invalid GLiNER provider')
    if result.get('entity_schema')!=ENTITY_SCHEMA:
        raise ValueError('GLiNER runtime is outdated; restart A3 to load the entity schema')
    if result.get('extraction_mode')!='entities_only' or result.get('relations')!=[]:
        raise ValueError('Source extraction must return entities only, without semantic relations')
    return result


def infer_identity_pairs(pairs, predictor=None):
    """Ask the existing multilingual RelEx model about bounded source pairs.

    Both endpoints must be literal names inside their own supplied source.
    Similar names or embeddings alone never produce an identity decision.
    """
    if (not isinstance(pairs,list) or len(pairs)>16 or any(
            not isinstance(p,dict) or not isinstance(p.get('id'),str)
            or not isinstance(p.get('left'),dict) or not isinstance(p.get('right'),dict) for p in pairs)
            or len({p['id'] for p in pairs})!=len(pairs)):
        raise ValueError('Invalid identity batch')
    texts, ranges, input_spans = [], [], []
    for pair in pairs:
        left,right=pair['left'],pair['right']
        if any(not isinstance(v.get('name'),str) or not v['name'] or not isinstance(v.get('context'),str)
               or v['name'] not in v['context'] or len(v['context'])>2500 for v in (left,right)):
            raise ValueError('Identity endpoints require their source quotations')
        text=left['context']+CONTEXT_SEPARATOR+right['context']
        texts.append(text);ranges.append((len(left['context']),len(left['context'])+len(CONTEXT_SEPARATOR)))
        a=left['context'].index(left['name'])
        b=len(left['context'])+len(CONTEXT_SEPARATOR)+right['context'].index(right['name'])
        input_spans.append([dict(start=a,end=a+len(left['name'])),dict(start=b,end=b+len(right['name']))])
    if not texts:return dict(model=MODEL_ID,decisions=[])
    detected,proposals=(predictor or runtime()).inference(texts=texts,labels=list(ENTITY_LABELS),
        relations=['same entity as','different entity from'],threshold=.5,adjacency_threshold=0.,
        input_spans=input_spans,relation_threshold=0.,return_relations=True,batch_size=2)
    if len(proposals)!=len(pairs) or len(detected)!=len(pairs):raise ValueError('Incomplete identity pair inference')
    decisions=[]
    for pair,text,links,(end,start),entities in zip(pairs,texts,proposals,ranges,detected):
        scores={'same entity as':0.,'different entity from':0.}
        grounded_links=0
        def grounded(endpoint,side):
            low,high=(0,end) if side=='left' else (start,len(text))
            a,b=endpoint.get('start'),endpoint.get('end')
            return (type(a) is int and type(b) is int and low<=a<b<=high
                and endpoint.get('text')==pair[side]['name']
                and text[a:b]==pair[side]['name'])
        detected_sides={side:any(grounded(e,side) for e in entities) for side in ('left','right')}
        for link in links:
            # Preserve actual low confidence. The acceptance threshold below is
            # unchanged; hiding scores below .5 made missing inference and low
            # confidence indistinguishable in persisted audit decisions.
            if link.get('relation') not in scores or not valid_score(link.get('score'),0.):continue
            endpoints=[link.get('head',{}),link.get('tail',{})]
            if not ((grounded(endpoints[0],'left') and grounded(endpoints[1],'right'))
                    or (grounded(endpoints[1],'left') and grounded(endpoints[0],'right'))):continue
            grounded_links+=1
            scores[link['relation']]=max(scores[link['relation']],float(link['score']))
        same,different=scores.values()
        status='same' if same>=.9 and same-different>=.1 else 'different' if different>=.9 and different-same>=.1 else 'unresolved'
        reason=('accepted_model_identity' if status=='same' else 'accepted_model_difference' if status=='different'
                else 'low_confidence' if grounded_links else 'endpoint_not_detected' if not all(detected_sides.values())
                else 'no_grounded_relation')
        decisions.append(dict(id=pair['id'],same=same,different=different,status=status,reason=reason,
            detected_endpoints=detected_sides,raw_relations=len(links),grounded_relations=grounded_links))
    return dict(model=MODEL_ID,decisions=decisions,policy='model_inference_not_verified_identity',schema='identity-diagnostics-v2')


def request_identity_pairs(pairs):
    result=_request_runtime('gliner_identity',dict(pairs=pairs))
    if len(result['decisions'])!=len(pairs) or {r['id'] for r in result['decisions']}!={p['id'] for p in pairs}:
        raise ValueError('Invalid identity runtime response')
    return result


def structure_review_policy():
    from .gliner_identity_classifier import MODEL_ID as classifier_model,REVISION
    return dict(model=classifier_model,revision=REVISION,schema=STRUCTURE_REVIEW_SCHEMA,
        representation='direct_source_pair_classification_with_direction_check',
        acceptance_threshold=.9,margin=.1,source_character_limit=1800,
        classification_max_words=4096,
        classification_labels={kind:structure_pair_labels(kind) for kind in ('NEXT','CONTAINS')})


def structure_pair_labels(kind):
    relation=('Source A should be read before Source B in the document reading flow'
              if kind=='NEXT' else 'Source B belongs under the section or content scope of Source A')
    return dict(supported=relation+'. The quoted source contents support this proposed relation.',
        disputed='The quoted source contents contradict the proposed relation: '+relation+'.',
        uncertain='The quoted contents do not establish whether this proposed relation is correct. '
            'Related entities, matching words or presentation order alone do not establish document reading order or containment.')


def infer_structure_pairs(pairs,predictor=None):
    """Classify known Surya source pairs independently of entity detection."""
    from .gliner_identity_classifier import read_scores,run_with_classifier
    if (not isinstance(pairs,list) or len(pairs)>16 or any(not isinstance(p,dict)
            or not isinstance(p.get('id'),str) or p.get('kind') not in {'NEXT','CONTAINS'}
            or any(not isinstance(p.get(side),str) or not p[side].strip() or len(p[side])>1800
                   for side in ('left','right')) for p in pairs)
            or len({p['id'] for p in pairs})!=len(pairs)):
        raise ValueError('Invalid structure review batch')
    policy=structure_review_policy()
    def classify_all(model):
        decisions=[]
        for pair in pairs:
            labels=policy['classification_labels'][pair['kind']]
            tasks={'source_relation':dict(labels=labels,multi_label=True,class_act='softmax',cls_threshold=0.,top_k=3)}
            inputs=[];raw=[]
            for swapped in (False,True):
                # Swapped endpoints are a direction-sensitivity probe, not gold.
                sources=dict(left=pair['right'],right=pair['left']) if swapped else pair
                for sides in (('left','right'),('right','left')):
                    # Within each probe, A/B remain fixed across presentations.
                    text='Proposed source relation: '+pair['kind']+'(A, B)\n\n'+'\n\n'.join(
                        ('Source A quotation: ' if side=='left' else 'Source B quotation: ')+sources[side] for side in sides)
                    inputs.append(text)
                    raw.append(model.classify_text(text,tasks,include_confidence=True,
                        max_len=policy['classification_max_words']))
            scores=[read_scores(r,task='source_relation',labels=labels) for r in raw]
            winners=[max(s,key=s.get) for s in scores]
            status='unresolved';reason='below_acceptance_threshold'
            if winners[0]!=winners[1] or winners[2]!=winners[3]:reason='presentation_disagreement'
            elif winners[0]=='uncertain':reason='model_uncertain'
            elif winners[2]!={'supported':'disputed','disputed':'supported'}[winners[0]]:
                reason='relation_direction_not_distinguished'
            elif all(s[label]>=policy['acceptance_threshold'] and
                    s[label]-max(v for k,v in s.items() if k!=label)>=policy['margin'] for s,label in zip(scores,winners)):
                status=winners[0];reason='accepted_direct_pair_classification'
            decisions.append(dict(id=pair['id'],status=status,reason=reason,
                supported=min(s['supported'] for s in scores[:2]),reverse=min(s['disputed'] for s in scores[:2]),
                scores_forward=scores[0],scores_reverse=scores[1],raw_outputs=raw,
                scores_swapped_endpoints=scores[2:],
                direction_check='model_consistency_probe_not_ground_truth',
                model_inputs=inputs,proposed_relation=pair['kind'],
                evidence=[dict(source='left',quote=pair['left']),dict(source='right',quote=pair['right'])],
                diagnostics_scope='direct_pair_classification_scores_not_verified_structure'))
        return dict(model=policy['model'],revision=policy['revision'],schema=STRUCTURE_REVIEW_SCHEMA,
            decisions=decisions,inference_policy=policy,policy='model_proposal_not_verified_structure')
    return run_with_classifier(classify_all,predictor) if pairs else classify_all(predictor)


def request_structure_pairs(pairs):
    policy=structure_review_policy()
    result=_request_runtime('gliner_structure',dict(pairs=pairs),expected_model=policy['model'])
    if (result.get('schema')!=STRUCTURE_REVIEW_SCHEMA
            or result.get('model')!=policy['model'] or result.get('revision')!=policy['revision']
            or result.get('inference_policy')!=policy or len(result['decisions'])!=len(pairs)
            or {r['id'] for r in result['decisions']}!={p['id'] for p in pairs}):
        raise ValueError('Invalid structure review response')
    return result


def document_context_units(tree,embedding=None):
    """Retrieve candidate windows; retrieval scores never create graph relations."""
    import numpy as np
    from .contract_pipeline_service import HybridBackend
    leaves=[l for l in tree['leaves'] if source_text_eligible(l)
        and l.get('layout_type') not in {'page_header','page_footer'}]
    windows=[];by_source={}
    for leaf in leaves:
        for offset,text in text_windows(leaf['text'],64,16):
            item=dict(chunk_id=str(len(windows)),id=leaf['chunk_id'],offset=offset,text=text)
            windows.append(item);by_source.setdefault(leaf['chunk_id'],[]).append(item)
    dense=None;error=None
    if len(by_source)>1:
        try:
            unique=list(dict.fromkeys(w['text'] for w in windows))
            vectors=(embedding or HybridBackend().embedding)(unique)
            lookup=dict(zip(unique,vectors))
            dense=np.asarray([lookup[w['text']] for w in windows],dtype=np.float32)
            norms=np.linalg.norm(dense,axis=1,keepdims=True)
            if dense.ndim!=2 or len(vectors)!=len(unique) or not np.isfinite(dense).all() or (norms==0).any():
                raise ValueError('Invalid document candidate embeddings')
            dense=dense/norms
        except Exception as exc:
            raise RuntimeError('Document neural candidate model unavailable; no keyword fallback') from exc
    limit=max(1,int(os.getenv('GLINER_DOCUMENT_CANDIDATES','16')))
    units={l['chunk_id']:dict(id=l['chunk_id'],text=l['text'],context_windows=[]) for l in leaves}
    pair_count=0;selected_pairs=[];leaf_map={l['chunk_id']:l for l in leaves}
    for index,window in enumerate(windows):
        identity=window['id']
        candidates=[w for w in windows if w['id']!=identity]
        scores=dense@dense[index] if dense is not None else []
        selected=[];seen_sources=set()
        for candidate in sorted(candidates,key=lambda w:float(scores[int(w['chunk_id'])]),reverse=True):
            if candidate['id'] in seen_sources:continue
            selected.append(candidate);seen_sources.add(candidate['id'])
            if len(selected)>=limit:break
        contexts=list({w['chunk_id']:dict(id=w['id'],offset=w['offset'],text=w['text']) for w in selected}.values())
        if contexts:
            units[identity]['context_windows'].append(dict(offset=window['offset'],text=window['text'],contexts=contexts))
            pair_count+=len(contexts)
            selected_pairs.extend(dict(from_id=identity,from_offset=window['offset'],from_text=window['text'],
                to_id=c['id'],to_offset=c['offset'],to_text=c['text'],rank=rank,
                from_document_position=leaf_map[identity].get('reading_document_position'),
                to_document_position=leaf_map[c['id']].get('reading_document_position'))
                for rank,c in enumerate(contexts,1))
    return [u for u in units.values() if u['context_windows']],dict(
        state='dense_model' if dense is not None else 'empty',
        dense_error=error,source_count=len(leaves),windows=len(windows),candidate_pairs=pair_count,
        candidate_limit=limit,candidate_diversity='one_best_window_per_source',
        selected_pairs=selected_pairs,
        semantics='candidates_only_not_relationships')


REFERENCE_SCHEMA='gliner2-layout-competing-targets-v3'
REFERENCE_DESCRIPTIONS={
    'references_figure':'The head explicitly refers to the figure identified by the tail.',
    'references_table':'The head explicitly refers to the table or previously stated values in the tail.',
    'references_previous_instruction':'The head refers back to the earlier method or instruction stated in the tail.',
}


def reference_policy():
    from .gliner_identity_classifier import MODEL_ID as model,REVISION
    threshold=float(os.getenv('GLINER2_REFERENCE_SPAN_THRESHOLD','.5'))
    if not 0<=threshold<=1:raise ValueError('Invalid reference span threshold')
    return dict(model=model,revision=REVISION,schema=REFERENCE_SCHEMA,
        relation_descriptions=REFERENCE_DESCRIPTIONS,endpoint_span_threshold=threshold,
        confidence_semantics='head_and_tail_span_confidences; no_edge_probability',
        source_order='preserve_known_Surya_order',max_words=2048,
        task='layout_candidate_competition_then_grounded_span_proposals',
        target_threshold=.9,target_margin=.1,target_views=2,
        publication='withheld_pending_reference_validation')


def reference_context_pairs(tree,search):
    """Build bounded, complete target groups from original source layout.

    Neighbours are candidates, not references. All excerpts remain literal
    substrings, with original offsets; selection and relation extraction follow.
    """
    leaves={l['chunk_id']:l for l in tree['leaves'] if source_text_eligible(l)
            and l.get('layout_type') not in {'page_header','page_footer'}}
    limit=int(os.getenv('GLINER_REFERENCE_GROUP_CANDIDATES','8'))
    if not 2<=limit<=16:raise ValueError('Reference group size must be 2-16')
    grouped={}
    for selected in search['selected_pairs']:
        for prefix in ('from','to'):
            identity,offset=selected[prefix+'_id'],selected[prefix+'_offset']
            if identity not in leaves or type(offset) is not int or not 0<=offset<len(leaves[identity]['text']):
                raise ValueError('Invalid retrieved reference source offset')
            if prefix+'_text' in selected:
                quote=selected[prefix+'_text']
                if leaves[identity]['text'][offset:offset+len(quote)]!=quote:
                    raise ValueError('Retrieved reference quote does not match original source')
        grouped.setdefault((selected['from_id'],selected['from_offset']),[]).append(selected)
    output=[];omitted=[]
    def excerpt(identity,offset):
        text=leaves[identity]['text']
        if len(text)<=6000 and len(re.findall(r'\S+',text))<=128:return 0,text
        end=offset
        for token in list(re.finditer(r'\S+',text[offset:]))[:128]:
            if token.end()>6000:break
            end=offset+token.end()
        if end==offset:raise ValueError('Reference excerpt cannot fit its source token')
        return offset,text[offset:end]
    def block(identity,offset,text,pair_id=None):
        leaf=leaves[identity]
        return dict(id=identity,offset=offset,text=text,pair_id=pair_id,
            source_length=len(leaf['text']),excerpt_complete=offset==0 and len(text)==len(leaf['text']),
            page=leaf.get('page_number'),position=leaf.get('reading_document_position'),
            layout_type=leaf.get('layout_type'),heading_path=leaf.get('heading_path',[]),
            bbox=leaf.get('bbox'),reading_stream_id=leaf.get('reading_stream_id'))
    for (source_id,source_offset),retrieved in grouped.items():
        source=leaves[source_id];position=source.get('reading_document_position')
        # Reserve room for layout neighbours as well as semantic retrieval.
        neighbours=[];structural=[]
        if type(position) is int:
            ordered=[l for l in leaves.values() if l['chunk_id']!=source_id
                     and type(l.get('reading_document_position')) is int]
            before=sorted((l for l in ordered if l['reading_document_position']<position),
                          key=lambda l:l['reading_document_position'],reverse=True)[:2]
            after=sorted((l for l in ordered if l['reading_document_position']>position),
                         key=lambda l:l['reading_document_position'])[:2]
            neighbours=[l['chunk_id'] for l in before+after]
            for kind in ('table','figure'):
                preceding=sorted((l for l in ordered if l.get('layout_type')==kind
                                  and l['reading_document_position']<position),
                                 key=lambda l:l['reading_document_position'],reverse=True)
                if preceding:structural.append(preceding[0]['chunk_id'])
        targets={}
        for selected in retrieved[:max(1,limit-6)]:targets.setdefault(selected['to_id'],selected['to_offset'])
        for identity in structural+neighbours:
            if len(targets)<limit:targets.setdefault(identity,0)
        for selected in retrieved:
            if len(targets)<limit:targets.setdefault(selected['to_id'],selected['to_offset'])
        start,text=excerpt(source_id,source_offset)
        group=dict(id=hashlib.sha256(json.dumps([source_id,source_offset]).encode()).hexdigest(),
                   source=block(source_id,start,text),candidates=[])
        rows=[]
        for target_id,target_offset in targets.items():
            target_start,target_text=excerpt(target_id,target_offset)
            pair_id=str(len(output)+len(rows));target=leaves[target_id]
            group['candidates'].append(block(target_id,target_start,target_text,pair_id))
            target_position=target.get('reading_document_position')
            rows.append(dict(id=pair_id,left=text,right=target_text,
                target_first=type(position) is int and type(target_position) is int and target_position<position,
                context_group=group,from_id=source_id,from_offset=start,from_text=text,
                to_id=target_id,to_offset=target_start,to_text=target_text,
                from_document_position=position,to_document_position=target_position))
        omitted.append(dict(group_id=group['id'],omitted_retrieval_targets=[s['to_id'] for s in retrieved if s['to_id'] not in targets]))
        output.extend(rows)
    return output,omitted


def reference_target_choice(votes,candidate_ids):
    """Require two candidate-ID assignments to agree, including abstention."""
    policy=reference_policy();expected=set(candidate_ids)|{'no_reference'}
    if not isinstance(votes,list) or len(votes)!=2:raise ValueError('Missing reference target votes')
    winners=[]
    for scores in votes:
        if not isinstance(scores,dict) or set(scores)!=expected or not all(valid_score(v,0.) for v in scores.values()):
            raise ValueError('Invalid reference target scores')
        winner=max(scores,key=scores.get);winners.append(winner)
    if winners[0]!=winners[1]:return None,'target_assignment_disagreement'
    winner=winners[0]
    if winner=='no_reference':return None,'model_no_reference'
    if not all(s[winner]>=policy['target_threshold'] and
               s[winner]-max(v for k,v in s.items() if k!=winner)>=policy['target_margin'] for s in votes):
        return None,'insufficient_target_confidence'
    return winner,'target_selected'


def infer_reference_group(pairs,model):
    """Compete targets using layout, then require literal relation endpoints."""
    from .gliner_identity_classifier import read_scores
    group=pairs[0]['context_group'];source=group['source'];candidates=group['candidates']
    if (any(p['context_group']!=group or p['left']!=source['text'] for p in pairs)
            or len({c['id'] for c in candidates})!=len(candidates)
            or {c['pair_id'] for c in candidates}!={p['id'] for p in pairs}
            or len(candidates)>16):raise ValueError('Incomplete reference target group')
    by_pair={p['id']:p for p in pairs}
    if any(not isinstance(b.get('id'),str) or type(b.get('offset')) is not int or b['offset']<0
           or not isinstance(b.get('text'),str) or not b['text'].strip() or len(b['text'])>6000
           for b in [source]+candidates):raise ValueError('Invalid reference layout block')
    for candidate in candidates:
        pair=by_pair[candidate['pair_id']]
        source_position,target_position=source.get('position'),candidate.get('position')
        target_first=type(source_position) is int and type(target_position) is int and target_position<source_position
        if (candidate['id']==source['id'] or candidate['text']!=pair['right']
                or pair.get('target_first',False)!=target_first):
            raise ValueError('Reference target differs from its quoted group source')
    policy=reference_policy();votes=[];raw_votes=[];texts=[];ranges=[]
    for view in (candidates,list(reversed(candidates))):
        labels={f'candidate_{i}':c for i,c in enumerate(view)}
        definitions={k:'The explicit reference in the referring source points to this candidate: '+c['text'][:160]
                     for k,c in labels.items()}
        definitions['no_reference']='No candidate is the identifiable target, or the source contains no explicit reference.'
        names={c['pair_id']:k for k,c in labels.items()}
        blocks=[(source,'referring_source')]+[(c,names[c['pair_id']]) for c in candidates]
        # Unknown positions are shown as unknown, never promoted to source order.
        blocks.sort(key=lambda item:(type(item[0].get('position')) is not int,
                                    item[0].get('position') if type(item[0].get('position')) is int else 0))
        text='Select the target of the referring source. Quoted blocks and layout metadata follow. Use no_reference when the target cannot be established.\n\n'
        span_ranges={}
        for b,name in blocks:
            metadata={k:b.get(k) for k in ('id','page','position','layout_type','heading_path','bbox','reading_stream_id','excerpt_complete')}
            text+=name+' '+json.dumps(metadata,ensure_ascii=False)+'\n'
            start=len(text);text+=b['text'];span_ranges[name]=(start,len(text));text+='\n\n'
        if len(re.findall(r'\S+',text+' '+json.dumps(definitions)))>1800:
            return [dict(id=p['id'],status='unresolved',reason='group_context_budget_exceeded',references=[],
                         rejected_instances=[],relation_confidence=None) for p in pairs]
        raw=model.classify_text(text,{'reference_target':dict(labels=definitions,multi_label=True,
            class_act='softmax',cls_threshold=0.,top_k=len(definitions))},include_confidence=True,max_len=policy['max_words'])
        scores=read_scores(raw,task='reference_target',labels=definitions)
        votes.append({labels[k]['pair_id'] if k in labels else 'no_reference':v for k,v in scores.items()})
        raw_votes.append(raw);texts.append(text);ranges.append(span_ranges)
    selected,reason=reference_target_choice(votes,by_pair)
    audit=dict(group_id=group['id'],votes=votes,selected_pair_id=selected,reason=reason,raw_outputs=raw_votes,
               confidence_semantics='uncalibrated_target_choice_not_edge_probability')
    raw={};normalized={};rejected=[];references=[]
    if selected is not None:
        schema=model.create_schema().relations({name:dict(description=description,
            threshold=policy['endpoint_span_threshold']) for name,description in REFERENCE_DESCRIPTIONS.items()})
        raw=model.extract(texts[0],schema,threshold=policy['endpoint_span_threshold'],
            include_confidence=True,include_spans=True,max_len=policy['max_words'])
        normalized=normalize_described_relations(raw,REFERENCE_DESCRIPTIONS)
        source_range=ranges[0]['referring_source']
        target_range=ranges[0][f'candidate_{next(i for i,c in enumerate(candidates) if c["pair_id"]==selected)}']
        for kind,instances in normalized.items():
            for row in instances:
                head,tail=row.get('head',{}),row.get('tail',{})
                grounded=True
                for endpoint,(low,high) in zip((head,tail),(source_range,target_range)):
                    start,end=endpoint.get('start'),endpoint.get('end')
                    if (type(start) is not int or type(end) is not int or not low<=start<end<=high
                            or endpoint.get('text')!=texts[0][start:end]
                            or not valid_score(endpoint.get('confidence'),policy['endpoint_span_threshold'])):
                        grounded=False;break
                if not grounded or (kind=='references_previous_instruction' and not by_pair[selected]['target_first']):
                    rejected.append(dict(type=kind,instance=row,reason='endpoint_or_target_direction_not_grounded'));continue
                references.append(dict(type=kind,evidence=dict(
                    source_span=dict(start=head['start']-source_range[0],end=head['end']-source_range[0],text=head['text']),
                    target_span=dict(start=tail['start']-target_range[0],end=tail['end']-target_range[0],text=tail['text'])),
                    endpoint_confidences=dict(head=float(head['confidence']),tail=float(tail['confidence'])),relation_confidence=None))
    return [dict(id=p['id'],status='reference' if p['id']==selected and references else 'unresolved',
        reason=('grounded_competing_target_proposal' if references else 'target_selected_without_grounded_relation')
               if p['id']==selected else reason if selected is None else 'other_candidate_selected',
        references=references if p['id']==selected else [],target_selection=audit,
        raw_output=raw if p['id']==selected else {},normalized_relations=normalized if p['id']==selected else {},
        rejected_instances=rejected if p['id']==selected else [],relation_confidence=None) for p in pairs]


def normalize_described_relations(raw,descriptions):
    """Handle canonical and name:description keys in the installed GLiNER2.
    
    Only known schema aliases are normalized. No new relation or score is made.
    The untouched raw response is separately retained in the audit output.
    """
    if not isinstance(raw,dict) or not isinstance(raw.get('relation_extraction'),dict):
        raise ValueError('Missing GLiNER2 relation extraction output')
    result={name:[] for name in descriptions};seen={name:set() for name in descriptions}
    for name,description in descriptions.items():
        for container in (raw,raw['relation_extraction']):
            for key in (name,name+': '+description):
                if key not in container:continue
                rows=container[key]
                if not isinstance(rows,list):raise ValueError('Invalid described relation instances')
                for row in rows:
                    if not isinstance(row,dict):raise ValueError('Reference requires endpoint spans')
                    identity=json.dumps(row,sort_keys=True,ensure_ascii=False)
                    if identity not in seen[name]:
                        seen[name].add(identity);result[name].append(row)
    return result


def infer_reference_pairs(pairs,predictor=None):
    """Extract described reference arguments with the existing GLiNER2 checkpoint."""
    from .gliner_identity_classifier import run_with_classifier
    if (not isinstance(pairs,list) or len(pairs)>128 or any(not isinstance(p,dict)
            or not isinstance(p.get('id'),str)
            or any(not isinstance(p.get(k),str) or not p[k].strip() or len(p[k])>6000
                   for k in ('left','right'))
            or type(p.get('target_first',False)) is not bool for p in pairs)
            or len({p['id'] for p in pairs})!=len(pairs)):
        raise ValueError('Invalid reference source pairs')
    policy=reference_policy()
    def extract_all(model):
        if any('context_group' in p for p in pairs):
            groups={}
            for pair in pairs:
                group=pair.get('context_group')
                if not isinstance(group,dict) or not isinstance(group.get('id'),str):
                    raise ValueError('Mixed grouped and independent reference inputs')
                groups.setdefault(group['id'],[]).append(pair)
            decisions=[d for group_pairs in groups.values() for d in infer_reference_group(group_pairs,model)]
            return dict(model=policy['model'],revision=policy['revision'],schema=REFERENCE_SCHEMA,
                inference_policy=policy,decisions=decisions,policy='competing_targets_and_spans_not_verified_graph_edges')
        schema=model.create_schema().relations({name:dict(description=description,
            threshold=policy['endpoint_span_threshold']) for name,description in REFERENCE_DESCRIPTIONS.items()})
        decisions=[]
        for pair in pairs:
            if pair.get('target_first'):
                text=pair['right']+'\n\n'+pair['left']
                source_range=(len(pair['right'])+2,len(text));target_range=(0,len(pair['right']))
            else:
                text=pair['left']+'\n\n'+pair['right']
                source_range=(0,len(pair['left']));target_range=(len(pair['left'])+2,len(text))
            raw=model.extract(text,schema,threshold=policy['endpoint_span_threshold'],
                include_confidence=True,include_spans=True,max_len=policy['max_words'])
            normalized=normalize_described_relations(raw,REFERENCE_DESCRIPTIONS)
            references=[];rejected=[]
            for kind,rows in normalized.items():
                for row in rows:
                    endpoints=[row.get('head',{}),row.get('tail',{})]
                    grounded=True
                    for endpoint,(low,high) in zip(endpoints,(source_range,target_range)):
                        start,end=endpoint.get('start'),endpoint.get('end')
                        if (type(start) is not int or type(end) is not int or not low<=start<end<=high
                                or endpoint.get('text')!=text[start:end]
                                or not valid_score(endpoint.get('confidence'),policy['endpoint_span_threshold'])):
                            grounded=False;break
                    if not grounded:
                        rejected.append(dict(type=kind,instance=row,reason='endpoint_not_in_original_source_or_below_span_threshold'))
                        continue
                    if kind=='references_previous_instruction' and not pair.get('target_first'):
                        rejected.append(dict(type=kind,instance=row,reason='previous_instruction_requires_known_earlier_target'))
                        continue
                    head,tail=endpoints
                    references.append(dict(type=kind,evidence=dict(
                        source_span=dict(start=head['start']-source_range[0],end=head['end']-source_range[0],text=head['text']),
                        target_span=dict(start=tail['start']-target_range[0],end=tail['end']-target_range[0],text=tail['text'])),
                        endpoint_confidences=dict(head=float(head['confidence']),tail=float(tail['confidence'])),
                        relation_confidence=None))
            decisions.append(dict(id=pair['id'],status='reference' if references else 'unresolved',
                reason='grounded_described_relation_proposal' if references else 'unqualified_model_output' if rejected else 'no_described_relation_returned',
                references=references,rejected_instances=rejected,raw_output=raw,
                normalized_relations=normalized,relation_confidence=None,
                presentation_order='target_first' if pair.get('target_first') else 'source_first'))
        return dict(model=policy['model'],revision=policy['revision'],schema=REFERENCE_SCHEMA,
            inference_policy=policy,decisions=decisions,policy='described_pretrained_relations_not_verified_graph_edges')
    return run_with_classifier(extract_all,predictor) if pairs else dict(model=policy['model'],
        revision=policy['revision'],schema=REFERENCE_SCHEMA,inference_policy=policy,decisions=[])


def request_reference_pairs(pairs):
    policy=reference_policy()
    result=_request_runtime('gliner2_references',dict(pairs=pairs),
        expected_model=policy['model'],timeout=(5,600))
    if (result.get('schema')!=REFERENCE_SCHEMA or result.get('revision')!=policy['revision']
            or result.get('inference_policy')!=policy or len(result.get('decisions',[]))!=len(pairs)
            or {d['id'] for d in result['decisions']}!={p['id'] for p in pairs}):
        raise ValueError('Invalid described reference response')
    return result


def enrich_document_relations(tree,knowledge,request_model=None,embedding=None):
    """Link source regions directly, without an entity-detection prerequisite."""
    units,search=document_context_units(tree,embedding)
    originals={l['chunk_id']:l['text'] for l in tree['leaves'] if source_text_eligible(l)}
    context_pairs,omitted=reference_context_pairs(tree,search)
    search=dict(search,retrieval_candidate_pairs=search['candidate_pairs'],
        selected_pairs=context_pairs,candidate_pairs=len(context_pairs),
        group_candidate_policy='dense_and_layout_neighbours_with_previous_table_or_figure',
        group_candidate_limit=int(os.getenv('GLINER_REFERENCE_GROUP_CANDIDATES','8')),omitted_targets=omitted)
    pairs=[];groups={}
    for selected in search['selected_pairs']:
        for prefix in ('from','to'):
            text=selected[prefix+'_text'];offset=selected[prefix+'_offset']
            if originals[selected[prefix+'_id']][offset:offset+len(text)]!=text:
                raise ValueError('Reference context does not match source')
        pair={k:selected[k] for k in ('id','left','right','target_first','context_group')}
        pairs.append(pair);groups.setdefault(pair['context_group']['id'],[]).append(pair)
    links={(r['from_id'],r['to_id'],r.get('from_char_start'),r.get('to_char_start')):r
           for r in knowledge.get('source_relations',[])}
    decisions=[];errors=[];proposals=[]
    batches=[];batch=[]
    for group in groups.values():
        if len(batch)+len(group)>128:batches.append(batch);batch=[]
        batch.extend(group)
    if batch:batches.append(batch)
    offset=0
    for batch in batches:
        try:
            response=(request_model or request_reference_pairs)(batch)
            if (response.get('schema')!=REFERENCE_SCHEMA or len(response['decisions'])!=len(batch)
                    or {d['id'] for d in response['decisions']}!={p['id'] for p in batch}):
                raise ValueError('Reference decision coverage mismatch')
            for decision in response['decisions']:
                p=search['selected_pairs'][int(decision['id'])]
                if decision['status'] not in {'reference','unresolved'}:raise ValueError('Invalid reference decision')
                instances=decision.get('references',[])
                if not isinstance(instances,list) or (decision['status']=='reference')!=bool(instances):
                    raise ValueError('Invalid described reference instances')
                if instances:
                    group=p['context_group'];audit=decision.get('target_selection') or {}
                    target,reason=reference_target_choice(audit.get('votes'),[c['pair_id'] for c in group['candidates']])
                    if target!=decision['id'] or audit.get('group_id')!=group['id'] or audit.get('selected_pair_id')!=target:
                        raise ValueError('Reference proposal lacks a consistent competing target selection')
                for instance in instances:
                    if instance.get('type') not in REFERENCE_DESCRIPTIONS or instance.get('relation_confidence') is not None:
                        raise ValueError('Invalid relation type or invented edge confidence')
                    if instance['type']=='references_previous_instruction':
                        source_position,target_position=p.get('from_document_position'),p.get('to_document_position')
                        if (type(source_position) is not int or type(target_position) is not int
                                or target_position>=source_position):
                            raise ValueError('Previous instruction target must be known to precede its source')
                    confidence=instance.get('endpoint_confidences',{})
                    if not all(valid_score(confidence.get(side),reference_policy()['endpoint_span_threshold']) for side in ('head','tail')):
                        raise ValueError('Unqualified reference endpoint confidence')
                    evidence=instance.get('evidence') or {}
                    for side,prefix in (('source_span','from'),('target_span','to')):
                        span=evidence.get(side,{})
                        start,end=span.get('start'),span.get('end');text=p[prefix+'_text']
                        if (type(start) is not int or type(end) is not int or not 0<=start<end<=len(text)
                                or span.get('text')!=text[start:end]):
                            raise ValueError('Reference endpoint is not grounded in its source quotation')
                    link=dict(from_id=p['from_id'],to_id=p['to_id'],kind='references',reference_type=instance['type'],
                        from_quote=p['from_text'],to_quote=p['to_text'],relation_confidence=None,
                        endpoint_confidences=confidence,score_semantics='endpoint_spans_only',
                        from_char_start=p['from_offset']+evidence['source_span']['start'],
                        from_char_end=p['from_offset']+evidence['source_span']['end'],
                        to_char_start=p['to_offset']+evidence['target_span']['start'],
                        to_char_end=p['to_offset']+evidence['target_span']['end'],
                        origin='gliner2_described_relation',review_status='unreviewed')
                    # Preserve the existing publication guard. Model output is
                    # an auditable proposal, not an automatically verified edge.
                    proposals.append(link)
                decisions.append(decision)
            offset+=len(batch)
        except Exception as error:
            errors.append(dict(offset=offset,error_type=type(error).__name__,reason=str(error)[:200]));break
    status='partial' if errors else knowledge.get('status','complete')
    proposed_pairs=sum(d['status']=='reference' for d in decisions)
    return dict(knowledge,status=status,source_relations=list(links.values()),document_relations=dict(
        schema=REFERENCE_SCHEMA,inference_policy=reference_policy(),
        status='partial' if errors else 'complete' if pairs else 'empty',
        candidate_search=search,errors=errors,metrics=dict(pairs=len(pairs),completed=len(decisions),
            proposed_pairs=proposed_pairs,unresolved_pairs=len(decisions)-proposed_pairs,
            missing_pairs=len(pairs)-len(decisions)),
        quality_status='unavailable' if errors else 'unreviewed_proposals' if proposals else 'unresolved' if pairs else 'empty',
        accepted_reference_relations=0,reference_proposals=proposals,decisions=decisions,
        reference_publication='withheld_unvalidated_source_pair_model',
        scope='direct_source_references_without_creating_entities'))


def extract_graph_entities(tree,request_model=None,context_units=None):
    if len({l['chunk_id'] for l in tree['leaves']})!=len(tree['leaves']):
        raise ValueError('Duplicate entity source IDs')
    readable=[l for l in tree['leaves'] if l.get('text','').strip()]
    excluded_layout={l['chunk_id'] for l in readable if source_layout_excluded(l)}
    excluded={l['chunk_id'] for l in readable if not source_text_eligible(l)}
    originals={l['chunk_id']:l['text'] for l in readable if l['chunk_id'] not in excluded}
    units=context_units if context_units is not None else [dict(id=i,text=text) for i,text in originals.items()]
    if (len({u['id'] for u in units})!=len(units)
            or any(u['id'] not in originals or u['text']!=originals[u['id']] for u in units)):
        raise ValueError('Entity units must uniquely bind original source text')
    entities={};errors=[];metrics=[];completed=[]
    for index in range(0,len(units),ENTITY_SOURCE_BATCH_SIZE):
        batch=units[index:index+ENTITY_SOURCE_BATCH_SIZE];sources={u['id']:u['text'] for u in batch};pairs=[]
        for unit in batch:
            for window in unit.get('context_windows',[]):
                if originals.get(unit['id'],'')[window['offset']:window['offset']+len(window['text'])]!=window['text']:
                    raise ValueError('Target context does not match source')
                for context in window['contexts']:
                    text=originals.get(context['id'],'')
                    if text[context['offset']:context['offset']+len(context['text'])]!=context['text']:
                        raise ValueError('Context does not match source')
                    sources[context['id']]=text
                    pairs.append((unit['id'],window['offset'],window['offset']+len(window['text']),
                                  context['id'],context['offset'],context['offset']+len(context['text'])))
        try:
            result=(request_model or request_entities)(batch)
            if result.get('relations') or result.get('relation_proposals'):
                raise ValueError('NER-only extraction cannot accept semantic relation output')
            for item in result.get('entities',[]):
                source=item.get('source_id');start,end=item.get('start'),item.get('end')
                if (source not in sources or not isinstance(start,int) or isinstance(start,bool) or not isinstance(end,int) or isinstance(end,bool)
                    or not 0<=start<end<=len(sources[source]) or item.get('name')!=sources[source][start:end]
                    or item.get('entity_type') not in ENTITY_LABELS.values()
                    or not valid_score(item.get('confidence'),float(os.getenv('GLINER_ENTITY_THRESHOLD','.5')))):
                    raise ValueError('Ungrounded entity mention')
                if pairs and not any((source==a and lo<=start<end<=hi) or (source==b and low<=start<end<=high)
                                     for a,lo,hi,b,low,high in pairs):
                    raise ValueError('Entity lies outside supplied context windows')
            # The runtime must return a complete flat source response. Reject
            # duplicate/overlapping proposals rather than hiding them here.
            by_source={}
            for item in result.get('entities',[]):
                by_source.setdefault(item['source_id'],[]).append(item)
            for mentions in by_source.values():
                ordered=sorted(mentions,key=lambda e:(e['start'],e['end']))
                if any(a['end']>b['start'] for a,b in zip(ordered,ordered[1:])):
                    raise ValueError('Duplicate or overlapping entity occurrences in flat NER response')
            # Validate the whole batch before accepting any proposal.
            for item in result.get('entities',[]):
                key=mention_key(item)
                entities[key]=dict(item,key=key,source_ids=[item['source_id']],origin='gliner_source_span',review_status='unreviewed')
            metrics.append(result.get('metrics',{}))
            completed.extend(u['id'] for u in batch)
        except Exception as error:
            errors.append(dict(source_ids=list(sources),error_type=type(error).__name__,reason=str(error)[:200]))
            # An unavailable runtime must not cause dozens of identical failing calls.
            if index==0 and not entities:break
    mentioned={e['source_id'] for e in entities.values()}
    coverage=dict(eligible_ids=[u['id'] for u in units],completed_ids=completed,
                  without_entity_ids=[i for i in completed if i not in mentioned],
                  excluded_ocr_ids=sorted(excluded),excluded_layout_ids=[],mention_excluded_ids=sorted(excluded_layout))
    print(json.dumps(dict(event='gliner_coverage',eligible=len(units),completed=len(completed),without_entity=len(coverage['without_entity_ids']),
        excluded_ocr=len(excluded),mention_excluded=len(excluded_layout))),flush=True)
    return dict(model=MODEL_ID,provider='gliner',schema=ENTITY_SCHEMA,extraction_mode='entities_only',status='partial' if errors and completed else 'unavailable' if errors else 'complete' if units else 'empty',
                entities=list(entities.values()),relations=[],source_relations=[],errors=errors,metrics=metrics,
                relation_proposals=[],relation_reviews=[],
                source_count=len(units),source_coverage=coverage,validation_scope='source_region_spans_and_allowlist_not_semantic_accuracy')

"""Diagnostic variants on real source pairs; no gold or graph modifications."""
import json
import os
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME='D:/hf_cache',
                  GLINER_MODEL_PATH='scratch/a3-models/gliner-relex-multi-v1.0')
import torch
from ai_extract.services.gliner_entity_service import runtime,ENTITY_LABELS,CONTEXT_SEPARATOR

directory=Path('evaluation/benchmarks/corpus20')
pairs=json.loads((directory/'gliner_identity_source_pairs_v3.json').read_text(encoding='utf-8'))
output=directory/'gliner_identity_variants_v3.json'
report=dict(policy='Diagnostic model outputs only; gold labels absent; not accuracy measurements',results=[])
model=runtime()
variants=['baseline','other_label','typed_label','typed_synonyms']
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
    for variant in variants:
        for pair in pairs:
            kind=pair['entity_type'].lower()
            labels=list(ENTITY_LABELS) if variant=='baseline' else ['other'] if variant=='other_label' else [kind,'other']
            names=['same entity as','different entity from'] if variant!='typed_synonyms' else [f'same {kind} as',f'different {kind} from','alias of','identical to']
            contexts=[];spans=[]
            for side in ('left','right'):
                source=pair[side];position=source['context'].index(source['name'])
                begin=max(0,position-160);end=min(len(source['context']),position+len(source['name'])+160)
                contexts.append(source['context'] if variant=='baseline' else source['context'][begin:end])
            text=CONTEXT_SEPARATOR.join(contexts)
            for i,side in enumerate(('left','right')):
                offset=0 if i==0 else len(contexts[0])+len(CONTEXT_SEPARATOR)
                start=offset+contexts[i].index(pair[side]['name'])
                spans.append(dict(start=start,end=start+len(pair[side]['name'])))
            entities,relations=model.inference(texts=[text],labels=labels,relations=names,
                threshold=.5,adjacency_threshold=0.,relation_threshold=0.,input_spans=[spans],
                return_relations=True,batch_size=1)
            def exact(endpoint,index):
                return all(endpoint.get(k)==spans[index][k] for k in ('start','end')) and endpoint.get('text')==pair[('left','right')[index]]['name']
            grounded=[r for r in relations[0] if (exact(r['head'],0) and exact(r['tail'],1)) or (exact(r['head'],1) and exact(r['tail'],0))]
            scores={name:max([float(r['score']) for r in grounded if r['relation']==name] or [0.]) for name in names}
            result=dict(variant=variant,pair_id=pair['id'],run_id=pair['run_id'],labels=labels,
                relation_labels=names,contexts=contexts,input_spans=spans,entities=entities[0],
                raw_relations=relations[0],grounded_scores=scores,
                endpoints_detected=[any(exact(e,i) for e in entities[0]) for i in (0,1)])
            report['results'].append(result)
            output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        rows=[r for r in report['results'] if r['variant']==variant]
        print(json.dumps(dict(variant=variant,pairs=len(rows),both_endpoints=sum(all(r['endpoints_detected']) for r in rows),
            maximum_grounded_score=max(max(r['grounded_scores'].values()) for r in rows))),flush=True)

"""Run GLiNER2 classification on actual exported pairs without graph writes."""
import json
import os
from pathlib import Path
import time
import argparse
import hashlib
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME='D:/hf_cache')
import torch
from gliner2 import GLiNER2
from ai_extract.services.gliner_identity_classifier import classify_pair,MODEL_ID,REVISION

directory=Path('evaluation/benchmarks/corpus20')
parser=argparse.ArgumentParser()
parser.add_argument('--groups',action='store_true')
options=parser.parse_args()
source=directory/('entity_merge_groups.json' if options.groups else 'gliner_identity_source_pairs_v3.json')
loaded=json.loads(source.read_text(encoding='utf-8'))
pairs=loaded['groups'] if options.groups else loaded
started=time.monotonic()
model=GLiNER2.from_pretrained('scratch/a3-models/gliner2-base-v1',map_location='cuda',quantize=True,local_files_only=True).eval()
report=dict(model=MODEL_ID,revision=REVISION,device='cuda',dtype='float16',
    source_pairs=source.name,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),gold_labels_available=False,
    policy='Shadow classification only. No accuracy claim or graph mutation.',results=[])
with torch.inference_mode():
    for pair in pairs:
        before=time.monotonic()
        result=classify_pair(pair,model)
        result['elapsed_s']=time.monotonic()-before
        report['results'].append(result)
        report['elapsed_s']=time.monotonic()-started
        (directory/('entity_merge_group_decisions.json' if options.groups else 'gliner2_identity_trial.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(pair_id=pair['id'],status=result['status'],reason=result['reason'],forward=result['scores_forward'],reverse=result['scores_reverse'])),flush=True)

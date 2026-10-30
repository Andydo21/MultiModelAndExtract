"""Apply updated gates to saved actual model outputs, without new inference."""
import hashlib
import json
from pathlib import Path
from ai_extract.services.gliner_identity_classifier import classify_pair

directory=Path('evaluation/benchmarks/corpus20')
source=directory/'gliner2_identity_trial.json'
trial=json.loads(source.read_text(encoding='utf-8'))
pairs={p['id']:p for p in json.loads((directory/'gliner_identity_source_pairs_v3.json').read_text(encoding='utf-8'))}
class RecordedModel:
    def __init__(self,row):
        self.responses=iter(row['raw_outputs']+[r['raw'] for r in row.get('reference_checks',[])])
    def classify_text(self,*args,**kwargs):
        return next(self.responses)

report=dict(mode='Postprocessing replay of recorded GLiNER2 outputs; no new model inference',
    model=trial['model'],revision=trial['revision'],source_model_log=source.name,
    source_model_log_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    gold_labels_available=False,graph_modified=False,
    gates=dict(threshold=.9,margin=.1,both_orientations=True,min_context_words_beyond_name=3,
               different_product_names_require_alias_evidence=True),results=[])
for row in trial['results']:
    report['results'].append(classify_pair(pairs[row['id']],RecordedModel(row)))
(directory/'gliner2_identity_policy_replay.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
from collections import Counter
print(dict(Counter(r['status'] for r in report['results'])))

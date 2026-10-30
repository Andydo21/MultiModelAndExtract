"""Read-only model probe over source pairs exported from the real graph."""
import json
import os
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HOME'] = 'D:/hf_cache'
os.environ['GLINER_MODEL_PATH'] = 'scratch/a3-models/gliner-relex-multi-v1.0'
from ai_extract.services.gliner_entity_service import infer_identity_pairs, runtime
import torch

directory = Path('evaluation/benchmarks/corpus20')
pairs = json.loads((directory/'identity_diagnostic_pairs.json').read_text(encoding='utf-8'))
results = []
model = runtime()
class Probe:
    def __init__(self, threshold):
        self.threshold = threshold
        self.raw = None
    def inference(self, **kwargs):
        kwargs['threshold'] = self.threshold
        self.raw = model.inference(**kwargs)
        return self.raw

with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
    # threshold=0 creates unbounded internal relation pairs before input_spans
    # are applied by this installed GLiNER decoder and OOMs on the 4 GB GPU.
    for threshold in (.5,):
        for offset in range(0, len(pairs), 8):
            probe = Probe(threshold)
            batch = pairs[offset:offset+8]
            decisions = infer_identity_pairs(batch, probe)
            record = dict(span_threshold=threshold, ids=[p['id'] for p in batch],
                          decisions=decisions, raw_entities=probe.raw[0], raw_relations=probe.raw[1])
            results.append(record)
            (directory/'identity_model_probe.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(dict(threshold=threshold, batch=offset,
                entities=[len(x) for x in probe.raw[0]], relations=[len(x) for x in probe.raw[1]],
                decisions=decisions)), flush=True)

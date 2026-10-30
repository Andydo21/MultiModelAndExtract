"""Live provider and frozen-input experiment execution. No synthetic fallbacks."""
import json
import os
import random
import time
from dataclasses import asdict
from datetime import datetime, timezone

from ..ablation_adapters import E1_FORMATS, E2_MODES, expand, messages
from ..dataset_loader import load_dataset, load_records
from ..io import append_record, digest, read_json, sha_file, write_json
from ..metrics import evidence_metrics, leakage_alerts
from ..provenance import git, verify_snapshot
from ..schema import RunResult


class Gateway:
    def __init__(self,config):
        import requests
        self.config=config
        self.base=os.getenv(config['base_url_env'],'').rstrip('/')
        if not self.base:
            raise ValueError('Generator base URL is not configured')
        self.session=requests.Session()
        self.session.headers['ngrok-skip-browser-warning']='true'
        key=os.getenv(config.get('api_key_env','VLLM_API_KEY'))
        if key:
            self.session.headers['Authorization']='Bearer '+key

    def health(self):
        response=self.session.get(self.base+'/models',timeout=10)
        response.raise_for_status()
        result=response.json()
        if self.config['model'] not in [item.get('id') for item in result.get('data',[])]:
            raise ValueError('Configured model not served')
        return result

    def count(self,chat):
        base=self.base[:-3] if self.base.endswith('/v1') else self.base
        response=self.session.post(base+'/tokenize',json=dict(model=self.config['model'],
            messages=chat,add_generation_prompt=True),timeout=30)
        response.raise_for_status()
        count=response.json().get('count')
        if type(count) is not int or count<0:
            raise ValueError('Exact server token count missing; no character estimate permitted')
        return count

    def request_body(self,chat,seed):
        return dict(model=self.config['model'],messages=chat,temperature=self.config['temperature'],
            top_p=self.config['top_p'],max_tokens=self.config['max_tokens'],seed=seed,
            response_format={'type':'json_object'})

    def generate(self,body):
        start=time.perf_counter()
        response=self.session.post(self.base+'/chat/completions',json=body,timeout=self.config['timeout_s'])
        latency=(time.perf_counter()-start)*1000
        response.raise_for_status()
        return response.json(),latency


def preflight(config,out):
    try:
        models=Gateway(config['generator']).health()
        result=dict(status='reachable',models=models,checkpoint_verified=False,
                    note='/models does not prove checkpoint revision or inference hardware')
    except Exception as error:
        response=getattr(error,'response',None)
        result=dict(status='unavailable',error_type=type(error).__name__,
                    http_status=response.status_code if response is not None else None)
    result['checked_at']=datetime.now(timezone.utc).isoformat()
    write_json(out/'gateway_preflight.json',result)
    return result


def check_response(row,response,model,expected):
    row['raw_response']=response
    usage=response.get('usage') or {}
    row.update(input_tokens=usage.get('prompt_tokens'),output_tokens=usage.get('completion_tokens'))
    choice=response['choices'][0]
    row.update(output=choice['message']['content'],finish_reason=choice.get('finish_reason'))
    if response.get('model')!=model or choice.get('finish_reason')!='stop':
        raise ValueError('Served model mismatch or incomplete output')
    if not all(type(row.get(k)) is int for k in ('input_tokens','output_tokens')):
        raise ValueError('API token usage missing')
    if expected is not None and row['input_tokens']!=expected:
        raise ValueError('Server tokenizer/chat usage mismatch')
    parsed=json.loads(row['output'])
    if not isinstance(parsed,dict) or not isinstance(parsed.get('answer'),str) or not isinstance(parsed.get('claims'),list):
        raise ValueError('Answer output does not follow the requested JSON schema')
    row['parsed_output']=parsed
    row['cited_evidence_ids']=list(dict.fromkeys(eid for claim in parsed['claims']
        for eid in claim.get('evidence_ids',[])))
    if set(row['cited_evidence_ids'])-set(row['selected_evidence_ids']):
        raise ValueError('Output cites evidence not sent to generator')


def execute(root,config,out,experiment,*,dataset,evidence_path,snapshot_path,mode='gold',
            retrieved_path=None,graph_path=None,repeat=None,formats=None,allow_unreviewed=False):
    snapshot=read_json(snapshot_path)
    verify_snapshot(root,snapshot)
    generator=config['generator']
    if not all(generator.get(k) for k in ('checkpoint_verified','checkpoint_revision','hardware')):
        raise ValueError('Verified checkpoint revision and actual inference hardware are required')
    if generator!=snapshot['generator']:
        raise ValueError('Generator differs from frozen snapshot')
    nodes=load_records(evidence_path)
    questions=load_dataset(dataset,nodes,allow_unreviewed)
    by_id={e['evidence_id']:e for e in nodes}
    retrieved=read_json(retrieved_path) if retrieved_path else None
    if mode not in ('gold','retrieved'):
        raise ValueError('Expected gold or retrieved evidence mode')
    if mode=='retrieved':
        if not retrieved or retrieved.get('corpus_sha256')!=snapshot['corpus_sha256'] or retrieved.get('dataset_sha256')!=sha_file(dataset):
            raise ValueError('Retrieval snapshot must bind exact corpus and dataset hashes')
        if not retrieved.get('retriever') or not retrieved.get('reranker_actual'):
            raise ValueError('Actual retriever and reranker (or explicit disabled) must be recorded')
    graph=read_json(graph_path) if graph_path else None
    if experiment=='E2' and (not graph or graph.get('nodes')!=nodes or
            (not allow_unreviewed and graph.get('status')!='approved')):
        raise ValueError('E2 needs a reviewed frozen graph with identical evidence nodes')
    repeats=repeat if repeat is not None else config['experiment'][experiment.lower()+'_repeats']
    if repeats<3:
        raise ValueError('E1/E2 require >=3 repetitions')
    variants=list(formats or (E1_FORMATS if experiment=='E1' else E2_MODES))
    allowed=E1_FORMATS if experiment=='E1' else tuple(E2_MODES)
    if not variants or len(set(variants))!=len(variants) or set(variants)-set(allowed):
        raise ValueError('Unknown or duplicate experiment format')
    schedule=[(q,rep,v) for rep in range(repeats) for q in questions for v in variants]
    random.Random(config['experiment']['schedule_seed']).shuffle(schedule)
    gateway=Gateway(generator)
    served=gateway.health()
    renderer=config['experiment'].get('renderer','canonical')
    manifest=dict(experiment=experiment,config=config,git_commit=git(root,'rev-parse','HEAD'),
        git_status=git(root,'status','--porcelain'),snapshot_id=snapshot['snapshot_id'],
        snapshot_sha256=sha_file(snapshot_path),dataset_sha256=sha_file(dataset),
        evidence_sha256=sha_file(evidence_path),retrieved_sha256=sha_file(retrieved_path) if retrieved_path else None,
        graph_sha256=sha_file(graph_path) if graph_path else None,served_models=served,
        evidence_mode=mode,exploratory=allow_unreviewed,renderer=renderer,
        independent_test=all(q['split']=='test' for q in questions) and not allow_unreviewed,
        latency_scope='frozen-evidence replay; excludes offline OCR, original retrieval and production queue',
        schedule=[dict(question_id=q['question_id'],repetition=r,variant=v) for q,r,v in schedule])
    write_json(out/'manifest.json',manifest)
    cache={}
    def count(chat):
        key=digest(chat)
        if key not in cache:
            cache[key]=gateway.count(chat)
        return cache[key]
    for question,rep,variant in schedule:
        row=asdict(RunResult(experiment,question['question_id'],variant,rep))
        row.update(evidence_mode=mode,exploratory=allow_unreviewed,split=question['split'],
            original_retrieval_ms=None,offline_ocr_ms=None,visual_tokens=None,gpu_memory_bytes=None)
        start=time.perf_counter()
        try:
            if mode=='gold':
                seeds=question['gold_evidence_ids']
            else:
                hit=retrieved['questions'][question['question_id']]
                if hit.get('question_sha256')!=digest(question['question']):
                    raise ValueError('Retrieval query hash mismatch')
                seeds=hit['evidence_ids']
                row['original_retrieval_ms']=hit.get('latency_ms')
            if len(set(seeds))!=len(seeds):
                raise ValueError('Duplicate retrieval seed IDs')
            selected=[by_id[eid] for eid in seeds]
            row.update(seed_evidence_ids=seeds,seed_evidence_sha256=digest(selected))
            budget=config['experiment']['max_input_tokens']
            render_options=dict(renderer=renderer,graph=graph or {})
            graph_start=time.perf_counter()
            if experiment=='E2':
                selected,row['graph_trace']=expand(seeds,nodes,graph['edges'],variant,
                    lambda es:count(messages(question['question'],es,'E',**render_options)),
                    budget,config['experiment']['graph_hops'])
            row['graph_latency_ms']=(time.perf_counter()-graph_start)*1000 if experiment=='E2' else None
            selected_ids=[e['evidence_id'] for e in selected]
            row.update(selected_evidence_ids=selected_ids,selected_evidence_sha256=digest(selected))
            row.update(evidence_metrics(selected_ids,question['gold_evidence_ids'],
                                       question['required_evidence_ids'],config['experiment']['top_k']))
            representation=variant if experiment=='E1' else 'E'
            chat=messages(question['question'],selected,representation,**render_options)
            row['prompt']=chat
            token_start=time.perf_counter()
            expected=count(chat)
            row['preflight_input_tokens']=expected
            if expected>budget:
                raise ValueError('Input budget exceeded; no source was truncated')
            model_info=next(m for m in served['data'] if m['id']==generator['model'])
            if model_info.get('max_model_len') and expected+generator['max_tokens']>model_info['max_model_len']:
                raise ValueError('Input plus reserved output exceeds served context window')
            if experiment=='E1':
                counts={v:count(messages(question['question'],selected,v,**render_options)) for v in variants}
                row['variant_token_counts']=counts
                if max(counts.values())>budget or max(counts.values())/max(1,min(counts.values()))>config['experiment']['max_variant_token_ratio']:
                    raise ValueError('Equivalent token budget constraint failed')
            row['tokenization_latency_ms']=(time.perf_counter()-token_start)*1000
            row['request']=gateway.request_body(chat,generator['seed']+rep)
            generation_start=time.perf_counter()
            try:
                response,latency=gateway.generate(row['request'])
                row['generation_http_ms']=latency
                check_response(row,response,generator['model'],expected)
            finally:
                row['generation_attempt_ms']=(time.perf_counter()-generation_start)*1000
            row['leakage_alerts']=leakage_alerts(row['output'],question)
            row['status']='completed_pending_review'
        except Exception as error:
            row.update(status='failed',error_type=type(error).__name__,
                error=str(error) if isinstance(error,(ValueError,KeyError)) else 'Provider request failed')
            response=getattr(error,'response',None)
            if response is not None:
                row['provider_http_status']=response.status_code
                try:
                    row['raw_error_response']=response.json()
                except ValueError:
                    row['raw_error_response']=response.text
        row['e2e_ms']=(time.perf_counter()-start)*1000
        append_record(out/'questions.jsonl',row)
        print(json.dumps({k:row[k] for k in ('question_id','variant','repetition','status')}),flush=True)
    return out

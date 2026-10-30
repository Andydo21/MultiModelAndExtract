"""New live E0 repetitions over real archived evidence, not exact historical replay."""
import random
import time

from ..ablation_adapters import messages
from ..io import append_record, digest, write_json
from ..provenance import git, source_hashes
from .common import Gateway, check_response
from .run_e0_verification import extract_originals


def archived_evidence(citations):
    result=[]
    for source in citations:
        item=dict(evidence_id=source['evidence_id'],text=source['text'],
            document_id=source.get('document_id'),version=source.get('version_id'),
            page=source.get('page_number'),region=source.get('bbox'),
            breadcrumb=source.get('heading_path',[]),scope=dict(document_id=source.get('document_id'),
                version_id=source.get('version_id'),heading_path=source.get('heading_path',[])),
            links=source.get('related_chunk_ids',[]),production_source=source)
        rows=source.get('structure_metadata',{}).get('table_rows')
        if rows:
            width=max(map(len,rows))
            padded=[list(row)+['']*(width-len(row)) for row in rows]
            item['table']=dict(columns=padded[0],rows=padded[1:])
        result.append(item)
    return result


def replay(root,config,out,repeat=5,*,paired_from_json=False):
    from ..io import read_json
    if repeat<5:
        raise ValueError('E0 needs >=5 repetitions')
    mapping=read_json(root/'evaluation/benchmarks/historical_audit.json')
    originals,missing=extract_originals(root,mapping)
    if missing or not originals:
        raise ValueError('Original archives missing; refuse partial replay')
    gateway=Gateway(config['generator'])
    served=gateway.health()
    cases=[]
    originals_for_cases=[o for o in originals if o['configuration']=='J'] if paired_from_json else originals
    for original in originals_for_cases:
        evidence=archived_evidence(original['citations'])
        for variant in ('J','M') if paired_from_json else (original['configuration'],):
            cases.append(dict(question_id=original['question_id'],configuration=variant,
                question=original['question'],evidence=evidence,archive_file=original['archive_file'],
                archive_sha256=original['archive_sha256'],archive_pointer=original['archive_pointer']))
    schedule=[(case,rep) for case in cases for rep in range(repeat)]
    random.Random(config['experiment']['schedule_seed']).shuffle(schedule)
    write_json(out/'manifest.json',dict(experiment='E0_archived_evidence_replay',config=config,
        git_commit=git(root,'rev-parse','HEAD'),git_status=git(root,'status','--porcelain'),
        source_hashes=source_hashes(root),served_models=served,cases=cases,
        exploratory=True,historical_prompt_reproduced=False,controlled_comparison=False,
        paired_input_renderings=paired_from_json,
        paired_note='Both renderings use the historical JSON-run evidence; this is not a reconstruction of missing historical Markdown runs.' if paired_from_json else None,
        unknown=['historical_full_prompts','historical_corpus_snapshot','historical_checkpoint','historical_hardware'],
        latency_scope='frozen evidence generation replay, excluding original retrieval/OCR/production queue',
        schedule=[dict(question_id=c['question_id'],variant=c['configuration'],repetition=r) for c,r in schedule]))
    prepared={}
    for case in cases:
        variant='B' if case['configuration']=='J' else 'C'
        chat=messages(case['question'],case['evidence'],variant)
        try:
            count=gateway.count(chat)
        except Exception as error:
            response=getattr(error,'response',None)
            if response is None or response.status_code!=404:
                raise
            # Some gateways do not expose tokenization. Keep this exploratory:
            # actual usage is taken from the generation response, not estimated.
            count=None
        if count is not None and count>config['experiment']['max_input_tokens']:
            write_json(out/'blocked.json',dict(reason='token_budget_exceeded',question_id=case['question_id'],tokens=count))
            raise ValueError('Archived prompt exceeds budget; no evidence truncated')
        model=next(m for m in served['data'] if m['id']==config['generator']['model'])
        if count is not None and model.get('max_model_len') and count+config['generator']['max_tokens']>model['max_model_len']:
            raise ValueError('Archived prompt plus output exceeds served model context')
        prepared[(case['configuration'],case['question_id'])]=(chat,count)
    for case,rep in schedule:
        start=time.perf_counter()
        chat,count=prepared[(case['configuration'],case['question_id'])]
        row=dict(experiment='E0_archived_evidence_replay',question_id=case['question_id'],
            variant=case['configuration'],repetition=rep,evidence_mode='archived_retrieved',
            exploratory=True,historical_prompt_reproduced=False,controlled_comparison=False,
            paired_input_renderings=paired_from_json,
            prompt=chat,selected_evidence_ids=[e['evidence_id'] for e in case['evidence']],
            selected_evidence_sha256=digest(case['evidence']),preflight_input_tokens=count,
            tokenizer_preflight='unavailable' if count is None else 'server_exact',
            archive_file=case['archive_file'],archive_sha256=case['archive_sha256'],archive_pointer=case['archive_pointer'],
            accuracy=None,attribute_leakage_rate=None,citation_faithfulness=None,
            original_retrieval_ms=None,offline_ocr_ms=None,graph_latency_ms=None)
        row['request']=gateway.request_body(chat,config['generator']['seed']+rep)
        # Keep the same schema for both historical renderings. A JSON-object
        # hint alone allows output without the requested answer field.
        row['request']['response_format']={'type':'json_schema','json_schema':{
            'name':'e0_answer','strict':True,'schema':{
                'type':'object','additionalProperties':False,'required':['answer','claims'],
                'properties':{'answer':{'type':'string'},'claims':{'type':'array','items':{
                    'type':'object','additionalProperties':False,
                    'required':['subject','attribute','value','condition','evidence_ids'],
                    'properties':{'subject':{'type':'string'},'attribute':{'type':'string'},
                        'value':{'type':'string'},'condition':{'type':'string'},
                        'evidence_ids':{'type':'array','items':{'type':'string','enum':row['selected_evidence_ids']}}}}}}}}}
        try:
            response,elapsed=gateway.generate(row['request'])
            row['generation_http_ms']=elapsed
            check_response(row,response,config['generator']['model'],count)
            row['status']='completed_pending_review'
        except Exception as error:
            row.update(status='failed',error_type=type(error).__name__,
                error=str(error) if isinstance(error,ValueError) else 'Provider request failed')
            response=getattr(error,'response',None)
            if response is not None:
                row['provider_http_status']=response.status_code
                try:
                    row['raw_error_response']=response.json()
                except ValueError:
                    row['raw_error_response']=response.text
        row['e2e_ms']=(time.perf_counter()-start)*1000
        append_record(out/'questions.jsonl',row)
        print({k:row[k] for k in ('question_id','variant','repetition','status')},flush=True)
    return out

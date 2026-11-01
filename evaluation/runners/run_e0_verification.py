"""Reconcile real Markdown cells against original JSON with explicit rounding."""
import re
from decimal import Decimal, ROUND_HALF_UP

from ..io import append_record, read_json, sha_file, write_csv, write_json

QUESTIONS = [
    'Dual-arm Control điều khiển bao nhiêu robot, tương thích với dòng robot và bộ điều khiển nào? Nó giúp giảm diện tích lắp đặt và rút ngắn việc hiệu chỉnh như thế nào?',
    'So sánh tải nâng của HSR Series và HS-A1 Series. Tải tối đa là bao nhiêu và có điều kiện nhiệt độ nào cần lưu ý?',
    'EVP2 Guidance và EVP2 Runtime có vai trò khác nhau như thế nào? Khi thiết lập và khi vận hành có cần máy tính PC không?',
    'Chức năng tránh va chạm khi gắp phôi trong EVP2 hoạt động như thế nào và giúp tránh va chạm với đối tượng nào?',
]
STAGES = ('corpus_index','question_planning','query_embedding','candidate_search',
          'reranking','source_and_graph_expansion','answer_generation')


def extract_originals(root, mapping):
    records, missing = [], []
    for archive in mapping['raw_archives']:
        path=root/archive['path']
        if not path.exists():
            missing.append(dict(path=archive['path'],reason='missing_archive'))
            continue
        for index, raw in enumerate(read_json(path)):
            question=raw.get('question')
            qid=f'Q{QUESTIONS.index(question)+1}' if question in QUESTIONS else 'UNMATCHED'
            result=raw.get('result',{})
            responses=[c.get('response',{}) for c in result.get('qwen_outputs',[])]
            usage_complete=bool(responses) and all(all(type((r.get('usage') or {}).get(k)) is int
                for k in ('prompt_tokens','completion_tokens')) for r in responses)
            timings=result.get('stage_timings_ms') or {}
            total=sum(Decimal(str(timings[s])) for s in STAGES) if all(s in timings for s in STAGES) else None
            latency=raw.get('elapsed_s')
            records.append(dict(configuration=archive['configuration'],question_id=qid,question=question,
                original_number=raw.get('number'),e2e_s=latency,
                service_latency_ms=result.get('latency_ms'),stage_timings_ms=timings,
                stage_sum_ms_arithmetic=float(total) if total is not None else None,
                http_minus_stage_sum_ms=float(Decimal(str(latency))*1000-total) if latency is not None and total is not None else None,
                residual_attribution='unknown; queue/network/overlap not independently instrumented',
                timing_scope='HTTP submission and polling; not isolated LLM API latency',
                token_scope='captured qwen_outputs only; planning completeness unknown',
                input_tokens=sum(r['usage']['prompt_tokens'] for r in responses) if usage_complete else None,
                output_tokens=sum(r['usage']['completion_tokens'] for r in responses) if usage_complete else None,
                answer=result.get('answer_text',result.get('answer')),raw_responses=responses,
                citations=result.get('citations'),answer_source_ids=result.get('answer_source_ids'),
                archive_file=archive['path'],archive_sha256=sha_file(path),archive_pointer=f'/{index}',
                raw_archive_record=raw,independent_test=False,historical_prompt_available=False,
                checkpoint_verified=False,hardware_verified=False))
    return records,missing


def parse_number(text, metric):
    text=text.strip()
    if metric.endswith('tokens'):
        return Decimal(re.sub(r'[.\s]','',text))
    return Decimal(text.replace('.','').replace(',','.'))


def compare_value(claim, raw, metric):
    if raw is None:
        return dict(status='MISSING_RAW_VALUE',printed=claim,raw=None,delta=None)
    printed=parse_number(claim,metric)
    actual=Decimal(str(raw))
    resolution=Decimal(1) if metric.endswith('tokens') else Decimal(1).scaleb(printed.as_tuple().exponent)
    matches=actual.quantize(resolution,rounding=ROUND_HALF_UP)==printed
    return dict(status='MATCH_VERIFIED' if matches else 'MISMATCH',printed=claim,raw=raw,
                delta=float(printed-actual),resolution=str(resolution),rounding='ROUND_HALF_UP')


def table_lines(lines, anchor):
    positions=[i for i,line in enumerate(lines) if line.startswith(anchor)]
    if len(positions)!=1:
        return []
    result=[]
    started=False
    for index in range(positions[0],len(lines)):
        line=lines[index]
        if line.startswith('|'):
            started=True
            cells=[x.strip() for x in re.split(r'(?<!\\)\|',line.strip())[1:-1]]
            result.append((index+1,cells))
        elif started:
            break
    return result


def reconcile(root, out, mapping_path=None):
    mapping_path=mapping_path or root/'evaluation/benchmarks/historical_audit.json'
    mapping=read_json(mapping_path)
    records,missing=extract_originals(root,mapping)
    index={}
    for record in records:
        key=(record['configuration'],record['question_id'])
        if key in index:
            raise ValueError('Ambiguous original run mapping: '+str(key))
        index[key]=record
        append_record(out/'historical_raw.jsonl',record)
    comparisons=[]
    def add(manuscript,line,qid,configuration,metric,claim):
        raw=index.get((configuration,qid))
        actual=raw.get(metric) if raw else None
        if metric.startswith('stage:') and raw:
            ms=raw['stage_timings_ms'].get(metric.split(':',1)[1])
            actual=float(Decimal(str(ms))/1000) if ms is not None else None
        row=dict(manuscript=manuscript['path'],manuscript_sha256=sha_file(root/manuscript['path']),
            line=line,configuration=configuration,question_id=qid,metric=metric,
            archive_file=raw.get('archive_file') if raw else None,
            archive_sha256=raw.get('archive_sha256') if raw else None,
            archive_pointer=raw.get('archive_pointer') if raw else None)
        try:
            row.update(compare_value(claim,actual,metric))
        except Exception as error:
            row.update(status='UNPARSEABLE_CLAIM',printed=claim,raw=actual,delta=None,error_type=type(error).__name__)
        comparisons.append(row)
    for manuscript in mapping['manuscripts']:
        path=root/manuscript['path']
        if not path.exists():
            missing.append(dict(path=manuscript['path'],reason='missing_manuscript'))
            continue
        lines=path.read_text(encoding='utf-8').splitlines()
        for table in manuscript['tables']:
            rows=table_lines(lines,table['anchor'])
            if not rows:
                missing.append(dict(path=manuscript['path'],anchor=table['anchor'],reason='missing_or_ambiguous_table'))
            if table['kind']=='stage_table':
                for stage in STAGES:
                    found=[(line,cells) for line,cells in rows if cells and stage in cells[0]]
                    if len(found)!=1:
                        missing.append(dict(stage=stage,path=manuscript['path'],reason='missing_or_ambiguous_stage'))
                        continue
                    line,cells=found[0]
                    add(manuscript,line,table['question_id'],table['configuration'],'stage:'+stage,cells[1])
                continue
            for selector in table['rows']:
                found=[(line,cells) for line,cells in rows if cells and
                       (cells[0]==selector['prefix'] if table['kind']=='embedded_table' else cells[0].startswith(selector['prefix']))]
                if len(found)!=1:
                    missing.append(dict(path=manuscript['path'],selector=selector,reason='missing_or_ambiguous_row'))
                    continue
                line,cells=found[0]
                if table['kind']=='run_table':
                    for metric,column in table['columns'].items():
                        add(manuscript,line,selector['question_id'],selector['configuration'],metric,cells[column])
                else:
                    for configuration,column in table['configurations'].items():
                        cell=cells[column]
                        seconds=re.search(r'([\d.,]+)\s*s',cell)
                        tokens=re.search(r'([\d.,]+)\s*tokens',cell)
                        for metric,match in [('e2e_s',seconds),('input_tokens',tokens)]:
                            if match:
                                add(manuscript,line,selector['question_id'],configuration,metric,match[1])
                            else:
                                missing.append(dict(path=manuscript['path'],line=line,metric=metric,reason='missing_numeric_claim'))
    write_csv(out/'historical_runs.csv',records,[
        'configuration','question_id','original_number','e2e_s','service_latency_ms','input_tokens','output_tokens',
        'stage_sum_ms_arithmetic','http_minus_stage_sum_ms','residual_attribution','archive_file','archive_sha256','archive_pointer'])
    write_csv(out/'reconciliation.csv',comparisons)
    write_json(out/'reconciliation.json',dict(mapping_sha256=sha_file(mapping_path),fields=comparisons,missing=missing,
        verified_fields=sum(r['status']=='MATCH_VERIFIED' for r in comparisons),
        disclaimer='Field-level numeric reconciliation only; no certificate of answer correctness, causal effect or complete provenance.',
        quality_claims_status='not_semantically_verified',historical_repetitions_completed=False))
    return records,comparisons,missing


def run(root,config,out,*,repeat=None,reconcile_only=False):
    result=reconcile(root,out)
    if repeat is not None and not reconcile_only:
        from .replay import replay
        return replay(root,config,out,repeat)
    return result

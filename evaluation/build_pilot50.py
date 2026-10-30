"""Build the source-authored manual test pack. Never invokes a chat/model API."""
import hashlib
import json
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from evaluation.benchmarks.corpus20.pilot50_authored import QUESTIONS
from evaluation.schema import validate_questions

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/benchmarks/corpus20'
documents=json.loads((OUT/'pilot_source_export.json').read_text(encoding='utf-8'))
by_alias={d['alias']:d for d in documents}
assert len(documents)==20 and len(QUESTIONS)==50
subjects=dict(TWIN='Digital Twin Software',BOX='RC8 Controller Protective Box Light',PLUG='Spark Plug Installation',
    RACE='Iridium Racing spark plug',INSUL='Iridium Racing plug models',COMP='A/C compressor',
    CLUTCH='Clutch / Expansion Valve',DUALAC='Rear expansion valve in dual A/C system',RING='Compressor snap ring',
    FLASH='Magnetic clutch relay',FAN='A/C water temperature switch',COBOPT='COBOTTA options',EVP2='EVP2',
    WINCAPS='WINCAPS Plus',DUALARM='Dual-arm Control',LPH='LPH-040A1',JACKET='Robot Protective Jacket for VS068/VS087',
    VS050='VS-050S2',BRAKE='DENSO First Time Fit brake pad kit',WIPER='DENSO Wiper Blades')
records=[];evidence={}
for number,q in enumerate(QUESTIONS,1):
    refs=[];locations=[]
    for alias,block in q['refs']:
        doc=by_alias[alias]
        matches=[e for e in doc['leaves'] if e['chunk_id']==f'block:{block}:1:0']
        assert len(matches)==1,(alias,block)
        leaf=matches[0]
        assert leaf['text'].strip() and leaf['page_number']>=1 and leaf['bbox'] is not None
        eid=f"a3:v{doc['version_id']}:r{doc['run_id']}:{leaf['chunk_id']}"
        source=dict(evidence_id=eid,text=leaf['text'],document_id=f"contract-version-{doc['version_id']}",
            version=doc['version_id'],run_id=doc['run_id'],page=leaf['page_number'],region=leaf['bbox'],
            alias=alias,source_path=doc['source_path'],chunk_id=leaf['chunk_id'],
            source_chunk_id=leaf.get('source_chunk_id'),source_fingerprint=doc['fingerprint'],
            layout_type=leaf.get('layout_type'),structure_metadata=leaf.get('structure_metadata') or {},
            heading_path=leaf.get('heading_path') or [],
            retrieval_excluded=leaf.get('retrieval_excluded'))
        evidence[eid]=source
        refs.append(eid)
        locations.append({k:source[k] for k in ('document_id','version','page','region','run_id','alias','source_path','chunk_id')})
    records.append(dict(question_id=f'P{number:03d}',question=q['question'],split='pilot',categories=q['categories'],
        reference_answer=q['reference_answer'],ground_truth_tuples=[dict(subject=subjects.get(q['alias'],q['alias']),
            attribute=q['attribute'],value=q['value'],condition=q['condition'],evidence_ids=refs)],
        gold_evidence_ids=refs,required_evidence_ids=refs,evidence_locations=locations,
        reviewer='',review_status='pending_independent_review',annotation_method='assistant-authored from actual A3 evidence; not model response',
        document_aliases=list(dict.fromkeys(alias for alias,_ in q['refs']))))
assert len({r['question'] for r in records})==50
assert set(by_alias)=={a for r in records for a in r['document_aliases']}
validate_questions(records,list(evidence.values()),allow_unreviewed=True)
def jsonl(path,rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
jsonl(ROOT/'evaluation/benchmarks/pilot50_draft.jsonl',records)
jsonl(ROOT/'evaluation/benchmarks/pilot50_evidence.jsonl',evidence.values())
questions=['50 CÂU HỎI KIỂM THỬ THỦ CÔNG — 20 TÀI LIỆU A3',
    'Chỉ là bộ pilot, không phải test độc lập. Chưa gửi câu nào qua chat.\n']
answers=['# Đáp án đối chiếu pilot50',
    '\nBản nháp biên soạn từ nguồn thật, chưa được người rà soát độc lập phê duyệt. Không phải phản hồi của hệ thống chat.']
for r in records:
    questions.append(f"{r['question_id']}. {r['question']}")
    answers.extend([f"\n## {r['question_id']}\n",r['question'],f"\nĐáp án đối chiếu: {r['reference_answer']}\n"])
    for eid in r['gold_evidence_ids']:
        source=evidence[eid]
        answers.append(f"- Nguồn: `{source['source_path']}`, trang {source['page']}; `{eid}`; vùng `{source['region']}`.")
        answers.append('  Trích nguồn: '+source['text'].replace('\n',' / '))
(OUT/'pilot50_questions.txt').write_text('\n\n'.join(questions)+'\n',encoding='utf-8')
(OUT/'pilot50_reference_answers.md').write_text('\n'.join(answers)+'\n',encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),question_count=50,document_count=20,
    chat_executed=False,execution_policy='MANUAL_ONLY_BY_USER_REQUEST',
    independent_gold_review=False,regression_Q1_Q4_included=False,
    diagnostic_only=True,unseen_test_set=False,
    category_counts=dict(Counter(c for r in records for c in r['categories'])),
    visual_source_check='tmp/pdfs/pilot50-source-check.png; selected tables and diagram only, not independent review of all evidence',
    document_question_counts=dict(Counter(a for r in records for a in r['document_aliases'])),
    source_export_sha256=sha(OUT/'pilot_source_export.json'),
    documents=[dict(alias=d['alias'],source_path=d['source_path'],pdf_sha256=sha(ROOT/d['source_path']),
        document_id=f"contract-version-{d['version_id']}",version_id=d['version_id'],run_id=d['run_id'],fingerprint=d['fingerprint']) for d in documents],
    question_file='evaluation/benchmarks/pilot50_draft.jsonl',evidence_file='evaluation/benchmarks/pilot50_evidence.jsonl')
(OUT/'pilot50_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(questions=len(records),documents=len(documents),evidence_records=len(evidence),chat_executed=False)))

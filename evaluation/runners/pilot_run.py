"""E1/E2 generation, automatic judging and reporting over a frozen pilot directory.

  python -X utf8 -m evaluation.runners.pilot_run e1 FROZEN OUT [--repeat 3] [--workers 8]
  python -X utf8 -m evaluation.runners.pilot_run e2 FROZEN OUT
  python -X utf8 -m evaluation.runners.pilot_run judge FROZEN RUN_DIR
  python -X utf8 -m evaluation.runners.pilot_run report FROZEN RUN_DIR [RUN_DIR ...] --out REPORT_DIR

Generation and judging append hash-chained JSONL (evaluation.io.read_records verifies it).
Token counts come from the served model's tokenizer (local /rag/tokenize, checked against API usage).
Judge labels are automatic (Qwen judge), not human review; reports say so.
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import statistics
import threading
import time
from collections import defaultdict
from pathlib import Path

import requests

from ..ablation_adapters import SYSTEM, messages
from ..io import canonical, digest, read_records
from ..metrics import evidence_metrics
from .common import Gateway, check_response

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT/'evaluation/config.json').read_text(encoding='utf-8'))
TOKENIZER_URL = os.getenv('RAG_TOKENIZE_URL', 'http://localhost:8010/rag/tokenize')
RETRIEVED_K = 5     # E1 retrieved mode: top-5 production query_match sources
E2_SEEDS = 3        # E2: top-3 production query_match sources as fixed seeds
E2_BUDGET = 6000    # E2: one prompt-token ceiling for every expansion mode
ONLY = None         # --questions: run only these question IDs (worked examples)
MAX_REPS = 10**9    # --max-reps: stop after this many repetitions of the schedule
E2_STRUCTURAL = ('NEXT', 'CONTAINS', 'FIGURE_CAPTION_CANDIDATE', 'FIGURE_NEAR_TEXT_CANDIDATE', 'HAS_FIGURE_TEXT')


def load_env():
    for line in (ROOT/'.env').read_text(encoding='utf-8').splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip().strip('\'"'))


def jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


class Chain:
    """Thread-safe append with the same sequence/hash chain as evaluation.io.append_record."""
    def __init__(self, path):
        self.path, self.lock = Path(path), threading.Lock()
        rows = read_records(path) if self.path.exists() else []
        self.rows = rows
        self.sequence, self.previous = len(rows), rows[-1]['record_sha256'] if rows else None

    def append(self, value):
        with self.lock:
            record = dict(value, sequence=self.sequence, previous_sha256=self.previous)
            record['record_sha256'] = digest(record)
            with self.path.open('a', encoding='utf-8') as stream:
                stream.write(canonical(record) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            self.sequence, self.previous = self.sequence + 1, record['record_sha256']
            return record


_session = requests.Session()
_token_cache, _token_lock = {}, threading.Lock()


def count_tokens(chat):
    key = digest(chat)
    with _token_lock:
        if key in _token_cache:
            return _token_cache[key]
    response = _session.post(TOKENIZER_URL, json=dict(model=CONFIG['generator']['model'], messages=chat), timeout=60)
    response.raise_for_status()
    count = response.json()['count']
    with _token_lock:
        _token_cache[key] = count
    return count


def answer_schema(ids):
    claim = {'type':'object','additionalProperties':False,
        'required':['subject','attribute','value','condition','evidence_ids'],
        'properties':{'subject':{'type':'string'},'attribute':{'type':'string'},'value':{'type':'string'},
            'condition':{'type':'string'},'evidence_ids':{'type':'array','items':{'type':'string','enum':ids}}}}
    return {'type':'json_schema','json_schema':{'name':'pilot_answer','strict':True,'schema':{'type':'object',
        'additionalProperties':False,'required':['answer','claims'],
        'properties':{'answer':{'type':'string'},'claims':{'type':'array','items':claim}}}}}


class Frozen:
    def __init__(self, path):
        self.path = Path(path)
        self.questions = jsonl(self.path/'gold_questions.jsonl')
        self.nodes = {n['evidence_id']: n for n in jsonl(self.path/'evidence.jsonl')}
        leaves = self.path/'graph_leaves.jsonl'
        if leaves.exists():
            for node in jsonl(leaves):
                self.nodes.setdefault(node['evidence_id'], node)
        self.retrieval = json.loads((self.path/'retrieval.json').read_text(encoding='utf-8'))
        self.pods = json.loads((self.path/'production_pods.json').read_text(encoding='utf-8'))
        gold_ids = {eid for q in self.questions for eid in q['gold_evidence_ids']}
        by_hash = {self.nodes[eid]['content_hash']: eid for eid in gold_ids}
        # Duplicate uploads of one PDF produce identical source text in another version.
        # Repeated text inside the gold version itself is a different region, not the gold one.
        self.equivalent = {}
        for eid, n in self.nodes.items():
            match = by_hash.get(n.get('content_hash'))
            self.equivalent[eid] = match if match and self.nodes[match]['version'] != n['version'] else eid

    def retrieved_ids(self, qid, k):
        return [x['evidence_id'] for x in self.retrieval['questions'][qid]['ranked'][:k]]

    def as_gold(self, ids):
        return list(dict.fromkeys(self.equivalent.get(i, i) for i in ids))


def generate(gateway, chat, ids, rep, row):
    body = dict(gateway.request_body(chat, CONFIG['generator']['seed'] + rep), response_format=answer_schema(ids))
    row['request'] = dict(body, messages=row['prompt'])  # E4 prompt copies omit base64 image payloads
    started = time.perf_counter()
    try:
        response, latency = gateway.generate(body)
        row['generation_http_ms'] = latency
        check_response(row, response, CONFIG['generator']['model'], row['preflight_input_tokens'])
        row['status'] = 'completed_pending_review'
    except Exception as error:
        row.update(status='failed', error_type=type(error).__name__, error=str(error)[:500])
        response = getattr(error, 'response', None)
        if response is not None:
            row['provider_http_status'] = response.status_code
    row['generation_attempt_ms'] = (time.perf_counter() - started) * 1000
    return row


def run_schedule(out, manifest, schedule, build, workers):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'manifest.json').exists() or (out/'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    chain = Chain(out/'questions.jsonl')
    done = {(r['question_id'], r['evidence_mode'], r['variant'], r['repetition'])
            for r in chain.rows if r['status'] != 'failed'}
    gateway = Gateway(CONFIG['generator'])
    gateway.health()
    # Finish every question/variant of a repetition before the next one, so a
    # partial run is already a complete paired comparison. Order inside a
    # repetition stays shuffled (stable sort).
    if ONLY is not None:
        # Worked examples need one completed answer per variant, whichever repetition produced it.
        cells = {s[:3] for s in done}
        todo = [s for s in schedule if s[0] in ONLY and s[3] == 0 and s[:3] not in cells]
    else:
        todo = sorted((s for s in schedule if s not in done and s[3] < MAX_REPS), key=lambda s: s[3])
    print(f'{len(todo)} of {len(schedule)} scheduled calls remain', flush=True)

    def work(item):
        row = build(*item)
        chat = row.pop('_chat', None) or row['prompt']
        if row.get('status') != 'over_budget':
            generate(gateway, chat, row['selected_evidence_ids'], item[3], row)
        if row.get('text_only_tokens') is not None and type(row.get('input_tokens')) is int:
            row['visual_tokens'] = row['input_tokens'] - row['text_only_tokens']
        chain.append(row)
        return row['status']

    statuses = defaultdict(int)
    with cf.ThreadPoolExecutor(workers) as pool:
        for index, status in enumerate(pool.map(work, todo), 1):
            statuses[status] += 1
            if index % 25 == 0 or index == len(todo):
                print(index, dict(statuses), flush=True)


def base_row(experiment, q, mode, variant, rep):
    return dict(experiment=experiment, question_id=q['question_id'], evidence_mode=mode, variant=variant,
                repetition=rep, split=q['split'], categories=q['categories'], gold_review=q['review_scope'],
                accuracy=None, attribute_leakage_rate=None, citation_faithfulness=None)


def e1(frozen, out, repeat, workers):
    fz = Frozen(frozen)
    plans = {}
    for q in fz.questions:
        for mode, ids in (('gold', q['gold_evidence_ids']), ('retrieved', fz.retrieved_ids(q['question_id'], RETRIEVED_K))):
            pod = fz.pods[q['question_id']+':'+mode]
            if sorted(pod['order']) != sorted(ids):
                raise ValueError('Production pod order does not cover the frozen evidence set')
            evidence = [fz.nodes[i] for i in pod['order']]
            chats = {v: messages(q['question'], evidence, v) for v in 'ABCD'}
            chats['E'] = [dict(role='system', content=SYSTEM),
                          dict(role='user', content=q['question']+'\n\nEVIDENCE\n'+pod['rendered'])]
            counts = {v: count_tokens(c) for v, c in chats.items()}
            plans[q['question_id'], mode] = dict(q=q, ids=pod['order'], chats=chats, counts=counts,
                evidence_sha256=digest(evidence), retrieval_latency_ms=fz.retrieval['questions'][q['question_id']]['latency_ms'])
    schedule = [(q['question_id'], mode, v, rep) for rep in range(repeat) for q in fz.questions
                for mode in ('gold', 'retrieved') for v in 'ABCDE']
    random.Random(CONFIG['experiment']['schedule_seed']).shuffle(schedule)

    def build(qid, mode, variant, rep):
        plan = plans[qid, mode]
        q, counts = plan['q'], plan['counts']
        row = base_row('E1', q, mode, variant, rep)
        gold = q['gold_evidence_ids']
        row.update(selected_evidence_ids=plan['ids'], evidence_sha256=plan['evidence_sha256'],
            prompt=plan['chats'][variant], preflight_input_tokens=counts[variant], variant_token_counts=counts,
            token_ratio_max_min=max(counts.values())/min(counts.values()), token_ratio_e_d=counts['E']/counts['D'],
            original_retrieval_ms=plan['retrieval_latency_ms'] if mode == 'retrieved' else None,
            **evidence_metrics(fz.as_gold(plan['ids']), gold, q['required_evidence_ids'], CONFIG['experiment']['top_k']))
        if counts[variant] > CONFIG['experiment']['max_input_tokens']:
            row['status'] = 'over_budget'
        return row

    manifest = dict(experiment='E1', frozen=str(frozen), frozen_hashes=frozen_hashes(fz.path), config=CONFIG,
        modes=dict(gold='gold evidence', retrieved=f'top-{RETRIEVED_K} production query_match sources'),
        formats=dict(A='flat text; table cells as "a ; b" lines', B='JSON {evidence_id,text,table}',
            C='Markdown tables', D='C + heading breadcrumb', E='production answer_source_pods + render_source_pods'),
        order='production pod order, shared by all formats', repeat=repeat, schedule=schedule,
        token_policy=f'exact server-tokenizer counts; ceiling {CONFIG["experiment"]["max_input_tokens"]}; '
                     'ratio recorded, never padded or truncated', gold_review='light_text_check (not independent)')
    run_schedule(out, manifest, schedule, build, workers)


def e2_expand(fz, graph_index, seeds, mode, question):
    """A none; B NEXT both directions; C B + CONTAINS parent/siblings + figure caption/text links;
    D C restricted to the seed's own heading scope (same heading path, or its direct section heading)."""
    selected, trace = list(seeds), []
    def chat(ids):
        return messages(question, [fz.nodes[i] for i in ids], 'E')
    if count_tokens(chat(selected)) > E2_BUDGET:
        raise ValueError('Seeds exceed E2 budget')
    if mode == 'A':
        return selected, trace
    for seed in seeds:
        g = graph_index[fz.nodes[seed]['version']]
        for target, edge, distance in g.candidates(fz.nodes[seed]['production_source']['chunk_id'], mode):
            eid = g.evidence_id(target)
            node = fz.nodes.get(eid)
            reason = None
            if node is None:
                reason = 'missing_leaf'
            elif eid in selected:
                reason = 'already_selected'
            elif not node['text'].strip() and not node.get('table'):
                reason = 'no_text'
            elif mode == 'D' and not same_scope(fz.nodes[seed], node):
                reason = 'scope_boundary'
            elif count_tokens(chat(selected + [eid])) > E2_BUDGET:
                reason = 'token_budget'
            trace.append(dict(seed=seed, target=eid, edge=edge, distance=distance, accepted=reason is None,
                              reason=reason or 'admitted'))
            if reason is None:
                selected.append(eid)
    return selected, trace


def same_scope(seed, node):
    left, right = seed['breadcrumb'], node['breadcrumb']
    return right == left or (right and left[:len(right)] == right and len(left) - len(right) == 1)


class GraphIndex:
    def __init__(self, graph):
        self.version, self.run = graph['version_id'], graph['run_id']
        self.nodes = {n['key']: n for n in graph['nodes']}
        self.out, self.into = defaultdict(list), defaultdict(list)
        for e in graph['edges']:
            self.out[e['source']].append(e)
            self.into[e['target']].append(e)
        self.evidence_key = {n['leaf_id']: k for k, n in self.nodes.items() if n.get('category') == 'evidence' and n.get('leaf_id')}

    def evidence_id(self, key):
        node = self.nodes[key]
        return f'a3:v{self.version}:r{self.run}:{node["leaf_id"]}'

    def owners(self, key):
        return [e['source'] for e in self.into[key] if e['type'] == 'HAS_EVIDENCE']

    def leaves(self, owner):
        return [e['target'] for e in self.out[owner] if e['type'] == 'HAS_EVIDENCE']

    def candidates(self, chunk, mode):
        """(evidence key, edge label, structural distance), nearest first; one structural step from the seed."""
        key = self.evidence_key.get(chunk)
        if key is None:
            return []
        found = []
        for owner in self.owners(key):
            for e in self.out[owner] + self.into[owner]:
                other = e['target'] if e['source'] == owner else e['source']
                if e['type'] == 'NEXT':
                    found += [(leaf, 'previous' if e['target'] == owner else 'next', 1) for leaf in self.leaves(other)]
                if mode in 'CD' and e['type'] == 'CONTAINS' and e['target'] == owner:
                    parent = self.nodes.get(e['source'], {})
                    found += [(leaf, 'parent', 2) for leaf in self.leaves(e['source'])]
                    if parent.get('kind') != 'document':
                        for child in self.out[e['source']]:
                            if child['type'] == 'CONTAINS' and child['target'] != owner:
                                found += [(leaf, 'sibling', 3) for leaf in self.leaves(child['target'])]
            if mode in 'CD':
                for e in self.out[key] + self.into[key] + self.out[owner] + self.into[owner]:
                    if e['type'] in E2_STRUCTURAL[2:]:
                        other = e['target'] if e['source'] in (key, owner) else e['source']
                        found += [(leaf, e['type'].lower(), 2) for leaf in ([other] if self.nodes.get(other, {}).get('category') == 'evidence' else self.leaves(other))]
        seen, result = set(), []
        for leaf, label, distance in sorted(found, key=lambda x: x[2]):
            if leaf != key and leaf not in seen and leaf in self.nodes:
                seen.add(leaf)
                result.append((leaf, label, distance))
        return result


def e2(frozen, out, repeat, workers):
    fz = Frozen(frozen)
    graphs = {g['version_id']: GraphIndex(g) for g in json.loads((fz.path/'graphs.json').read_text(encoding='utf-8'))}
    plans = {}
    for q in fz.questions:
        seeds = fz.retrieved_ids(q['question_id'], E2_SEEDS)
        for mode in 'ABCD':
            started = time.perf_counter()
            ids, trace = e2_expand(fz, graphs, seeds, mode, q['question'])
            graph_ms = (time.perf_counter() - started) * 1000
            plans[q['question_id'], mode] = dict(q=q, seeds=seeds, ids=ids, trace=trace, graph_ms=graph_ms,
                chat=messages(q['question'], [fz.nodes[i] for i in ids], 'E'))
    schedule = [(q['question_id'], 'retrieved', v, rep) for rep in range(repeat) for q in fz.questions for v in 'ABCD']
    random.Random(CONFIG['experiment']['schedule_seed']).shuffle(schedule)

    def build(qid, mode, variant, rep):
        plan = plans[qid, variant]
        q = plan['q']
        row = base_row('E2', q, mode, variant, rep)
        row.update(seed_evidence_ids=plan['seeds'], selected_evidence_ids=plan['ids'], graph_trace=plan['trace'],
            added_nodes=len(plan['ids']) - len(plan['seeds']), graph_latency_ms=plan['graph_ms'],
            prompt=plan['chat'], preflight_input_tokens=count_tokens(plan['chat']), evidence_sha256=digest(plan['ids']),
            **evidence_metrics(fz.as_gold(plan['ids']), q['gold_evidence_ids'], q['required_evidence_ids'], CONFIG['experiment']['top_k']))
        return row

    manifest = dict(experiment='E2', frozen=str(frozen), frozen_hashes=frozen_hashes(fz.path), config=CONFIG,
        seeds=f'top-{E2_SEEDS} production query_match sources', budget_tokens=E2_BUDGET, renderer='canonical E',
        modes=dict(A='no expansion', B='NEXT predecessor/successor blocks', C='B + CONTAINS parent heading, '
            'siblings (not under document root), figure caption/near-text/figure-text links',
            D='C admitted only inside the seed heading scope'), hops=1, repeat=repeat, schedule=schedule,
        graph_latency_scope='in-memory traversal incl. local tokenizer calls for the budget; excludes Neo4j export')
    run_schedule(out, manifest, schedule, build, workers)


E4_VARIANTS = dict(F='full-page VLM: page images only', K='crop VLM: evidence-region crops only',
                   O='OCR-only text (format A)', S='OCR + caption structured text (production pods, format E)')


def image_part(path):
    import base64
    return dict(type='image_url', image_url=dict(url='data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode()))


def region_crop(node, cache_dir):
    """Crop the evidence bbox from its rendered page image (bbox is in page-image pixels)."""
    from PIL import Image
    source = node['production_source']
    target = Path(cache_dir)/(re.sub(r'[^\w.-]', '_', node['evidence_id'])+'.png')
    if not target.exists():
        page = Image.open(ROOT/'media'/source['page_image_path'])
        x0, y0, x1, y1 = node['region']
        box = (max(0, int(x0)-8), max(0, int(y0)-8), min(page.width, int(x1)+8), min(page.height, int(y1)+8))
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            raise ValueError('Evidence bbox does not fit the page image')
        page.crop(box).save(target)
    return target


def e4(frozen, out, repeat, workers):
    fz = Frozen(frozen)
    subset = [q for q in fz.questions if {'table', 'figure_text'} & set(q['categories'])
              or any(fz.nodes[i]['layout_type'] in ('table', 'figure') for i in q['gold_evidence_ids'])]
    crops = Path(out)/'crops'
    crops.mkdir(parents=True, exist_ok=True)
    plans = {}
    for q in subset:
        pod = fz.pods[q['question_id']+':gold']
        ids = pod['order']
        nodes = [fz.nodes[i] for i in ids]
        instruction = q['question']+'\n\nEVIDENCE IMAGES (cite the IDs shown before each image)'
        pages = {}
        for n in nodes:
            pages.setdefault(n['production_source']['page_image_path'], []).append(n['evidence_id'])
        full = [dict(type='text', text=instruction)]
        for path, page_ids in pages.items():
            full += [dict(type='text', text='Page image containing '+', '.join('['+i+']' for i in page_ids)),
                     image_part(ROOT/'media'/path)]
        crop = [dict(type='text', text=instruction)]
        for n in nodes:
            crop += [dict(type='text', text='['+n['evidence_id']+']'), image_part(region_crop(n, crops))]
        chats = dict(F=[dict(role='system', content=SYSTEM), dict(role='user', content=full)],
                     K=[dict(role='system', content=SYSTEM), dict(role='user', content=crop)],
                     O=messages(q['question'], nodes, 'A'),
                     S=[dict(role='system', content=SYSTEM), dict(role='user', content=q['question']+'\n\nEVIDENCE\n'+pod['rendered'])])
        text_only = {v: count_tokens([dict(m, content=m['content'] if isinstance(m['content'], str) else
                     [p for p in m['content'] if p['type'] == 'text']) for m in chat]) for v, chat in chats.items()}
        plans[q['question_id']] = dict(q=q, ids=ids, chats=chats, text_only=text_only,
            pages=len(pages), evidence_sha256=digest(nodes))
    schedule = [(q['question_id'], 'gold', v, rep) for rep in range(repeat) for q in subset for v in E4_VARIANTS]
    random.Random(CONFIG['experiment']['schedule_seed']).shuffle(schedule)

    def build(qid, mode, variant, rep):
        plan = plans[qid]
        row = base_row('E4', plan['q'], mode, variant, rep)
        row.update(selected_evidence_ids=plan['ids'], evidence_sha256=plan['evidence_sha256'], images=plan['pages']
            if variant == 'F' else len(plan['ids']) if variant == 'K' else 0, text_only_tokens=plan['text_only'][variant],
            preflight_input_tokens=plan['text_only'][variant] if variant in 'OS' else None,
            prompt=[dict(m, content=m['content'] if isinstance(m['content'], str) else
                    [p if p['type'] == 'text' else dict(type='image_url', image_url='<base64 omitted>') for p in m['content']])
                    for m in plan['chats'][variant]], gpu_memory_bytes=None)
        row['_chat'] = plan['chats'][variant]
        return row

    manifest = dict(experiment='E4', frozen=str(frozen), frozen_hashes=frozen_hashes(fz.path), config=CONFIG,
        subset=[q['question_id'] for q in subset], variants=E4_VARIANTS, repeat=repeat, schedule=schedule,
        visual_tokens='API prompt_tokens minus exact tokenizer count of the same messages without images',
        gpu_memory='not measurable: inference runs on the remote Kaggle gateway; recorded null',
        offline_latency='see e4_offline_latency.json (A3 run wall time per page; whole pipeline, not OCR alone)')
    run_schedule(out, manifest, schedule, build, workers)


def frozen_hashes(path):
    from ..io import sha_file
    return {p.name: sha_file(p) for p in sorted(Path(path).iterdir()) if p.is_file() and p.suffix in ('.json', '.jsonl')}


JUDGE_SYSTEM = (
    'You grade answers to technical document questions. Use ONLY the reference answer, gold facts and evidence '
    'given. Rules: "correct" = every main requested point is right (values, units, subjects, important conditions); '
    'paraphrase and other languages are fine. "partial" = some main point right but a requested point or an '
    'important condition missing. "wrong" = a wrong value/unit/subject/condition, unsupported conclusion or off-topic. '
    '"no_answer" = the answer declines or says evidence is insufficient. For each claim: supported_by_cited = the '
    'cited evidence text itself states the claim (not merely related). attribute_leakage = the claim attaches a value '
    'or condition to the wrong subject/attribute/condition (e.g. a value of model X given for model Y). Return JSON.')


def judge_schema(n):
    claim = {'type':'object','additionalProperties':False,'required':['index','supported_by_cited','attribute_leakage'],
             'properties':{'index':{'type':'integer'},'supported_by_cited':{'type':'boolean'},'attribute_leakage':{'type':'boolean'}}}
    return {'type':'json_schema','json_schema':{'name':'judge','strict':True,'schema':{'type':'object',
        'additionalProperties':False,'required':['answer_label','missing_points','claims'],
        'properties':{'answer_label':{'type':'string','enum':['correct','partial','wrong','no_answer']},
            'missing_points':{'type':'array','items':{'type':'string'}},
            'claims':{'type':'array','items':claim,'minItems':n,'maxItems':n}}}}}


def judge_prompt(q, row, nodes):
    claims = row['parsed_output']['claims']
    cited = {eid for c in claims for eid in c.get('evidence_ids', [])}
    lines = ['QUESTION: '+q['question'], 'REFERENCE ANSWER: '+q['reference_answer'],
             'GOLD FACTS: '+json.dumps(q['ground_truth_tuples'], ensure_ascii=False),
             'GOLD EVIDENCE:'] + ['['+i+'] '+nodes[i]['text'] for i in q['gold_evidence_ids']]
    lines += ['CITED EVIDENCE:'] + ['['+i+'] '+nodes[i]['text'] for i in row['selected_evidence_ids'] if i in cited]
    lines += ['MODEL ANSWER: '+row['parsed_output']['answer'], 'MODEL CLAIMS:']
    lines += [f'{n}. '+json.dumps(c, ensure_ascii=False) for n, c in enumerate(claims)]
    return [dict(role='system', content=JUDGE_SYSTEM), dict(role='user', content='\n'.join(lines))]


def key_values(value):
    """Numbers, codes and short atoms in gold tuples; deterministic recall check, not a semantic judgment."""
    if isinstance(value, dict):
        return [x for v in value.values() for x in key_values(v)]
    if isinstance(value, list):
        return [x for v in value for x in key_values(v)]
    if isinstance(value, bool) or value is None:
        return []
    return re.findall(r'[A-Za-z]*[\d][\w.\-/±°]*', str(value))


def normalize(text):
    return re.sub(r'\s+', '', str(text)).replace(',', '.').lower()


def judge(frozen, run_dir, workers):
    load_env()
    fz = Frozen(frozen)
    questions = {q['question_id']: q for q in fz.questions}
    rows = [r for r in read_records(Path(run_dir)/'questions.jsonl') if r['status'] == 'completed_pending_review']
    chain = Chain(Path(run_dir)/'judgments.jsonl')
    done = {r['record_sha256_judged'] for r in chain.rows if r['status'] == 'judged'}
    gateway = Gateway(CONFIG['generator'])
    cache, lock = {}, threading.Lock()

    def work(row):
        q = questions[row['question_id']]
        chat = judge_prompt(q, row, fz.nodes)
        claims = row['parsed_output']['claims']
        key = digest(chat)
        result = dict(record_sha256_judged=row['record_sha256'], question_id=row['question_id'],
            experiment=row['experiment'], evidence_mode=row['evidence_mode'], variant=row['variant'],
            repetition=row['repetition'], judge_model=CONFIG['generator']['model'], judge='automatic_qwen_same_model',
            judge_prompt_sha256=key)
        answer = normalize(row['parsed_output']['answer'] + ' ' + ' '.join(c['value'] for c in claims))
        keys = list(dict.fromkeys(k for t in q['ground_truth_tuples'] for k in map(normalize, key_values(t['value'])) if len(k) > 1))
        result['key_value_recall'] = sum(k in answer for k in keys)/len(keys) if keys else None
        try:
            with lock:
                cached = cache.get(key)
            if cached is None:
                body = dict(model=CONFIG['generator']['model'], messages=chat, temperature=0.0, top_p=1.0,
                            max_tokens=1024, seed=CONFIG['generator']['seed'], response_format=judge_schema(len(claims)))
                response, latency = gateway.generate(body)
                choice = response['choices'][0]
                if choice.get('finish_reason') != 'stop':
                    raise ValueError('Judge output incomplete')
                cached = json.loads(choice['message']['content'])
                with lock:
                    cache[key] = cached
            verdict = cached
            flags = {c['index']: c for c in verdict['claims']}
            cited = [i for i, c in enumerate(claims) if c.get('evidence_ids')]
            eligible = [i for i, c in enumerate(claims) if c.get('value', '').strip()]
            result.update(status='judged', verdict=verdict, answer_label=verdict['answer_label'],
                accuracy=1 if verdict['answer_label'] == 'correct' else 0,
                partial_credit=dict(correct=1, partial=0.5).get(verdict['answer_label'], 0),
                eligible_assertions=len(eligible), leaked_assertions=sum(bool(flags.get(i, {}).get('attribute_leakage')) for i in eligible),
                cited_claims=len(cited), supported_cited_claims=sum(bool(flags.get(i, {}).get('supported_by_cited')) for i in cited))
            result['attribute_leakage_rate'] = result['leaked_assertions']/len(eligible) if eligible else None
            result['citation_faithfulness'] = result['supported_cited_claims']/len(cited) if cited else None
        except Exception as error:
            result.update(status='failed', error_type=type(error).__name__, error=str(error)[:300])
        chain.append(result)
        return result['status']

    todo = [r for r in rows if r['record_sha256'] not in done]
    print(f'judging {len(todo)} of {len(rows)} completed generations', flush=True)
    statuses = defaultdict(int)
    with cf.ThreadPoolExecutor(workers) as pool:
        for index, status in enumerate(pool.map(work, todo), 1):
            statuses[status] += 1
            if index % 50 == 0 or index == len(todo):
                print(index, dict(statuses), flush=True)


METRICS = ('accuracy', 'partial_credit', 'attribute_leakage_rate', 'citation_faithfulness', 'key_value_recall',
           'input_tokens', 'output_tokens', 'generation_http_ms', 'evidence_recall_at_k', 'evidence_precision_at_k',
           'required_evidence_recall', 'added_nodes', 'graph_latency_ms', 'visual_tokens')


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def report(frozen, run_dirs, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from openpyxl import Workbook
    from scipy.stats import wilcoxon
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    questions = {q['question_id']: q for q in jsonl(Path(frozen)/'gold_questions.jsonl')}
    rows = []
    for run_dir in run_dirs:
        generations = read_records(Path(run_dir)/'questions.jsonl')
        judged = {}
        if (Path(run_dir)/'judgments.jsonl').exists():
            for j in read_records(Path(run_dir)/'judgments.jsonl'):
                if j['status'] == 'judged':
                    judged[j['record_sha256_judged']] = j
        latest = {}
        for g in generations:
            latest[g['question_id'], g['evidence_mode'], g['variant'], g['repetition']] = g
        for g in latest.values():
            j = judged.get(g['record_sha256'], {})
            if not j and g.get('parsed_output'):
                # Deterministic metric needs no judge call, so interim reports can show it.
                q = questions[g['question_id']]
                answer = normalize(g['parsed_output']['answer'] + ' ' + ' '.join(c['value'] for c in g['parsed_output']['claims']))
                keys = list(dict.fromkeys(k for t in q['ground_truth_tuples'] for k in map(normalize, key_values(t['value'])) if len(k) > 1))
                j = dict(key_value_recall=sum(k in answer for k in keys)/len(keys) if keys else None, interim=True)
            rows.append(dict({k: g.get(k) for k in ('experiment','question_id','evidence_mode','variant','repetition','status',
                'input_tokens','output_tokens','generation_http_ms','evidence_recall_at_k','evidence_precision_at_k',
                'required_evidence_recall','added_nodes','graph_latency_ms','visual_tokens','images','token_ratio_max_min',
                'token_ratio_e_d','error')},
                categories='|'.join(g.get('categories', [])), judged=bool(j) and not j.get('interim'), answer_label=j.get('answer_label'),
                **{k: j.get(k) for k in ('accuracy','partial_credit','attribute_leakage_rate','citation_faithfulness',
                                         'key_value_recall','leaked_assertions','eligible_assertions','cited_claims','supported_cited_claims')}))
    groups = defaultdict(list)
    for r in rows:
        groups[r['experiment'], r['evidence_mode'], r['variant']].append(r)
    summary = []
    for (experiment, mode, variant), items in sorted(groups.items()):
        line = dict(experiment=experiment, evidence_mode=mode, variant=variant, scheduled=len(items),
                    completed=sum(r['status'] == 'completed_pending_review' for r in items),
                    failed=sum(r['status'] == 'failed' for r in items), over_budget=sum(r['status'] == 'over_budget' for r in items),
                    judged=sum(r['judged'] for r in items),
                    wrong=sum(r['answer_label'] == 'wrong' for r in items), partial=sum(r['answer_label'] == 'partial' for r in items),
                    correct=sum(r['answer_label'] == 'correct' for r in items), no_answer=sum(r['answer_label'] == 'no_answer' for r in items))
        reps = sorted({r['repetition'] for r in items})
        for metric in METRICS:
            per_rep = [mean(r[metric] for r in items if r['repetition'] == rep) for rep in reps]
            per_rep = [v for v in per_rep if v is not None]
            line[metric+'_mean'] = statistics.mean(per_rep) if per_rep else None
            line[metric+'_sd'] = statistics.stdev(per_rep) if len(per_rep) > 1 else None
        summary.append(line)
    tests = []
    for experiment, mode, left, right in (('E1','gold','E','D'), ('E1','retrieved','E','D'), ('E1','gold','E','A'),
                                          ('E1','retrieved','E','A'), ('E2','retrieved','D','A'), ('E2','retrieved','D','C'),
                                          ('E2','retrieved','C','A'), ('E2','retrieved','B','A')):
        for metric in ('partial_credit', 'accuracy', 'attribute_leakage_rate', 'citation_faithfulness', 'input_tokens'):
            per_q = defaultdict(dict)
            for r in rows:
                if r['experiment'] == experiment and r['evidence_mode'] == mode and r['variant'] in (left, right):
                    per_q[r['question_id']].setdefault(r['variant'], []).append(r[metric])
            pairs = [(mean(v[left]), mean(v[right])) for v in per_q.values() if left in v and right in v]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            diffs = [a - b for a, b in pairs]
            result = dict(experiment=experiment, evidence_mode=mode, comparison=f'{left} vs {right}', metric=metric,
                          questions=len(pairs), mean_left=mean(a for a, _ in pairs), mean_right=mean(b for _, b in pairs),
                          mean_difference=mean(diffs), wins=sum(d > 0 for d in diffs), losses=sum(d < 0 for d in diffs),
                          ties=sum(d == 0 for d in diffs), p_wilcoxon=None)
            if len(pairs) >= 5 and any(diffs):
                result['p_wilcoxon'] = float(wilcoxon([a for a, _ in pairs], [b for _, b in pairs], zero_method='zsplit').pvalue)
            tests.append(result)
    m = len([t for t in tests if t['p_wilcoxon'] is not None])
    ordered = sorted((t for t in tests if t['p_wilcoxon'] is not None), key=lambda t: t['p_wilcoxon'])
    running = 0
    for rank, t in enumerate(ordered):
        running = max(running, min(1.0, t['p_wilcoxon'] * (m - rank)))
        t['p_holm'] = running
    failures = [r for r in rows if r['status'] != 'completed_pending_review' or r['answer_label'] in ('wrong', 'no_answer')
                or (r['leaked_assertions'] or 0) > 0]
    category = defaultdict(list)
    for r in rows:
        for c in r['categories'].split('|'):
            category[r['experiment'], r['evidence_mode'], r['variant'], c].append(r['partial_credit'])
    by_category = [dict(experiment=e, evidence_mode=m_, variant=v, category=c, n=len(vals), partial_credit_mean=mean(vals))
                   for (e, m_, v, c), vals in sorted(category.items())]
    for name, data in (('runs', rows), ('summary', summary), ('paired_tests', tests), ('failures', failures),
                       ('by_category', by_category)):
        write_rows(out/(name+'.csv'), data)
    book = Workbook()
    book.remove(book.active)
    for name, data in (('summary', summary), ('paired_tests', tests), ('by_category', by_category),
                       ('failures', failures), ('runs', rows)):
        sheet = book.create_sheet(name)
        fields = list(dict.fromkeys(k for r in data for k in r))
        sheet.append(fields)
        for r in data:
            sheet.append([r.get(k) if not isinstance(r.get(k), (list, dict)) else canonical(r.get(k)) for k in fields])
    book.save(out/'results.xlsx')
    for experiment in sorted({s['experiment'] for s in summary}):
        lines = [s for s in summary if s['experiment'] == experiment]
        figure, axes = plt.subplots(1, 4, figsize=(17, 4))
        for ax, metric, title in zip(axes, ('partial_credit', 'attribute_leakage_rate', 'citation_faithfulness', 'input_tokens'),
                                     ('Answer accuracy (partial credit)', 'Attribute leakage rate', 'Citation faithfulness', 'Input tokens')):
            modes = sorted({s['evidence_mode'] for s in lines})
            variants = sorted({s['variant'] for s in lines})
            width = 0.8/len(modes)
            for k, mode in enumerate(modes):
                vals = [next((s[metric+'_mean'] for s in lines if s['evidence_mode'] == mode and s['variant'] == v), None) or 0 for v in variants]
                errs = [next((s[metric+'_sd'] for s in lines if s['evidence_mode'] == mode and s['variant'] == v), None) or 0 for v in variants]
                ax.bar([i + k*width for i in range(len(variants))], vals, width, yerr=errs, capsize=3, label=mode)
            ax.set_xticks([i + width*(len(modes)-1)/2 for i in range(len(variants))], variants)
            ax.set_title(title, fontsize=10)
            if len(modes) > 1:
                ax.legend(fontsize=8)
        figure.suptitle(f'{experiment} — mean ± SD over repetitions (automatic Qwen judge; not human review)', fontsize=10)
        figure.tight_layout()
        figure.savefig(out/f'{experiment}_metrics.png', dpi=150)
        plt.close(figure)
    print('report', out, len(rows), 'rows')


def write_rows(path, rows):
    import csv
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: canonical(v) if isinstance(v, (dict, list)) else v for k, v in r.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=('e1', 'e2', 'e4', 'judge', 'report'))
    parser.add_argument('frozen')
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out')
    parser.add_argument('--questions', help='comma-separated question IDs to run')
    parser.add_argument('--max-reps', type=int)
    args = parser.parse_args()
    global ONLY, MAX_REPS
    if args.questions:
        ONLY = set(args.questions.split(','))
    if args.max_reps:
        MAX_REPS = args.max_reps
    load_env()
    if args.task in ('e1', 'e2', 'e4'):
        if args.repeat < 3:
            raise SystemExit('Experiments require >=3 repetitions')
        dict(e1=e1, e2=e2, e4=e4)[args.task](args.frozen, args.runs[0], args.repeat, args.workers)
    elif args.task == 'judge':
        judge(args.frozen, args.runs[0], args.workers)
    else:
        report(args.frozen, args.runs, args.out)


if __name__ == '__main__':
    main()

"""Compute every number the paper reports from the stored logs (no model calls, no gateway).

  python -X utf8 -m evaluation.runners.pilot_metrics research_evaluation_20261030
  python -X utf8 -m evaluation.runners.pilot_metrics research_evaluation_20261030 --write

Without --write it recomputes the per-question tables, compares them with results/*.csv and lists any
difference; the exit code is 1 if there is one. With --write it also rewrites results/paper_numbers.csv
and the two summary CSVs and two workbooks derived from the per-question tables.

Inputs: frozen_20261030/ (gold, evidence, retrieval, graphs), mainflow_20261030/answers_images_*.jsonl (the
last row of a question is the reported run), runs/E*/questions.jsonl, and the human-reviewed labels in
results/mainflow_manual_grades.csv and E4_manual_grades.csv. Labels, leaked claims and unsupported claims
are judgments, so they are read from those files and never computed here.

Regions are matched by content_hash, because the same document can be indexed more than once.
"""
import argparse
import csv
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ..metrics import citation_id_overlap, evidence_metrics
from .pilot_run import jsonl, key_values, normalize

K = 10
LABELS = ('correct', 'partial', 'wrong', 'no_answer')
BRIDGE = 'visual_text_shared_entity_candidate'
DUPLICATE_VERSION = 1000020  # repeat upload of DigitalTwinSoftware_en, not counted in Table 1
TABLE1 = ('CONTAINS', 'HAS_EVIDENCE', 'NEXT', 'HAS_ROW', 'HAS_CELL', 'HAS_FIGURE_TEXT', 'FIGURE_CAPTION_CANDIDATE',
          'FIGURE_NEAR_TEXT_CANDIDATE', 'ABOVE', 'LEFT_OF', 'MENTIONS', 'RELATED_PASSAGE', 'HAS_VISUAL', 'APPEARS_IN',
          'HAS_EMBEDDING')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, data):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def latest(path):
    last = {}
    for row in jsonl(path):
        last[row['question_id']] = row
    return last


def expected_keys(question):
    """Numbers and codes the reference answer must contain; a question without any has no key-value recall."""
    return list(dict.fromkeys(k for t in question['ground_truth_tuples'] for k in map(normalize, key_values(t['value'])) if len(k) > 1))


def key_value_recall(question, text):
    keys = expected_keys(question)
    return sum(k in normalize(text) for k in keys)/len(keys) if keys else None


class Data:
    def __init__(self, base):
        self.base = Path(base)
        self.out = self.base/'results'
        frozen = self.base/'frozen_20261030'
        self.questions = {q['question_id']: q for q in jsonl(frozen/'gold_questions.jsonl')}
        self.nodes = {n['evidence_id']: n for n in jsonl(frozen/'evidence.jsonl')}
        self.answers = {m: latest(self.base/f'mainflow_20261030/answers_images_{m}.jsonl') for m in ('off', 'on')}
        # An ablation can select a duplicate region from a repeat upload; it counts as the gold region with the same text.
        self.gold_by_hash = {self.nodes[i]['content_hash']: i for q in self.questions.values() for i in q['gold_evidence_ids']}

    def hashes(self, question, key='gold_evidence_ids'):
        return [self.nodes[i]['content_hash'] for i in question[key]]

    def as_gold(self, evidence_id):
        node = self.nodes.get(evidence_id)
        return self.gold_by_hash.get(node['content_hash'], evidence_id) if node else evidence_id


def retrieval_row(data, mode, qid, row):
    """Evidence metrics over the sources sent to the generator, citation metrics over the sources its claims cite."""
    q = data.questions[qid]
    gold, required = data.hashes(q), data.hashes(q, 'required_evidence_ids')
    sent = [s['content_hash'] for s in row['sources']]
    by_id = {s['evidence_id']: s['content_hash'] for s in row['sources']}
    cited = list(dict.fromkeys(by_id[i] for c in row['claims'] for i in c['evidence_ids'] if i in by_id))
    ranked = evidence_metrics(sent, gold, required, K)
    overlap = citation_id_overlap(cited, gold)
    return dict(question_id=qid, images=mode, categories='|'.join(q['categories']), gold_regions=len(gold), sources_sent=len(sent),
                query_match_sources=sum(s['expansion_reason'] == 'query_match' for s in row['sources']),
                evidence_recall_at_10=ranked['evidence_recall_at_k'], evidence_precision_at_10=ranked['evidence_precision_at_k'],
                evidence_recall_all_sent=len(set(sent) & set(gold))/len(set(gold)),
                required_evidence_recall=ranked['required_evidence_recall'], cited_sources=len(cited),
                citation_precision=overlap['precision'] if cited else None, citation_recall=overlap['recall'])


def ranking_row(data, qid, row):
    """Rank of the first gold region in the order the sources were sent; reciprocal rank is 0 if none was sent."""
    gold = set(data.hashes(data.questions[qid]))
    sent = [s['content_hash'] for s in row['sources']]
    rank = next((n+1 for n, h in enumerate(sent) if h in gold), None)
    return dict(question_id=qid, first_gold_rank=rank, hit_at_1=int(rank == 1), reciprocal_rank=round(1/rank, 4) if rank else 0,
                recall_at_3=round(len(gold & set(sent[:3]))/len(gold), 4), recall_at_5=round(len(gold & set(sent[:5]))/len(gold), 4))


def answer_recall(data, qid, row):
    return key_value_recall(data.questions[qid], (row['answer_text'] or '')+' '+' '.join(c['text'] for c in row['claims']))


def first_completed(data, name):
    best = {}
    for r in jsonl(data.base/f'runs/{name}/questions.jsonl'):
        if r['status'] == 'completed_pending_review':
            best.setdefault((r['question_id'], r['evidence_mode'], r['variant']), r)
    return best


def experiment_row(data, r):
    q = data.questions[r['question_id']]
    gold = set(q['gold_evidence_ids'])
    cited = [data.as_gold(i) for i in dict.fromkeys(i for c in r['parsed_output']['claims'] for i in c['evidence_ids'])]
    text = r['parsed_output']['answer']+' '+' '.join(c['value'] for c in r['parsed_output']['claims'])
    return dict(citation_precision=len(set(cited) & gold)/len(cited) if cited else None, citation_recall=len(set(cited) & gold)/len(gold),
                key_value_recall=key_value_recall(q, text), input_tokens=r['input_tokens'], output_tokens=r['output_tokens'],
                visual_tokens=r.get('visual_tokens'), generation_s=round(r['generation_http_ms']/1000, 1))


def experiments(data):
    e1, e4 = first_completed(data, 'E1'), first_completed(data, 'E4')
    e1_rows = [dict(question_id=q, evidence=m, format=v, **experiment_row(data, e1[q, m, v]), answer=e1[q, m, v]['parsed_output']['answer'])
               for q in ('P002', 'P018', 'P023', 'P029') for m in ('gold', 'retrieved') for v in 'ABCDE']
    e4_rows = [dict(question_id=q, input=v, **experiment_row(data, e4[q, 'gold', v]), answer=e4[q, 'gold', v]['parsed_output']['answer'])
               for q in ('P013', 'P028', 'P039', 'P040') for v in 'FKOS']

    e2 = {}
    for r in jsonl(data.base/'runs/E2/questions.jsonl'):
        if r['status'] == 'completed_pending_review':
            e2.setdefault((r['question_id'], r['variant']), r)
    e2_rows = []
    for qid in ('P003', 'P007', 'P022', 'P027'):
        for v in 'ABCD':
            r = e2[qid, v]
            reasons = Counter(t['reason'] for t in r['graph_trace'])
            text = r['parsed_output']['answer']+' '+' '.join(c['value'] for c in r['parsed_output']['claims'])
            e2_rows.append(dict(question_id=qid, mode=v, seeds=len(r['seed_evidence_ids']), added_nodes=r['added_nodes'],
                                admitted=reasons.get('admitted', 0), rejected=sum(n for k, n in reasons.items() if k != 'admitted'),
                                rejection_reasons=json.dumps({k: n for k, n in reasons.items() if k != 'admitted'}),
                                evidence_recall_at_10=r['evidence_recall_at_k'], evidence_precision_at_10=r['evidence_precision_at_k'],
                                required_evidence_recall=r['required_evidence_recall'], input_tokens=r['input_tokens'],
                                graph_latency_ms=round(r['graph_latency_ms'], 1), key_value_recall=key_value_recall(data.questions[qid], text),
                                answer=r['parsed_output']['answer']))

    lowest = {}
    for r in jsonl(data.base/'runs/E1/questions.jsonl'):
        if r['status'] != 'failed':
            key = r['question_id'], r['evidence_mode'], r['variant']
            if key not in lowest or r['repetition'] < lowest[key]['repetition']:
                lowest[key] = r
    table_rows = []
    for qid in ('P006', 'P008', 'P013', 'P015', 'P048'):
        gold = {data.nodes[i]['content_hash'] for i in data.questions[qid]['gold_evidence_ids']}
        for mode in ('gold', 'retrieved'):
            for v in 'ABCDE':
                r = lowest[qid, mode, v]
                parsed = r['parsed_output'] or {}
                text = (parsed.get('answer') or '')+' '+' '.join(f"{c.get('subject', '')} {c.get('attribute', '')} {c.get('value', '')}" for c in parsed.get('claims', []))
                by_id = {i: data.nodes[i]['content_hash'] for i in r['selected_evidence_ids']}
                cited = {by_id.get(i) for i in (r.get('cited_evidence_ids') or [])}
                table_rows.append(dict(question_id=qid, evidence_mode=mode, format=v, citation_recall=len(gold & cited)/len(gold),
                                       key_value_recall=key_value_recall(data.questions[qid], text), input_tokens=r['input_tokens'],
                                       answer=parsed.get('answer', '')))
    return dict(E1_examples=e1_rows, E2_examples=e2_rows, E4_examples=e4_rows, E1_table_examples=table_rows)


def same(new, old):
    if new is None or new == '':
        return old in ('', None)
    try:
        return abs(float(new)-float(old)) < 1e-3
    except (TypeError, ValueError):
        return str(new).strip() == str(old).strip()


def compare(name, new, stored, key):
    """List differences between recomputed rows and the stored CSV; columns not recomputed are ignored."""
    old = {tuple(r[k] for k in key): r for r in stored}
    differences = []
    for row in new:
        match = old.get(tuple(str(row[k]) for k in key))
        if match is None:
            differences.append((name, tuple(row[k] for k in key), 'missing from the stored file', '', ''))
            continue
        for field, value in row.items():
            if field in match and not same(value, match[field]):
                differences.append((name, tuple(row[k] for k in key), field, value, match[field]))
    return differences


def paper_numbers(data, retrieval, ranking, grades):
    """One row per number in the paper, with its numerator, denominator and source file."""
    out = []

    def add(name, value, numerator='', denominator='', source=''):
        out.append(dict(number=name, value=round(value, 3) if isinstance(value, float) else value,
                        numerator=numerator, denominator=denominator, source=source))

    questions = data.questions
    off = data.answers['off']
    g = {r['question_id']: r for r in grades if r['images'] == 'off'}
    rt = {r['question_id']: r for r in retrieval if r['images'] == 'off'}
    grades_file, answers_file = 'results/mainflow_manual_grades.csv', 'mainflow_20261030/answers_images_off.jsonl'
    mean = lambda rows, field: st.mean(float(r[field]) for r in rows if r[field] not in ('', None))

    add('questions answered', sum(r['generation_status'] == 'generated' for r in off.values()), '', len(off), answers_file)
    hits = sum(r['hit_at_1'] for r in ranking)
    add('Hit@1', hits/len(ranking), hits, len(ranking), answers_file)
    add('MRR', mean(ranking, 'reciprocal_rank'), '', len(ranking), answers_file)
    for label, field in (('Evidence recall@10', 'evidence_recall_at_10'), ('Evidence precision@10', 'evidence_precision_at_10'),
                         ('Required-evidence recall', 'required_evidence_recall'), ('Citation precision', 'citation_precision'),
                         ('Citation recall', 'citation_recall')):
        add(label, mean(rt.values(), field), '', sum(r[field] is not None for r in rt.values()), answers_file)
    add('Precision@10 ceiling', st.mean(min(r['gold_regions'], K)/K for r in rt.values()), '', len(rt), 'frozen_20261030/gold_questions.jsonl')

    counts = Counter(r['label'] for r in g.values())
    for label in LABELS:
        add('Answers labelled '+label, counts[label], '', len(g), grades_file)
    add('Answer accuracy (partial = 0.5)', (counts['correct']+.5*counts['partial'])/len(g), counts['correct']+.5*counts['partial'], len(g), grades_file)
    add('Strict accuracy', counts['correct']/len(g), counts['correct'], len(g), grades_file)
    claims = sum(int(r['claims']) for r in g.values())
    leaked = sum(int(r['leaked_claims']) for r in g.values())
    unsupported = sum(int(r['unsupported_claims']) for r in g.values())
    add('Attribute leakage rate', leaked/claims, leaked, claims, grades_file)
    add('Citation faithfulness', (claims-unsupported)/claims, claims-unsupported, claims, grades_file)

    recalls = [v for v in (answer_recall(data, q, r) for q, r in off.items()) if v is not None]
    add('Key-value recall', st.mean(recalls), '', len(recalls), answers_file)
    add('Key-value questions complete', sum(v == 1 for v in recalls), '', len(recalls), answers_file)

    for category in ('table', 'footnote', 'condition'):
        members = [q for q, v in questions.items() if category in v['categories']]
        add(f'Accuracy, {category} questions', st.mean(float(g[q]['partial_credit']) for q in members), '', len(members), grades_file)
        add(f'Required-evidence recall, {category} questions', st.mean(rt[q]['required_evidence_recall'] for q in members), '', len(members), answers_file)

    pairs = [(q, i) for q, v in questions.items() for i in v['gold_evidence_ids']]
    add('Gold question-region pairs', len(pairs), '', '', 'frozen_20261030/gold_questions.jsonl')
    add('Gold distinct regions', len({i for _, i in pairs}), '', '', 'frozen_20261030/gold_questions.jsonl')
    tables = [(q, i) for q, i in pairs if data.nodes[i]['layout_type'] == 'table']
    add('Gold question-table pairs', len(tables), '', '', 'frozen_20261030/evidence.jsonl')
    add('Gold distinct tables', len({i for _, i in tables}), '', '', 'frozen_20261030/evidence.jsonl')
    reached = sum(data.nodes[i]['content_hash'] in {s['content_hash'] for s in off[q]['sources']} for q, i in tables)
    add('Gold table pairs that reached the generator', reached, '', len(tables), answers_file)

    frozen = json.loads((data.base/'frozen_20261030/retrieval.json').read_text(encoding='utf-8'))['questions']
    frozen = frozen if isinstance(frozen, list) else list(frozen.values())
    top = [[s['streams'] == ['colpali'] for s in q['ranked'][:K]] for q in frozen]
    source = 'frozen_20261030/retrieval.json'
    add('ColPali-only sources in the first ten', sum(map(sum, top)), '', sum(map(len, top)), source)
    add('Questions with a ColPali-only source', sum(map(any, top)), '', len(top), source)

    bridge = [sum(s.get('expansion_reason') == BRIDGE for s in r['sources']) for r in off.values()]
    add('Visual bridge passages', sum(bridge), '', '', answers_file)
    add('Questions with a visual bridge passage', sum(v > 0 for v in bridge), '', len(bridge), answers_file)

    pilot = {int(r['version_id']) for r in read_csv(data.base/'documents/DOCUMENT_INDEX.csv') if r['role'] == 'pilot'} - {DUPLICATE_VERSION}
    graphs = [d for d in json.loads((data.base/'frozen_20261030/graphs.json').read_text(encoding='utf-8')) if d['version_id'] in pilot]
    source = 'frozen_20261030/graphs.json'
    add('Table 1 documents', len(graphs), '', '', source)
    add('Table 1 nodes', sum(len(d['nodes']) for d in graphs), '', '', source)
    add('Table 1 edges', sum(len(d['edges']) for d in graphs), '', '', source)
    add('Table 1 median nodes per document', st.median(len(d['nodes']) for d in graphs), '', '', source)
    add('Table 1 median edges per document', st.median(len(d['edges']) for d in graphs), '', '', source)
    add('Table 1 entity nodes', sum(n.get('kind') == 'entity' for d in graphs for n in d['nodes']), '', '', source)
    edges = Counter(e.get('type') for d in graphs for e in d['edges'])
    for edge in TABLE1:
        add('Table 1 '+edge, edges[edge], '', '', source)

    cells = defaultdict(list)
    for name in ('E1', 'E2', 'E4'):
        for r in jsonl(data.base/f'runs/{name}/questions.jsonl'):
            if r.get('status') != 'failed':
                cells[name, r['question_id'], r.get('evidence_mode'), r.get('variant')].append(r.get('output'))
    repeated = [v for v in cells.values() if len(v) > 1]
    add('Repeated configurations with identical output', sum(len(set(map(str, v))) == 1 for v in repeated), '', len(repeated), 'runs/E*/questions.jsonl')
    return out


def summaries(retrieval, grades):
    measures = ('evidence_recall_at_10', 'evidence_precision_at_10', 'evidence_recall_all_sent', 'required_evidence_recall',
                'citation_precision', 'citation_recall', 'sources_sent', 'cited_sources')
    retrieval_summary, grade_summary = [], []
    for mode in ('off', 'on'):
        rows = [r for r in retrieval if r['images'] == mode]
        line = dict(images=mode, questions=len(rows))
        for m in measures:
            values = [r[m] for r in rows if r[m] is not None]
            line[m+'_mean'], line[m+'_sd'] = round(st.mean(values), 3), round(st.stdev(values), 3)
        retrieval_summary.append(line)
        g = [r for r in grades if r['images'] == mode]
        counts = Counter(r['label'] for r in g)
        claims = sum(int(r['claims']) for r in g)
        grade_summary.append(dict(images=mode, answers=len(g), correct=counts['correct'], partial=counts['partial'], wrong=counts['wrong'],
                                  no_answer=counts['no_answer'], answer_accuracy=round(st.mean(float(r['partial_credit']) for r in g), 3),
                                  strict_accuracy=round(st.mean(int(r['accuracy']) for r in g), 3),
                                  attribute_leakage_rate=round(sum(int(r['leaked_claims']) for r in g)/claims, 3),
                                  answers_with_leakage=sum(int(r['leaked_claims']) > 0 for r in g),
                                  citation_faithfulness=round(1-sum(int(r['unsupported_claims']) for r in g)/claims, 3), claims=claims))
    return retrieval_summary, grade_summary


def workbook(path, sheets):
    from openpyxl import Workbook
    book = Workbook()
    book.remove(book.active)
    for name, data in sheets:
        sheet = book.create_sheet(name)
        sheet.append(list(data[0]))
        for row in data:
            sheet.append(list(row.values()))
    book.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('base', help='The research_evaluation_20261030 folder')
    parser.add_argument('--write', action='store_true', help='Rewrite paper_numbers.csv and the derived summaries')
    args = parser.parse_args()
    data = Data(args.base)
    out = data.out

    retrieval, ranking = [], []
    for mode, answers in data.answers.items():
        for qid, row in sorted(answers.items()):
            if row['generation_status'] == 'generated':
                retrieval.append(retrieval_row(data, mode, qid, row))
    for qid, row in sorted(data.answers['off'].items()):
        ranking.append(ranking_row(data, qid, row))
    grades = read_csv(out/'mainflow_manual_grades.csv')
    examples = experiments(data)

    answers_stored = read_csv(out/'mainflow_answers.csv')
    recalls = [dict(question_id=q, images=m, key_value_recall=answer_recall(data, q, r)) for m, a in data.answers.items()
               for q, r in a.items() if r['generation_status'] == 'generated']
    differences = compare('mainflow_retrieval_citation', retrieval, read_csv(out/'mainflow_retrieval_citation.csv'), ('question_id', 'images'))
    differences += compare('mainflow_ranking_metrics', ranking, read_csv(out/'mainflow_ranking_metrics.csv'), ('question_id',))
    differences += compare('mainflow_answers', recalls, answers_stored, ('question_id', 'images'))
    keys = dict(E1_examples=('question_id', 'evidence', 'format'), E2_examples=('question_id', 'mode'), E4_examples=('question_id', 'input'),
                E1_table_examples=('question_id', 'evidence_mode', 'format'))
    for name, rows in examples.items():
        differences += compare(name, rows, read_csv(out/f'{name}.csv'), keys[name])

    numbers = paper_numbers(data, retrieval, ranking, grades)
    for row in numbers:
        detail = f" ({row['numerator']}/{row['denominator']})" if row['numerator'] != '' else f" (n={row['denominator']})" if row['denominator'] != '' else ''
        print(f"{row['number']}: {row['value']}{detail}")
    print(f'\n{len(differences)} difference(s) from the stored per-question tables')
    for name, key, field, new, old in differences:
        print(' ', name, key, field, 'recomputed', new, 'stored', old)

    if args.write:
        write_csv(out/'paper_numbers.csv', numbers)
        retrieval_summary, grade_summary = summaries(retrieval, grades)
        write_csv(out/'mainflow_retrieval_citation_summary.csv', retrieval_summary)
        write_csv(out/'mainflow_manual_grades_summary.csv', grade_summary)
        workbook(out/'recall_precision.xlsx', [('retrieval_summary', retrieval_summary), ('retrieval_per_question', retrieval),
                                               ('E2_examples', examples['E2_examples']), ('E1_examples', examples['E1_examples']),
                                               ('E4_examples', examples['E4_examples'])])
        workbook(out/'mainflow_results.xlsx', [('summary', read_csv(out/'mainflow_summary.csv')), ('answers', answers_stored),
                                               ('failures', read_csv(out/'mainflow_failures.csv'))])
        print('wrote', out/'paper_numbers.csv')
    return 1 if differences else 0


if __name__ == '__main__':
    sys.exit(main())

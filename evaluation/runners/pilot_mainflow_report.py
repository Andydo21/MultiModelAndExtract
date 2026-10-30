"""Judge and report main-flow answers (images on vs off) against the pilot gold set.

  python -X utf8 -m evaluation.runners.pilot_mainflow_report FROZEN MAINFLOW_DIR [--workers 8]

Writes judgments.jsonl (resumable) and report/{runs,summary,paired}.csv, results.xlsx, mainflow.png and
paper/mainflow_table.tex. Labels come from the automatic Qwen judge (same model as the generator), not people.
"""
import argparse
import concurrent.futures as cf
import json
import statistics
import threading
from pathlib import Path

from ..io import digest
from .common import Gateway
from .pilot_run import CONFIG, JUDGE_SYSTEM, ROOT, judge_schema, jsonl, key_values, load_env, normalize, write_rows

MODES = ('off', 'on')


def judge_prompt(q, row, gold_text):
    by_id = {s['evidence_id']: s['text'] for s in row['sources']}
    cited = list(dict.fromkeys(i for c in row['claims'] for i in c['evidence_ids']))
    lines = ['QUESTION: '+q['question'], 'REFERENCE ANSWER: '+q['reference_answer'],
             'GOLD FACTS: '+json.dumps(q['ground_truth_tuples'], ensure_ascii=False), 'GOLD EVIDENCE:']
    lines += ['['+i+'] '+gold_text[i] for i in q['gold_evidence_ids']]
    lines += ['CITED EVIDENCE:'] + ['['+i+'] '+by_id.get(i, '(not among supplied sources)') for i in cited]
    lines += ['MODEL ANSWER: '+(row['answer_text'] or ''), 'MODEL CLAIMS:']
    lines += [f'{n}. '+json.dumps(dict(claim=c['text'], evidence_ids=c['evidence_ids']), ensure_ascii=False)
              for n, c in enumerate(row['claims'])]
    return [dict(role='system', content=JUDGE_SYSTEM), dict(role='user', content='\n'.join(lines))]


def latest(path):
    rows = {}
    for r in jsonl(path):
        rows[r['question_id']] = r
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('frozen')
    parser.add_argument('mainflow')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    load_env()
    frozen, base = Path(args.frozen), Path(args.mainflow)
    questions = {q['question_id']: q for q in jsonl(frozen/'gold_questions.jsonl')}
    nodes = {n['evidence_id']: n for n in jsonl(frozen/'evidence.jsonl')}
    gold_text = {i: nodes[i]['text'] for q in questions.values() for i in q['gold_evidence_ids']}
    gold_hash = {i: nodes[i]['content_hash'] for i in gold_text}
    answers = {m: latest(base/f'answers_images_{m}.jsonl') for m in MODES}
    judged_path = base/'judgments.jsonl'
    judged = {(j['question_id'], j['images']): j for j in jsonl(judged_path)} if judged_path.exists() else {}
    gateway, lock = Gateway(CONFIG['generator']), threading.Lock()

    def work(item):
        mode, row = item
        q = questions[row['question_id']]
        result = dict(question_id=row['question_id'], images=mode, judge_model=CONFIG['generator']['model'],
                      judge='automatic_qwen_same_model')
        if row['generation_status'] != 'generated' or not row['claims']:
            result.update(status='judged', answer_label='no_answer', verdict=None)
        else:
            chat = judge_prompt(q, row, gold_text)
            body = dict(model=CONFIG['generator']['model'], messages=chat, temperature=0.0, top_p=1.0, max_tokens=1024,
                        seed=CONFIG['generator']['seed'], response_format=judge_schema(len(row['claims'])))
            try:
                response, _ = gateway.generate(body)
                choice = response['choices'][0]
                if choice.get('finish_reason') != 'stop':
                    raise ValueError('Judge output incomplete')
                verdict = json.loads(choice['message']['content'])
                result.update(status='judged', verdict=verdict, answer_label=verdict['answer_label'], judge_prompt_sha256=digest(chat))
            except Exception as error:
                result.update(status='failed', error=f'{type(error).__name__}: {str(error)[:200]}')
        with lock, judged_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
        return result

    todo = [(m, r) for m in MODES for qid, r in answers[m].items() if judged.get((qid, m), {}).get('status') != 'judged']
    print(f'judging {len(todo)} answers', flush=True)
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for result in pool.map(work, todo):
            judged[result['question_id'], result['images']] = result

    rows = []
    for mode in MODES:
        for qid, r in sorted(answers[mode].items()):
            q, j = questions[qid], judged.get((qid, mode), {})
            verdict = j.get('verdict') or {}
            flags = {c['index']: c for c in verdict.get('claims', [])}
            claims = r['claims']
            cited = [i for i, c in enumerate(claims) if c['evidence_ids']]
            text = normalize((r['answer_text'] or '') + ' ' + ' '.join(c['text'] for c in claims))
            keys = list(dict.fromkeys(k for t in q['ground_truth_tuples'] for k in map(normalize, key_values(t['value'])) if len(k) > 1))
            sent_hashes = {s.get('content_hash') for s in r['sources']}
            label = j.get('answer_label')
            rows.append(dict(question_id=qid, images=mode, categories='|'.join(q['categories']),
                generation_status=r['generation_status'], images_requested_by_policy=r['images_requested_by_policy'],
                images_sent=r['images_sent'], answer_label=label,
                accuracy=None if label is None else int(label == 'correct'),
                partial_credit=None if label is None else dict(correct=1, partial=0.5).get(label, 0),
                attribute_leakage_rate=(sum(bool(flags.get(i, {}).get('attribute_leakage')) for i in range(len(claims)))/len(claims)) if claims and verdict else None,
                citation_faithfulness=(sum(bool(flags.get(i, {}).get('supported_by_cited')) for i in cited)/len(cited)) if cited and verdict else None,
                key_value_recall=sum(k in text for k in keys)/len(keys) if keys else None,
                gold_sources_sent=sum(gold_hash[i] in sent_hashes for i in q['gold_evidence_ids'])/len(q['gold_evidence_ids']),
                sources_sent=len(r['sources']), claims=len(claims), retrieval_s=r['retrieval_ms']/1000,
                generation_s=r['generation_ms']/1000, answer_text=r['answer_text']))
    out = base/'report'
    out.mkdir(exist_ok=True)
    metrics = ('partial_credit', 'accuracy', 'attribute_leakage_rate', 'citation_faithfulness', 'key_value_recall',
               'gold_sources_sent', 'generation_s', 'retrieval_s')
    summary = []
    for mode in MODES:
        items = [r for r in rows if r['images'] == mode]
        line = dict(images=mode, questions=len(items), generated=sum(r['generation_status'] == 'generated' for r in items),
                    images_sent=sum(bool(r['images_sent']) for r in items),
                    **{k: sum(r['answer_label'] == k for r in items) for k in ('correct', 'partial', 'wrong', 'no_answer')})
        for metric in metrics:
            values = [r[metric] for r in items if r[metric] is not None]
            line[metric+'_mean'] = statistics.mean(values) if values else None
            line[metric+'_n'] = len(values)
        summary.append(line)
    from scipy.stats import wilcoxon
    paired = []
    for metric in metrics[:5]:
        pairs = [(a[metric], b[metric]) for a in rows if a['images'] == 'off' for b in rows
                 if b['images'] == 'on' and b['question_id'] == a['question_id'] and a[metric] is not None and b[metric] is not None]
        diffs = [b - a for a, b in pairs]
        test = dict(metric=metric, questions=len(pairs), mean_off=statistics.mean(a for a, _ in pairs) if pairs else None,
                    mean_on=statistics.mean(b for _, b in pairs) if pairs else None, on_better=sum(d > 0 for d in diffs),
                    off_better=sum(d < 0 for d in diffs), ties=sum(d == 0 for d in diffs), p_wilcoxon=None)
        if len(pairs) >= 5 and any(diffs):
            test['p_wilcoxon'] = float(wilcoxon([a for a, _ in pairs], [b for _, b in pairs], zero_method='zsplit').pvalue)
        paired.append(test)
    for name, data in (('runs', rows), ('summary', summary), ('paired', paired)):
        path = out/(name+'.csv')
        path.unlink(missing_ok=True)
        write_rows(path, data)
    from openpyxl import Workbook
    book = Workbook()
    book.remove(book.active)
    for name, data in (('summary', summary), ('paired', paired), ('runs', rows)):
        sheet = book.create_sheet(name)
        fields = list(dict.fromkeys(k for r in data for k in r))
        sheet.append(fields)
        for r in data:
            sheet.append([r.get(k) for k in fields])
    book.save(out/'results.xlsx')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, metric, title in zip(axes, ('partial_credit', 'attribute_leakage_rate', 'citation_faithfulness', 'generation_s'),
                                 ('Answer accuracy (partial credit)', 'Attribute leakage rate', 'Citation faithfulness', 'Generation time (s)')):
        ax.bar(['images off', 'images on'], [next(s[metric+'_mean'] for s in summary if s['images'] == m) or 0 for m in MODES],
               color=['#4C78A8', '#F58518'])
        ax.set_title(title, fontsize=10)
    figure.suptitle('Production main flow, 50 pilot questions, 1 run each (automatic Qwen judge; not human review)', fontsize=10)
    figure.tight_layout()
    figure.savefig(out/'mainflow.png', dpi=150)
    fmt = lambda v, d=2: r'\tbd' if v is None else f'{v:.{d}f}'
    lines = [r'\begin{table}[t]', r'\centering\footnotesize',
             r'\caption{Production main flow on the 50 pilot questions, one run per setting (automatic Qwen judge, not human review). '
             r'``Images on'' attaches crops only when the production policy asks for them.}', r'\label{tab:mainflow}',
             r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{@{}lcccccc@{}}', r'\toprule',
             r'Setting & Acc. & Strict & ALR $\downarrow$ & Faith. & KV rec. & Gen.\ s \\', r'\midrule']
    for s in summary:
        lines.append(f'Images {s["images"]} ({s["images_sent"]} sent) & {fmt(s["partial_credit_mean"])} & {fmt(s["accuracy_mean"])} & '
                     f'{fmt(s["attribute_leakage_rate_mean"])} & {fmt(s["citation_faithfulness_mean"])} & '
                     f'{fmt(s["key_value_recall_mean"])} & {fmt(s["generation_s_mean"], 1)} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    (ROOT/'paper'/'mainflow_table.tex').write_text('% Generated by evaluation.runners.pilot_mainflow_report\n' + '\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(paired, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()

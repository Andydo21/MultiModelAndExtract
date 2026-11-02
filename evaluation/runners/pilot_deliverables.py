"""Build the hand-over tables and charts from the stored pilot results (no model calls).

  python -X utf8 -m evaluation.runners.pilot_deliverables research_evaluation_20261030

Writes into results/: SUMMARY_TABLES.xlsx (summary with mean +- SD, label counts, full comparison,
error cases, worked examples), error_cases.csv, summary_mean_sd.csv and charts/*.png.
SD is taken over questions: every configuration was run once.
"""
import csv
import statistics as st
import sys
from collections import Counter
from pathlib import Path

LABELS = ('correct', 'partial', 'wrong', 'no_answer')


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def number(value):
    return None if value in ('', None) else float(value)


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None, 0
    return round(st.mean(values), 3), round(st.stdev(values), 3) if len(values) > 1 else None, len(values)


def main(base):
    base = Path(base)
    out = base/'results'
    grades = rows(out/'mainflow_manual_grades.csv')
    retrieval = rows(out/'mainflow_retrieval_citation.csv')
    ranking = {r['question_id']: r for r in rows(out/'mainflow_ranking_metrics.csv')}
    answers = rows(out/'mainflow_answers.csv')
    failures = rows(out/'mainflow_failures.csv')
    e1, e2, e4 = (rows(out/f'{n}_examples.csv') for n in ('E1', 'E2', 'E4'))
    e4_grades = rows(out/'E4_manual_grades.csv')

    summary = []
    per_question = {}
    for mode in ('off', 'on'):
        g = [r for r in grades if r['images'] == mode]
        rt = [r for r in retrieval if r['images'] == mode]
        kv = [number(r['key_value_recall']) for r in answers if r['images'] == mode and r['generation_status'] == 'generated']
        metrics = dict(
            answer_accuracy=[number(r['answer_accuracy']) for r in g],
            strict_accuracy=[number(r['strict_accuracy']) for r in g],
            attribute_leakage_rate=[number(r['attribute_leakage_rate']) for r in g],
            citation_faithfulness=[number(r['citation_faithfulness']) for r in g],
            key_value_recall=kv,
            evidence_recall_at_10=[number(r['evidence_recall_at_10']) for r in rt],
            evidence_precision_at_10=[number(r['evidence_precision_at_10']) for r in rt],
            required_evidence_recall=[number(r['required_evidence_recall']) for r in rt],
            citation_precision=[number(r['citation_precision']) for r in rt],
            citation_recall=[number(r['citation_recall']) for r in rt])
        if mode == 'off':
            metrics.update(hit_at_1=[number(r['hit_at_1']) for r in ranking.values()],
                           mrr=[number(r['reciprocal_rank']) for r in ranking.values()])
        counts = Counter(r['label'] for r in g)
        if mode == 'off':
            per_question = metrics
        for name, values in metrics.items():
            m, sd, n = mean_sd(values)
            used = 'yes' if mode == 'off' else 'no (zero-shot VLM; kept for the record)'
            if mode == 'off' and name in ('attribute_leakage_rate', 'citation_faithfulness'):
                used = 'no (mean over questions; the paper uses the share of claims)'
            summary.append(dict(setting='text_only' if mode == 'off' else 'image_policy', metric=name, mean=m, sd=sd, n=n, used_in_paper=used))
        claims = sum(int(r['claims']) for r in g)
        for name, value in (('attribute_leakage_rate_of_claims', sum(int(r['leaked_claims']) for r in g)/claims),
                            ('citation_faithfulness_of_claims', 1-sum(int(r['unsupported_claims']) for r in g)/claims)):
            summary.append(dict(setting='text_only' if mode == 'off' else 'image_policy', metric=name, mean=round(value, 3), sd=None, n=claims,
                                used_in_paper='yes' if mode == 'off' else 'no'))
        for label in LABELS:
            summary.append(dict(setting='text_only' if mode == 'off' else 'image_policy', metric='count_'+label,
                                mean=counts.get(label, 0), sd=None, n=len(g), used_in_paper='yes' if mode == 'off' else 'no'))

    questions = {r['question_id']: r['question'] for r in answers}
    errors = []
    for r in grades:
        if r['label'] != 'correct' or int(r['leaked_claims']) or int(r['unsupported_claims']):
            kind = ('leakage' if int(r['leaked_claims']) else 'unsupported_content' if r['label'] == 'wrong'
                    else 'abstained' if r['label'] == 'no_answer' else 'missing_detail')
            errors.append(dict(question_id=r['question_id'], setting='text_only' if r['images'] == 'off' else 'image_policy',
                               label=r['label'], error_type=kind, leaked_claims=r['leaked_claims'],
                               unsupported_claims=r['unsupported_claims'], note=r['note'], question=questions.get(r['question_id'], '')))
    for r in failures:
        errors.append(dict(question_id=r['question_id'], setting='text_only' if r['images'] == 'off' else 'image_policy',
                           label='not_generated', error_type=r['failure_state'], leaked_claims='', unsupported_claims='',
                           note='generator offline during the run' if r['failure_state'] == 'gateway_unavailable'
                           else 'withheld by the production citation check', question=r['question']))

    comparison = []
    for name, data, key in (('E1', e1, 'format'), ('E2', e2, 'mode'), ('E4', e4, 'input')):
        groups = {}
        for r in data:
            groups.setdefault((r.get('evidence', ''), r[key]), []).append(r)
        for (evidence, variant), items in sorted(groups.items()):
            line = dict(experiment=name, evidence=evidence or ('retrieved' if name == 'E2' else 'gold'), variant=variant,
                        questions=len(items))
            for metric in ('citation_precision', 'citation_recall', 'key_value_recall', 'input_tokens', 'visual_tokens',
                           'added_nodes', 'evidence_recall_at_10', 'evidence_precision_at_10', 'required_evidence_recall',
                           'graph_latency_ms', 'generation_s'):
                values = [number(r[metric]) for r in items if metric in r]
                if values:
                    line[metric+'_mean'], line[metric+'_sd'], _ = mean_sd(values)
            if name == 'E4':
                g = [x for x in e4_grades if x['input'] == variant]
                line['accuracy_mean'] = round(st.mean({'correct': 1, 'partial': .5}.get(x['label'], 0) for x in g), 3)
                line['attribute_leakage_rate'] = round(sum(int(x['leaked_claims']) for x in g)/sum(int(x['claims']) for x in g), 3)
            comparison.append(line)

    def write(name, data):
        fields = list(dict.fromkeys(k for r in data for k in r))
        with (out/name).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    write('summary_mean_sd.csv', summary)
    write('error_cases.csv', errors)
    write('comparison_full.csv', comparison)

    from openpyxl import Workbook
    from openpyxl.styles import Font
    book = Workbook()
    book.remove(book.active)
    notes = [['Pilot evaluation 2026-10-30: 50 questions, 20 documents, one run per configuration; SD over questions.'],
             ['Paper reports the text-only main flow and the E1/E2/E4 worked examples. Image-policy rows are kept for the record.'],
             ['Gold questions, reference answers, tuples and evidence regions: written by Claude from the extracted text, light text check, accepted by the project owner (frozen_20261030/gold_questions.jsonl).'],
             ['Answer labels: one LLM-assisted reading pass (Claude) against reference answers, tuples and gold evidence; not independent (column grader of sheet manual_grades).'],
             ['Text-only main flow: all 50 answered; P047-P050 and P007 come from a second run.'],
             ['Image policy: 45 of 50 answered; not completed: P047-P050 (generator offline), P007 (withheld by citation check); see sheet error_cases.'],
             ['In summary_mean_sd, *_of_claims rows are shares of all claims (used in the paper); the rows without that suffix are means over questions.'],
             ['Not used and not included in this repository: mainflow_20261030/judgments.jsonl and report/ (automatic Qwen judge stopped after 14 of 100 answers).']]
    for title, data in (('README', None), ('summary_mean_sd', summary), ('comparison_full', comparison), ('error_cases', errors),
                        ('manual_grades', grades), ('ranking', list(ranking.values())), ('E1_examples', e1),
                        ('E2_examples', e2), ('E4_examples', e4)):
        sheet = book.create_sheet(title)
        if data is None:
            for line in notes:
                sheet.append(line)
            continue
        fields = list(dict.fromkeys(k for r in data for k in r))
        sheet.append(fields)
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        for r in data:
            sheet.append([r.get(k) for k in fields])
    book.save(out/'SUMMARY_TABLES.xlsx')
    charts(out, grades, summary, e2, e4, e4_grades, per_question)
    print('wrote', out/'SUMMARY_TABLES.xlsx', len(summary), 'summary rows', len(errors), 'error cases', len(comparison), 'comparison rows')


def bootstrap_ci(values, draws=5000, seed=1729):
    """95% percentile interval of the mean over questions."""
    import random
    values = [v for v in values if v is not None]
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return means[int(.025*draws)], means[int(.975*draws)-1]


def charts(out, grades, summary, e2, e4, e4_grades, per_question):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = out/'charts'
    folder.mkdir(exist_ok=True)
    text = {r['metric']: r for r in summary if r['setting'] == 'text_only'}

    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    names = ['hit_at_1', 'mrr', 'evidence_recall_at_10', 'required_evidence_recall', 'citation_precision', 'citation_recall',
             'answer_accuracy', 'strict_accuracy', 'citation_faithfulness', 'key_value_recall']
    labels = ['Hit@1', 'MRR', 'Recall@10', 'Req. recall', 'Cit. P', 'Cit. R', 'Accuracy', 'Strict', 'Faithful.', 'KV recall']
    means = [text[n]['mean'] for n in names]
    intervals = [bootstrap_ci(per_question[n]) for n in names]
    errors = [[m-lo for m, (lo, _) in zip(means, intervals)], [hi-m for m, (_, hi) in zip(means, intervals)]]
    axes[0].bar(labels, means, yerr=errors, capsize=3, color=['#4C78A8']*6 + ['#59A14F']*4)
    for i, m in enumerate(means):
        axes[0].text(i, 0.03, f'{m:.2f}', ha='center', fontsize=7, color='white')
    axes[0].set_ylim(0, 1.05)
    questions_answered = len(per_question['answer_accuracy'])
    axes[0].set_title(f'Main flow, text-only: mean and 95% bootstrap CI ({questions_answered} questions)', fontsize=10)
    axes[0].tick_params(axis='x', rotation=45, labelsize=8)
    counts = [text['count_'+l]['mean'] for l in LABELS]
    axes[1].bar(['Correct', 'Partial', 'Wrong', 'No answer'], counts, color=['#59A14F', '#EDC948', '#E15759', '#BAB0AC'])
    for i, c in enumerate(counts):
        axes[1].text(i, c + .5, str(int(c)), ha='center', fontsize=9)
    axes[1].set_title(f'Answer labels ({questions_answered} questions)', fontsize=10)
    figure.tight_layout()
    figure.savefig(folder/'mainflow_text_only.png', dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    modes = ['A', 'B', 'C', 'D']
    questions = sorted({r['question_id'] for r in e2})
    for q in questions:
        items = {r['mode']: r for r in e2 if r['question_id'] == q}
        axes[0].plot(modes, [float(items[m]['input_tokens']) for m in modes], marker='o', label=q)
        axes[1].plot(modes, [float(items[m]['evidence_recall_at_10']) for m in modes], marker='o', label=q)
    axes[0].set_title('E2: prompt tokens by expansion mode', fontsize=10)
    axes[1].set_title('E2: evidence recall@10 by expansion mode', fontsize=10)
    axes[1].set_ylim(0, 1.1)
    for ax in axes:
        ax.set_xticks(range(4), ['A none', 'B prev/next', 'C structural', 'D scope'])
        ax.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(folder/'e2_expansion.png', dpi=160)
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(5.5, 3.8))
    names = {'F': ('Full-page image', (-30, 8)), 'K': ('Region crops', (6, 6)),
             'O': ('OCR text', (-10, -16)), 'S': ('OCR + caption pods', (8, 8))}
    for v, (name, offset) in names.items():
        tokens = st.mean(float(r['input_tokens']) for r in e4 if r['input'] == v)
        g = [x for x in e4_grades if x['input'] == v]
        acc = st.mean({'correct': 1, 'partial': .5}.get(x['label'], 0) for x in g)
        ax.scatter(tokens, acc, s=60)
        ax.annotate(name, (tokens, acc), textcoords='offset points', xytext=offset, fontsize=8)
    ax.set_xlabel('prompt tokens (mean)')
    ax.set_ylabel('answer accuracy')
    ax.set_ylim(0, 1)
    ax.set_title('E4: accuracy vs prompt tokens (4 questions)', fontsize=10)
    figure.tight_layout()
    figure.savefig(folder/'e4_accuracy_tokens.png', dpi=160)
    plt.close(figure)


if __name__ == '__main__':
    main(sys.argv[1])

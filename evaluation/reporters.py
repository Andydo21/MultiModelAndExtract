"""Derived CSV/Markdown/LaTeX and question-paired statistics from verified raw logs."""
import math
import statistics
from collections import defaultdict

from .io import new_output, read_json, read_records, sha_file, write_csv, write_json
from .metrics import review_metrics

METRICS=('e2e_ms','generation_http_ms','graph_latency_ms','input_tokens','output_tokens',
         'evidence_recall_at_k','evidence_precision_at_k','required_evidence_recall',
         'accuracy','attribute_leakage_rate','citation_faithfulness')


def paired_test(differences,method='wilcoxon',minimum_questions=5):
    if method not in ('wilcoxon','paired_t'):
        raise ValueError('Unknown paired test')
    if len(differences)<minimum_questions:
        return dict(method=method,n_question_pairs=len(differences),p_value=None,reason='insufficient_independent_question_pairs')
    if all(d==0 for d in differences):
        return dict(method=method,n_question_pairs=len(differences),p_value=1.0,reason='all_question_differences_zero')
    if method=='paired_t' and statistics.stdev(differences)==0:
        return dict(method=method,n_question_pairs=len(differences),p_value=None,reason='zero_variance_nonzero_difference')
    try:
        from scipy.stats import ttest_1samp,wilcoxon
    except ImportError:
        return dict(method=method,n_question_pairs=len(differences),p_value=None,reason='scipy_unavailable')
    result=wilcoxon(differences,alternative='two-sided',zero_method='wilcox',method='auto') if method=='wilcoxon' else ttest_1samp(differences,0)
    p=float(result.pvalue)
    return dict(method=method,n_question_pairs=len(differences),p_value=p if math.isfinite(p) else None,
                statistic=float(result.statistic),unit='question; repetitions averaged within question')


def holm_adjust(rows):
    valid=sorted((r for r in rows if r.get('p_value') is not None),key=lambda r:r['p_value'])
    previous=0.0
    for index,row in enumerate(valid):
        adjusted=max(previous,min(1.0,row['p_value']*(len(valid)-index)))
        row['p_value_holm']=adjusted
        previous=adjusted


def export_audit(out,records,comparisons,missing):
    lines=['# Historical reconciliation','',
        'Numeric checks apply to individual fields only. They do not verify answer correctness or experimental isolation.','',
        f'Original runs: {len(records)}. Verified numeric fields: {sum(r["status"]=="MATCH_VERIFIED" for r in comparisons)}.',
        f'Mismatches: {sum(r["status"]=="MISMATCH" for r in comparisons)}. Missing selectors/files: {len(missing)}.','',
        'Residual time is unattributed. Rounded stage sums are arithmetic summaries, not independent measurements.','']
    for row in comparisons:
        if row['status']!='MATCH_VERIFIED':
            lines.append(f'- {row["status"]}: {row["manuscript"]}:{row["line"]}, {row["configuration"]}/{row["question_id"]}, '
                         f'{row["metric"]}; printed={row["printed"]}, raw={row["raw"]}.')
    for row in records:
        lines.append(f'- {row["configuration"]}/{row["question_id"]}: HTTP minus arithmetic stage sum '
                     f'= {row["http_minus_stage_sum_ms"]} ms; attribution unknown.')
    (out/'historical_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def export_historical_report(folder):
    rows=read_records(folder/'historical_raw.jsonl')
    out=new_output(folder,'report')
    fields=['configuration','question_id','e2e_s','input_tokens','output_tokens',
            'stage_sum_ms_arithmetic','http_minus_stage_sum_ms','archive_file','archive_sha256','archive_pointer']
    write_csv(out/'historical_values.csv',rows,fields)
    latex=[r'\begin{tabular}{llrrr}',r'Config & Question & HTTP (s) & Input & Output \\',r'\hline']
    for row in rows:
        values=[str(row[k]) if row.get(k) is not None else 'NA' for k in
                ('configuration','question_id','e2e_s','input_tokens','output_tokens')]
        latex.append(' & '.join(values)+r' \\')
    latex.append(r'\end{tabular}')
    (out/'historical_values.tex').write_text('\n'.join(latex)+'\n',encoding='utf-8')
    (out/'report.md').write_text('# Historical observations\n\n'
        f'{len(rows)} archived runs. Each configuration/question has one observation. '
        'No repeated-run mean, SD, p-value, accuracy or causal speedup is available.\n\n'
        'HTTP time includes submission/polling. Tokens cover captured API responses only. '
        'Queue/network attribution is unknown. See historical_values.csv for traceable values.\n',encoding='utf-8')
    write_json(out/'report_provenance.json',dict(raw_log_sha256=sha_file(folder/'historical_raw.jsonl'),
        observations=len(rows),raw_log_modified=False,quality_scores_available=False))
    plot_historical(rows,out)
    return out


def plot_historical(rows,out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        write_json(out/'chart_status.json',dict(status='unavailable',reason='matplotlib_missing'))
        return
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,metric,title in zip(axes,('e2e_s','input_tokens','output_tokens'),
            ('HTTP submission + polling (s)','Captured input tokens','Captured output tokens')):
        for label,marker in [('J','o'),('M','s')]:
            present=[r for r in rows if r['configuration']==label and r.get(metric) is not None and
                     r['question_id'] in ('Q1','Q2','Q3','Q4')]
            ax.scatter([int(r['question_id'][1:]) for r in present],[r[metric] for r in present],label=label,marker=marker)
        ax.set_xticks([1,2,3,4],['Q1','Q2','Q3','Q4'])
        ax.set_ylim(bottom=0)
        ax.set_title(title,fontsize=10)
        ax.grid(axis='y',alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle('Historical observations only; n=1 per point, configurations/evidence differ',fontsize=11)
    fig.text(0.5,0.015,'M has Q2 only. Missing observations are not zero. No controlled effect is inferred.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,0.05,1,0.92])
    fig.savefig(out/'historical_observations.png',dpi=160)
    plt.close(fig)


def export_report(folder,reviews_path=None,method='wilcoxon'):
    raw_path=folder/'questions.jsonl'
    rows=read_records(raw_path)
    manifest=read_json(folder/'manifest.json')
    if not rows:
        raise ValueError('No real run records to report')
    out=new_output(folder,'report')
    reviews=read_json(reviews_path) if reviews_path else []
    indexed={}
    for review in reviews:
        key=review['record_sha256']
        if key in indexed:
            raise ValueError('Duplicate review')
        indexed[key]=review
    seen=set()
    templates=[]
    groups=defaultdict(list)
    run_keys=set()
    for row in rows:
        key=(row['question_id'],row['variant'],row['repetition'])
        if key in run_keys:
            raise ValueError('Duplicate scheduled result')
        run_keys.add(key)
        identity=row['record_sha256']
        seen.add(identity)
        if identity in indexed and row['status']=='completed_pending_review' and not row.get('exploratory'):
            row.update(review_metrics(indexed[identity]))
        templates.append(dict(record_sha256=identity,question_id=row['question_id'],variant=row['variant'],
            repetition=row['repetition'],status='pending',reviewer=None,answer_accuracy=None,
            leaked_assertions=None,eligible_assertions=None,supported_cited_claims=None,cited_claims=None,notes=''))
        groups[row['variant']].append(row)
    if set(indexed)-seen:
        raise ValueError('Review references unknown raw record')
    scheduled={(s['question_id'],s['variant'],s['repetition']) for s in manifest['schedule']}
    if run_keys-scheduled:
        raise ValueError('Run record is absent from the frozen schedule')
    summaries=[]
    for variant in dict.fromkeys(s['variant'] for s in manifest['schedule']):
        runs=groups[variant]
        planned=[s for s in manifest['schedule'] if s['variant']==variant]
        expected_q={s['question_id'] for s in planned}
        expected_reps={s['repetition'] for s in planned}
        complete=[r for r in runs if r['status']=='completed_pending_review']
        for metric in METRICS:
            by_rep=defaultdict(dict)
            eligible=runs if metric=='e2e_ms' else complete
            for row in eligible:
                if type(row.get(metric)) in (int,float):
                    by_rep[row['repetition']][row['question_id']]=row[metric]
            means=[statistics.mean(values.values()) for rep,values in by_rep.items() if set(values)==expected_q]
            summaries.append(dict(variant=variant,metric=metric,n_scheduled=len(planned),n_attempted=len(runs),
                n_completed=len(complete),n_failed=len(runs)-len(complete),n_missing=len(planned)-len(runs),
                n_observed=sum(map(len,by_rep.values())),n_expected_repetitions=len(expected_reps),n_complete_repetitions=len(means),
                mean=statistics.mean(means) if means else None,sd=statistics.stdev(means) if len(means)>1 else None,
                aggregation='mean and sample SD of complete repetition means',exploratory=manifest.get('exploratory',False)))
    pairs=defaultdict(dict)
    for row in rows:
        if row['variant'] in ('D','E') and row['status']=='completed_pending_review':
            pairs[(row['question_id'],row['repetition'])][row['variant']]=row
    differences=[]
    for (qid,rep),pair in pairs.items():
        if set(pair)!={'D','E'} or pair['D']['selected_evidence_sha256']!=pair['E']['selected_evidence_sha256']:
            continue
        for metric in METRICS:
            d,e=pair['D'].get(metric),pair['E'].get(metric)
            if type(d) in (int,float) and type(e) in (int,float):
                differences.append(dict(question_id=qid,repetition=rep,metric=metric,e_minus_d=e-d))
    tests=[]
    if manifest.get('experiment')=='E1' and not manifest.get('exploratory'):
        for metric in METRICS:
            by_question=defaultdict(list)
            for row in differences:
                if row['metric']==metric:
                    by_question[row['question_id']].append(row['e_minus_d'])
            required_reps=len({s['repetition'] for s in manifest['schedule']})
            values=[statistics.mean(vals) for vals in by_question.values() if len(vals)==required_reps]
            tests.append(dict(metric=metric,comparison='E minus D',**paired_test(values,method)))
        holm_adjust(tests)
    write_csv(out/'summary.csv',summaries)
    write_csv(out/'runs.csv',rows)
    write_csv(out/'paired_E_minus_D.csv',differences,['question_id','repetition','metric','e_minus_d'])
    write_json(out/'paired_tests.json',tests)
    write_json(out/'review_template.json',templates)
    write_json(out/'errors.json',[r for r in rows if r['status']=='failed' or r.get('accuracy')==0])
    write_json(out/'report_provenance.json',dict(raw_log_sha256=sha_file(raw_path),
        manifest_sha256=sha_file(folder/'manifest.json'),reviews_sha256=sha_file(reviews_path) if reviews_path else None,
        expected_records=len(scheduled),observed_records=len(run_keys),raw_log_modified=False))
    lines=['# Experiment report','',
        'Generated from verified raw JSONL. Missing or unreviewed measurements remain NA.',
        'SD describes complete repetition means. Paired tests use questions as units, averaging repeats first.',
        'E2 graph timing may include tokenizer HTTP; these runs are frozen-evidence replay, not production end-to-end.','',
        '| Variant | Metric | Mean ± SD | Observed | Missing |',
        '| --- | --- | --- | --- | --- |']
    latex=['\\begin{tabular}{llrr}','Variant & Metric & Mean & SD \\\\','\\hline']
    for row in summaries:
        mean='NA' if row['mean'] is None else f'{row["mean"]:.4f}'
        sd='NA' if row['sd'] is None else f'{row["sd"]:.4f}'
        lines.append(f'| {row["variant"]} | {row["metric"]} | {mean} ± {sd} | {row["n_observed"]} | {row["n_missing"]} |')
        escaped=row['metric'].replace('_',r'\_')
        latex.append(f'{row["variant"]} & {escaped} & {mean} & {sd} '+r'\\')
    latex.append('\\end{tabular}')
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'tables.tex').write_text('\n'.join(latex)+'\n',encoding='utf-8')
    return out

"""Shared CLI for Django management and standalone local audit."""
from pathlib import Path

from .io import new_output, read_json, write_json


def add_arguments(parser):
    parser.add_argument('--task',choices=['E0','E1','E2'])
    parser.add_argument('--config',default='evaluation/config.json')
    parser.add_argument('--reconcile',action='store_true')
    parser.add_argument('--prepare',action='store_true',help='Freeze real PDFs and source hashes; no model requests')
    parser.add_argument('--preflight',action='store_true',help='Read provider /models only')
    parser.add_argument('--repeat',type=int)
    parser.add_argument('--formats',help='E1: A,B,C,D,E; E2: A,B,C,D')
    parser.add_argument('--dataset',default='evaluation/benchmarks/pilot50_gold.jsonl')
    parser.add_argument('--evidence')
    parser.add_argument('--snapshot')
    parser.add_argument('--mode',choices=['gold','retrieved'],default='gold')
    parser.add_argument('--retrieved')
    parser.add_argument('--graph')
    parser.add_argument('--allow-unreviewed',action='store_true')
    parser.add_argument('--export-report',action='store_true')
    parser.add_argument('--run-dir')
    parser.add_argument('--reviews')
    parser.add_argument('--test',choices=['wilcoxon','paired_t'],default='wilcoxon')


def dispatch(options,root=None):
    root=Path(root or Path(__file__).resolve().parents[1])
    def resolve(value):
        return (root/value).resolve() if value else None
    config=read_json(resolve(options['config']))
    base=resolve(config['output_dir'])
    try:
        from dotenv import load_dotenv
        load_dotenv(root/'.env',override=False)
    except ImportError:
        pass
    if options.get('prepare'):
        from .provenance import freeze
        out=new_output(base,'snapshot')
        freeze(root,config,out)
        from .dataset_loader import load_records
        candidates=load_records(root/'evaluation/benchmarks/pilot50_gold.jsonl')
        write_json(out/'dataset_status.json',dict(candidate_records=len(candidates),
            records_marked_approved=sum(q.get('review_status')=='approved' and bool(q.get('reviewer')) for q in candidates),
            note='No labels generated. Approval flags are not validated gold counts; use dataset validation before experiments.'))
        return out
    if options.get('preflight'):
        from .runners.common import preflight
        out=new_output(base,'preflight')
        preflight(config,out)
        return out
    if options.get('export_report') and not options.get('task'):
        from .reporters import export_report,export_historical_report
        if options.get('run_dir'):
            folder=resolve(options['run_dir'])
        else:
            folders=sorted((p for p in base.iterdir() if p.is_dir() and
                           ((p/'questions.jsonl').exists() or (p/'historical_raw.jsonl').exists())),
                           key=lambda p:p.name.rsplit('-',1)[-1]) if base.exists() else []
            if not folders:
                raise ValueError('No real logs to export; run E0 --reconcile first')
            folder=folders[-1]
        if (folder/'questions.jsonl').exists():
            return export_report(folder,resolve(options.get('reviews')),options.get('test','wilcoxon'))
        return export_historical_report(folder)
    task=options.get('task')
    if not task:
        raise ValueError('Choose --task E0|E1|E2, --prepare, --preflight or --export-report')
    if options.get('reconcile') and task!='E0':
        raise ValueError('--reconcile is only valid for E0')
    if options.get('repeat') is not None and options['repeat']<(5 if task=='E0' else 3):
        raise ValueError('Minimum repetitions: E0=5, E1/E2=3')
    out=new_output(base,task)
    if task=='E0':
        from .runners.run_e0_verification import reconcile
        from .reporters import export_audit
        records,comparisons,missing=reconcile(root,out)
        export_audit(out,records,comparisons,missing)
        if options.get('repeat') is not None and not options.get('reconcile'):
            from .runners.replay import replay
            replay(root,config,out,options['repeat'])
    else:
        if not options.get('evidence') or not options.get('snapshot'):
            raise ValueError('E1/E2 require --evidence and --snapshot; no example data is substituted')
        if task=='E1':
            from .runners.run_e1_pods_ablation import run
        else:
            from .runners.run_e2_graph_ablation import run
        formats=options['formats'].split(',') if options.get('formats') else None
        run(root,config,out,dataset=resolve(options['dataset']),evidence_path=resolve(options['evidence']),
            snapshot_path=resolve(options['snapshot']),mode=options.get('mode','gold'),
            retrieved_path=resolve(options.get('retrieved')),graph_path=resolve(options.get('graph')),
            repeat=options.get('repeat'),formats=formats,allow_unreviewed=options.get('allow_unreviewed',False))
    if options.get('export_report'):
        from .reporters import export_report,export_historical_report
        if (out/'questions.jsonl').exists():
            return export_report(out,resolve(options.get('reviews')),options.get('test','wilcoxon'))
        return export_historical_report(out)
    return out

"""Reproducible synthetic baseline/structure benchmark; optional live Qwen."""
import json
import time
from pathlib import Path
from django.core.management.base import BaseCommand
from ai_extract.services.contract_pipeline_service import build_structure, structural_retrieve
from ai_extract.services.contract_pipeline_service import retrieval_gate, validate_answer
from shared.retrieval import lexical_search


class Command(BaseCommand):
    help = 'Evaluate 30 labeled synthetic contract questions without touching Neon.'

    def add_arguments(self, parser):
        parser.add_argument('--dataset',default='ai_extract/demo/contracts_qa.json')
        parser.add_argument('--output',default='scratch/contracts_demo_results.json')
        parser.add_argument('--generate',action='store_true',help='Call the configured Qwen with synthetic sources.')

    def handle(self, **options):
        data=json.loads(Path(options['dataset']).read_text(encoding='utf-8'))
        rows=[]
        for case in data['questions']:
            records=data['documents'][case['document_id']]
            started=time.perf_counter()
            baseline=lexical_search(case['query'],records,limit=2)
            tree=build_structure(records,case['document_id'])
            selected,coverage=structural_retrieve(case['query'],tree)
            quality=retrieval_gate(case['query'],selected,coverage)
            for i,leaf in enumerate(selected,1):
                leaf['evidence_id']=f'E{i}'
            expected=case.get('required_phrases',[])
            def recall(sources):
                text=' '.join(s['text'] for s in sources).casefold()
                return sum(p.casefold() in text for p in expected)/len(expected) if expected else None
            row=dict(id=case['id'],answerable=case['answerable'],baseline_source_recall=recall(baseline),
                structured_source_recall=recall(selected),generation_allowed=quality['accepted'],
                correct_answerability_decision=quality['accepted']==case['answerable'],
                retrieval_ms=round((time.perf_counter()-started)*1000,3))
            if options['generate'] and quality['accepted']:
                from ai_extract.services.contract_pipeline_service import request_qwen
                try:
                    answer=request_qwen('contract_qa',dict(query=case['query'],evidence=selected,coverage=coverage))['answer']
                    row.update(answer=answer,answer_check=validate_answer(answer,selected),
                        required_answer_values_present=all(value.casefold() in answer.casefold() for value in case.get('answer_values',[])))
                except RuntimeError as error:
                    row.update(generation_error=type(error).__name__)
            rows.append(row)
        positives=[r for r in rows if r['answerable']]
        summary=dict(dataset=data['label'],questions=len(rows),answerable=len(positives),
            baseline_mean_required_source_recall=round(sum(r['baseline_source_recall'] for r in positives)/len(positives),3),
            structured_mean_required_source_recall=round(sum(r['structured_source_recall'] for r in positives)/len(positives),3),
            answerability_decision_accuracy=round(sum(r['correct_answerability_decision'] for r in rows)/len(rows),3),
            model_generation_requested=options['generate'],synthetic_only=True,
            retrieval_mode='lexical_baseline_vs_structural_not_live_hybrid',
            not_measured=['real_contract_answer_accuracy','faithfulness','neural_hybrid_quality'])
        path=Path(options['output'])
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(dict(summary=summary,results=rows),ensure_ascii=False,indent=2),encoding='utf-8')
        self.stdout.write(json.dumps(summary,ensure_ascii=False))

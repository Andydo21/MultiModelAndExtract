import copy
import tempfile
import unittest
from pathlib import Path

from evaluation.ablation_adapters import expand,render
from evaluation.dataset_loader import load_records
from evaluation.io import append_record,digest,read_json,read_records,write_json
from evaluation.metrics import compute_alr,citation_id_overlap,evaluate_citation_faithfulness,evidence_metrics,leakage_alerts,review_metrics
from evaluation.reporters import export_report,holm_adjust,paired_test
from evaluation.runners.run_e0_verification import compare_value,table_lines
from evaluation.schema import validate_questions


def node(eid,document='fixture-doc',version='fixture-version',scope='fixture-subject-A'):
    return dict(evidence_id=eid,text='software fixture only',document_id=document,
                version=version,scope=scope,page=1,region=[0,0,1,1],breadcrumb=['fixture'])


class FrameworkTests(unittest.TestCase):
    def test_actual_usage_can_be_recorded_when_tokenizer_unavailable(self):
        from evaluation.runners.common import check_response
        row=dict(selected_evidence_ids=['E1'])
        response=dict(model='model',usage=dict(prompt_tokens=120,completion_tokens=20),
                      choices=[dict(finish_reason='stop',message=dict(content='{"answer":"ok","claims":[]}'))])
        check_response(row,response,'model',None)
        self.assertEqual(row['input_tokens'],120)
        self.assertEqual(row['output_tokens'],20)
        with self.assertRaises(ValueError):
            check_response(dict(selected_evidence_ids=['E1']),response,'model',121)

    def test_format_cells_and_input_unchanged(self):
        item=node('x')
        item['table']=dict(columns=['subject','value'],rows=[['A','8|unit'],['B','5\nunit']])
        original=copy.deepcopy(item)
        for variant in 'ABCDE':
            text=render([item],variant)
            self.assertIn('subject',text)
            self.assertIn('8',text)
            self.assertIn('5',text)
        self.assertEqual(item,original)
        self.assertIn('8\\|unit',render([item],'C'))

    def test_d_e_metadata_isolation(self):
        item=node('x')
        self.assertNotIn('"document_id"',render([item],'D'))
        self.assertIn('"document_id"',render([item],'E'))
        self.assertIn('software fixture only',render([item],'D'))

    def test_cross_document_and_version_rejected(self):
        nodes=[node('s'),node('v',version='other'),node('d',document='other')]
        edges=[dict(source='s',target=x['evidence_id'],type='CONDITION') for x in nodes[1:]]
        chosen,trace=expand(['s'],nodes,edges,'C',len,10)
        self.assertEqual(len(chosen),1)
        self.assertTrue(all(t['reason']=='document_version_boundary' for t in trace))

    def test_structural_vs_scope(self):
        nodes=[node('s'),node('t',scope='fixture-subject-B')]
        edges=[dict(source='s',target='t',type='CONDITION')]
        self.assertEqual(len(expand(['s'],nodes,edges,'C',len,10)[0]),2)
        chosen,trace=expand(['s'],nodes,edges,'D',len,10)
        self.assertEqual(len(chosen),1)
        self.assertEqual(trace[0]['reason'],'scope_boundary')

    def test_seed_budget_never_truncates(self):
        with self.assertRaises(ValueError):
            expand(['s'],[node('s')],[],'A',len,0)

    def test_added_node_budget_traced(self):
        chosen,trace=expand(['s'],[node('s'),node('t')],
            [dict(source='s',target='t',type='NEXT')],'B',len,1)
        self.assertEqual(len(chosen),1)
        self.assertEqual(trace[0]['reason'],'token_budget')

    def test_hops_and_cycle(self):
        nodes=[node('s'),node('t'),node('u')]
        edges=[dict(source=a,target=b,type='NEXT') for a,b in [('s','t'),('t','s'),('t','u')]]
        self.assertEqual([n['evidence_id'] for n in expand(['s'],nodes,edges,'B',len,10,1)[0]],['s','t'])

    def test_missing_labels_never_become_zero_error(self):
        self.assertIsNone(compute_alr('text',{}))
        self.assertIsNone(evaluate_citation_faithfulness())
        self.assertEqual(review_metrics(dict(status='pending')), {})
        self.assertIsNone(compute_alr(leaked_assertions=0,eligible_assertions=0))
        with self.assertRaises(ValueError):
            compute_alr(leaked_assertions=2,eligible_assertions=1)

    def test_regex_alert_not_a_score(self):
        case=dict(leakage_rules=[dict(subject='fixture-B',forbidden_value='8 units')])
        text='fixture-B does not have 8 units.'
        self.assertEqual(leakage_alerts(text,case)[0]['status'],'needs_semantic_review')
        self.assertIsNone(compute_alr(text,case))

    def test_citation_overlap_is_not_faithfulness(self):
        self.assertEqual(citation_id_overlap(['a','a'],['a','b'])['recall'],0.5)
        self.assertIsNone(evaluate_citation_faithfulness())
        self.assertIsNone(citation_id_overlap([],[])['precision'])

    def test_evidence_metrics_deduplicate(self):
        result=evidence_metrics(['a','a','b'],['a','c'],['c'],2)
        self.assertEqual(result['evidence_recall_at_k'],0.5)
        self.assertEqual(result['evidence_precision_at_k'],0.5)
        self.assertEqual(result['required_evidence_recall'],0)

    def test_decimal_claim_rounding(self):
        self.assertEqual(compare_value('2,01',2.00551,'stage:corpus_index')['status'],'MATCH_VERIFIED')
        self.assertEqual(compare_value('5.714',5714,'input_tokens')['status'],'MATCH_VERIFIED')
        self.assertEqual(compare_value('7.890',5714,'input_tokens')['status'],'MISMATCH')
        self.assertEqual(compare_value('65,41',None,'e2e_s')['status'],'MISSING_RAW_VALUE')

    def test_table_parser_preserves_escaped_pipes(self):
        lines=['heading','| h | val |','| --- | --- |',r'| Q1 | 1 s \| 2 tokens |','']
        rows=table_lines(lines,'| h |')
        self.assertEqual(len(rows[-1][1]),2)
        self.assertIn('2 tokens',rows[-1][1][1])

    def test_jsonl_empty_is_not_pilot50(self):
        path=Path(__file__).resolve().parents[1]/'benchmarks/pilot50_gold.jsonl'
        self.assertEqual(load_records(path),[])
        with self.assertRaises(ValueError):
            validate_questions([], [node('s')])

    def test_q1_cannot_be_test_split(self):
        with self.assertRaisesRegex(ValueError,'regression'):
            validate_questions([dict(question_id='Q1',split='test')],[node('s')])

    def test_hash_chain_detects_edit(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'fixture.jsonl'
            append_record(path,dict(test_fixture='first'))
            append_record(path,dict(test_fixture='second'))
            self.assertEqual(len(read_records(path)),2)
            text=path.read_text(encoding='utf-8').replace('first','edited')
            path.write_text(text,encoding='utf-8')
            with self.assertRaises(ValueError):
                read_records(path)

    def test_json_artifact_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'fixture.json'
            write_json(path,dict(test=True))
            with self.assertRaises(FileExistsError):
                write_json(path,dict(test=False))

    def test_pvalue_requires_independent_question_pairs(self):
        self.assertIsNone(paired_test([1,2,3,4])['p_value'])
        self.assertEqual(paired_test([0]*5)['p_value'],1)

    def test_holm_adjustment(self):
        rows=[dict(p_value=0.01),dict(p_value=0.04),dict(p_value=None)]
        holm_adjust(rows)
        self.assertEqual(rows[0]['p_value_holm'],0.02)
        self.assertEqual(rows[1]['p_value_holm'],0.04)

    def test_report_missing_repetition_and_unreviewed_accuracy(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            write_json(folder/'manifest.json',dict(experiment='E1',exploratory=False,schedule=[
                dict(question_id='fixture',variant='D',repetition=r) for r in range(3)]))
            append_record(folder/'questions.jsonl',dict(experiment='E1',evidence_mode='gold',
                question_id='fixture',variant='D',repetition=0,status='completed_pending_review',
                e2e_ms=12,selected_evidence_sha256='fixture',exploratory=False))
            out=export_report(folder)
            import csv
            with (out/'summary.csv').open(encoding='utf-8-sig',newline='') as stream:
                rows=list(csv.DictReader(stream))
            elapsed=next(r for r in rows if r['metric']=='e2e_ms')
            self.assertEqual(elapsed['n_missing'],'2')
            self.assertEqual(elapsed['sd'],'')
            self.assertEqual(next(r for r in rows if r['metric']=='accuracy')['mean'],'')
            self.assertTrue((out/'tables.tex').exists())
            self.assertEqual(len(read_records(folder/'questions.jsonl')),1)


if __name__=='__main__':
    unittest.main()

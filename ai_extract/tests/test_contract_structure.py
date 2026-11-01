from django.test import SimpleTestCase
from ai_extract.services.contract_pipeline_service import build_structure, structural_retrieve, review_batches
from ai_extract.services.contract_pipeline_service import review_contract


class StructureTests(SimpleTestCase):
    def tree(self):
        raw = [
            dict(chunk_id='p1', page_number=1, text='Điều 1. Định nghĩa\nNgày làm việc là từ thứ hai đến thứ sáu.'),
            dict(chunk_id='p2', page_number=2, text='Điều 2. Chấm dứt\n1. Được chấm dứt nếu bên kia vi phạm.\na) Phải thông báo theo Điều 3.'),
            dict(chunk_id='p3', page_number=3, text='Điều 3. Thông báo\n1. Gửi văn bản trước 30 ngày.\n2. Không được bỏ thời hạn này.')]
        records=[]
        roles=[('article',2),(None,None),('article',2),('clause',3),('point',4),
               ('article',2),('clause',3),('clause',3)]
        for row in raw:
            for line in row['text'].splitlines():
                kind,level=roles[len(records)]
                records.append(dict(chunk_id=str(len(records)),page_number=row['page_number'],text=line,
                    layout_type='title' if kind else 'paragraph',
                    structure_metadata=dict(heading_level=level,structure_kind=kind) if kind else {}))
        tree=build_structure(records,'Contract')
        leaves={l['source_chunk_id']:l for l in tree['leaves']}
        for a,b in [('4','6'),('1','4')]:
            leaves[a].setdefault('graph_related_chunk_ids',[]).append(leaves[b]['chunk_id'])
            leaves[b].setdefault('graph_related_chunk_ids',[]).append(leaves[a]['chunk_id'])
        return tree


    def test_tree_keeps_article_clause_point_and_source(self):
        tree = self.tree()
        point = next(n for n in tree['nodes'] if n['kind']=='point')
        parent = next(n for n in tree['nodes'] if n['id']==point['parent_id'])
        self.assertEqual(parent['kind'], 'clause')
        leaf = next(l for l in tree['leaves'] if l['node_id']==point['id'])
        self.assertEqual(leaf['page_number'], 2)
        self.assertIn('Điều 2. Chấm dứt', leaf['heading_path'])

    def test_reference_parent_and_definition_are_all_retrieved(self):
        leaves, coverage = structural_retrieve('chấm dứt', self.tree(), budget_chars=20000)
        text = '\n'.join(l['text'] for l in leaves)
        self.assertIn('Gửi văn bản trước 30 ngày', text)
        self.assertIn('Ngày làm việc', text)
        self.assertIn('bên kia vi phạm', text)
        self.assertEqual(coverage['unresolved_article_references'], [])

    def test_cross_page_continuation_keeps_same_parent(self):
        tree = build_structure([dict(chunk_id='a', text='Article 1. Payment\nPay within', page_number=1),
                                dict(chunk_id='b', text='30 days.', page_number=2)])
        self.assertEqual(tree['leaves'][-1]['node_id'], tree['leaves'][-2]['node_id'])

    def test_numeric_amount_is_not_a_clause_heading(self):
        tree = build_structure([dict(chunk_id='a', text='Article 1. Payment\n30 days notice\n1.1 Payment schedule')])
        self.assertEqual(sum(n['kind']=='clause' for n in tree['nodes']), 0)

    def test_long_parent_is_not_cut_to_600_chars(self):
        original = 'Payment details. ' * 200 + 'CRITICAL EXCEPTION'
        tree = build_structure([dict(chunk_id='h',text='Article 1. Payment',layout_type='title',heading_level=1), dict(chunk_id='a',text=original)])
        self.assertEqual(''.join(l['text'] for l in tree['leaves'][1:]), original)
        self.assertIn('CRITICAL EXCEPTION', ''.join(l['text'] for batch in review_batches(tree) for l in batch))

    def test_reference_cycle_is_bounded_and_deduplicated(self):
        tree = build_structure([dict(chunk_id='a', text='Article 1. Payment\nSee Article 2.\nArticle 2. Term\nSee Article 1.')])
        leaves, coverage = structural_retrieve('payment', tree)
        self.assertEqual(len(leaves), len({l['chunk_id'] for l in leaves}))
        self.assertIsNone(coverage['budget_chars'])
        self.assertLessEqual(len(leaves),len(tree['leaves']))

    def test_budget_reports_omitted_context_and_missing_reference(self):
        tree = build_structure([dict(chunk_id='h',text='Article 1. Payment',layout_type='title',heading_level=1), dict(chunk_id='a',text='Payment per Article 99.\n'+'Extra text. '*500)])
        leaves, coverage = structural_retrieve('Payment per', tree, budget_chars=1000)
        self.assertGreater(coverage['omitted_expansions'], 0)
        self.assertNotIn('document_relation_status',coverage)

    def test_review_visits_all_sources(self):
        calls = []
        def fake(task, payload):
            calls.append(payload)
            ids = payload.get('allowed_ids') or [e['evidence_id'] for e in payload['evidence']]
            return dict(answer='Supported finding '+''.join(f'[{i}]' for i in ids))
        tree = self.tree()
        answer, status, check, evidence, coverage = review_contract('Rà soát toàn bộ', tree, fake)
        reviewed_ids = {e['chunk_id'] for c in calls if c.get('review_stage')=='section' for e in c['evidence']}
        self.assertEqual(reviewed_ids, {l['chunk_id'] for l in tree['leaves']})
        self.assertTrue(coverage['whole_contract_review'])
        self.assertTrue(check['valid'])

    def test_review_failure_never_claims_full_coverage(self):
        def fake(*args):
            raise RuntimeError('Gateway offline')
        result = review_contract('Review', self.tree(), fake)
        self.assertFalse(result[-1]['whole_contract_review'])
        self.assertEqual(result[1], 'review_failed')

    def test_batch_limit_reports_partial_review(self):
        tree = build_structure([dict(chunk_id='a', text='Article 1. Payment\n'+'Data '*2000)])
        def fake(task, payload):
            return dict(answer='Finding ['+payload['evidence'][0]['evidence_id']+']')
        result = review_contract('Review', tree, fake, max_batches=1)
        self.assertEqual(result[1], 'partial_review')
        self.assertGreater(result[-1]['batches_not_attempted'], 0)

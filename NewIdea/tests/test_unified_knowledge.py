"""Real in-memory Qdrant filtering; model calls are explicit test fixtures."""
from unittest.mock import patch
from types import SimpleNamespace
from django.test import TestCase, override_settings
from django.core.management import call_command
from io import StringIO
from qdrant_client import QdrantClient
from contracts.models import KnowledgeSearchLog
from contracts.tests.test_intelligence import ContractIntelligenceTests
from NewIdea.services.search_evidence_service import submission_records, visible_text_records
from NewIdea.services.knowledge_search_service import SearchBackend, search_knowledge
from NewIdea.services.knowledge_answer_service import answer_from_sources
from ai_extract.services.contract_pipeline_service import PlanACollectionIndexer, HybridBackend, DENSE_COLLECTION


class UnifiedKnowledgeTests(TestCase):
    setUp = ContractIntelligenceTests.setUp

    def submission(self, user=None, answer='Check the AX-203 signal cable. Only replace a confirmed faulty sensor.'):
        user = user or self.user
        unit = dict(id='unit-1', source_id='expert-answer:0', quote='Check the AX-203 signal cable.',
                    title='Signal check', kind='recommendation', concepts=[])
        return KnowledgeSearchLog.objects.create(user=user, scope='expert', status='COMPLETED',
            query='How to repair AX-203?', result=dict(request=dict(company_id=user.company_id,
                answer=answer, note='', attachment_version_ids=[]), semantic_analysis=dict(units=[unit],
                relations=[dict(from_id='answer', to_id='question', kind='ANSWERS'),
                           dict(from_id='answer', to_id='unit-1', kind='HAS_UNIT')])))

    def test_records_preserve_question_answer_units_and_tenant(self):
        receipt = self.submission()
        records = submission_records(receipt)
        self.assertEqual(len(records), 3)
        self.assertEqual({r['source_role'] for r in records}, {'question', 'expert-answer'})
        self.assertEqual({r['company_id'] for r in records}, {self.company.pk})
        self.assertEqual(records[0]['evidence_kind'], 'expert_question')
        self.assertEqual(records[-1]['review_status'], 'draft')
        self.assertEqual(len({r['knowledge_key'] for r in records}), 3)
        self.assertEqual(submission_records(receipt), records)
        receipt.result['request']['answer'] = 'Unrelated content.'
        self.assertEqual(len(submission_records(receipt)), 2, 'Stale unit quote must not be searchable')

    def test_explicit_demo_marker_labels_question_answer_and_short_units(self):
        receipt=self.submission()
        receipt.result['request']['note']='[DEMO] Sample advice; not verified on a robot.'
        records=submission_records(receipt)
        self.assertTrue(all(r['is_demo'] for r in records))
        unit=records[-1]
        self.assertNotIn('DEMO',unit['text'])
        with patch('NewIdea.services.knowledge_answer_service.request_qwen') as model:
            result=answer_from_sources('AX-203 signal cable',
                [dict(unit,evidence_id='E1')],{})
        model.assert_not_called()
        self.assertEqual(result['generation_status'],'abstained')

    @override_settings(A3_HYBRID_ENABLED=True)
    def test_one_collection_search_returns_expert_units_and_preserves_full_answer_context(self):
        own = self.submission()
        foreign = self.submission(self.outsider)
        client = QdrantClient(location=':memory:')
        self.addCleanup(client.close)
        vector = [1.0]+[0.0]*1023
        with patch('ai_extract.services.contract_pipeline_service.PlanAQdrantClient.get', return_value=client), \
             patch('ai_extract.services.contract_pipeline_service._BgeM3Embedding.embed_batch',
                   side_effect=lambda texts:[vector for _ in texts]), \
             patch.object(HybridBackend, 'embedding', return_value=[vector]), \
             patch.object(SearchBackend, 'rerank', side_effect=lambda q,c:c) as rerank, \
             patch.object(SearchBackend, 'dense') as per_doc:
            indexer = PlanACollectionIndexer()
            records = submission_records(own)+submission_records(foreign)
            self.assertEqual(indexer.index_text_records(records).status, 'COMPLETED')
            self.assertEqual(indexer.index_text_records(records).status, 'COMPLETED')
            self.assertEqual(client.count(DENSE_COLLECTION).count, len(records), 'Retries must not duplicate points')
            from ai_extract.models import LayoutExtractionRun, LayoutExtractionPage, LayoutExtractionBlock
            run = LayoutExtractionRun.objects.create(version=self.new_version, status='COMPLETED', dense_index_status='COMPLETED')
            page = LayoutExtractionPage.objects.create(run=run, page_number=1)
            LayoutExtractionBlock.objects.create(page=page, block_order=1, text='AX-203 uses a signal cable and 24 V supply.')
            self.assertEqual(indexer.index_dense(run, replace_existing=False).status, 'COMPLETED')
            with patch.object(client, 'query_points', wraps=client.query_points) as lookup:
                result = search_knowledge(self.user, 'AX-203 signal cable')
                lookup.assert_called_once()
            rerank.assert_called_once()
            ranked_sources = rerank.call_args.args[1]
            self.assertTrue(any(c.get('submission_id') for c in ranked_sources))
            self.assertTrue(any(c.get('version_id') == self.new_version.pk for c in ranked_sources))
        per_doc.assert_not_called()
        self.assertEqual(result['context_coverage']['dense_state'], 'ready')
        self.assertEqual({c['submission_id'] for c in result['citations'] if c.get('submission_id')}, {own.pk})
        self.assertTrue(any('Only replace a confirmed faulty sensor' in c['text'] for c in result['citations']))
        self.assertTrue(any(n['node_id'].endswith(':unit-1') for n in result['matched_nodes']))
        self.assertTrue(any(e['type']=='ANSWERS' for g in result['context_coverage']['graph_contexts'] for e in g['edges']))

    @override_settings(A3_HYBRID_ENABLED=True)
    def test_forged_or_stale_expert_payload_cannot_become_evidence(self):
        receipt = self.submission()
        record = submission_records(receipt)[-1]
        good = dict(source_app='contract_dl', source_kind='knowledge_text', company_id=self.company.pk,
                    knowledge_key=record['knowledge_key'], content_sha256=record['content_sha256'])
        for change in ({'company_id':self.other.pk}, {'content_sha256':'outdated'}, {'knowledge_key':'foreign'}):
            with self.subTest(change=change), \
                 patch.object(SearchBackend, 'dense_knowledge', return_value=[SimpleNamespace(payload=dict(good, **change), score=1)]), \
                 patch.object(SearchBackend, 'rerank', side_effect=lambda q,c:c):
                result = search_knowledge(self.user, 'unrelated synonyms')
            self.assertEqual(result['citations'], [])
            self.assertEqual(result['context_coverage']['dense_state'], 'empty')

    def test_question_alone_is_not_treated_as_a_factual_answer(self):
        self.submission(answer='')
        with patch('NewIdea.services.knowledge_answer_service.request_qwen') as model:
            result = answer_from_sources('AX-203', [dict(evidence_id='E1', text='Does AX-203 have a faulty sensor?',
                evidence_kind='expert_question')], {})
        self.assertEqual(result['generation_status'], 'abstained')
        model.assert_not_called()

    def test_backfill_indexes_saved_records_without_ocr_or_qwen(self):
        self.submission()
        output = StringIO()
        with patch('ai_extract.management.commands.index_expert_knowledge.PlanACollectionIndexer') as factory, \
             patch('ai_extract.services.contract_pipeline_service.request_qwen') as qwen, \
             patch('ai_extract.services.contract_pipeline_service.LayoutExtractionService.extract_version') as ocr:
            factory.return_value.index_text_records.return_value = SimpleNamespace(status='COMPLETED', error='')
            call_command('index_expert_knowledge', username=self.user.username, stdout=output)
        qwen.assert_not_called()
        ocr.assert_not_called()
        self.assertEqual(len(factory.return_value.index_text_records.call_args.args[0]), 3)

    def test_source_link_reopens_saved_answer_without_exposing_another_company(self):
        receipt = self.submission()
        foreign = self.submission(self.outsider)
        response = self.client.get('/api/newidea/expert-submissions/', {'id':receipt.pk})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['answer'], receipt.result['request']['answer'])
        self.assertEqual(self.client.get('/api/newidea/expert-submissions/', {'id':foreign.pk}).status_code, 404)
        self.client.logout()
        self.assertEqual(self.client.get('/api/newidea/expert-submissions/', {'id':receipt.pk}).status_code, 401)

    @override_settings(A3_HYBRID_ENABLED=True)
    def test_new_expert_submission_indexes_units_using_shared_a3_indexer(self):
        from NewIdea.services.expert_submission_service import process
        from ai_extract.services.contract_pipeline_service import IndexResult
        receipt = self.submission()
        def extract(task, payload):
            source = payload['sources'][0]
            return dict(units=[dict(id='u', source_id=source['id'], quote=source['text'],
                                   title='Source unit', kind='fact')])
        with patch('ai_extract.services.contract_pipeline_service.request_qwen', side_effect=extract), \
             patch('NewIdea.repositories.expert_submission_repository.sync_semantic_graph', return_value=dict(status='ready')), \
             patch('NewIdea.repositories.expert_submission_repository.related_units', return_value=[]), \
             patch('ai_extract.services.contract_pipeline_service.PlanACollectionIndexer') as factory:
            factory.return_value.index_text_records.return_value = IndexResult('COMPLETED', 4)
            result = process(receipt)
        indexed = factory.return_value.index_text_records.call_args.args[0]
        self.assertTrue(any(r.get('unit_kind') for r in indexed))
        self.assertEqual(result['vector_index']['status'], 'COMPLETED')

    def test_question_provenance_reaches_generation_and_verification(self):
        from ai_extract.services.contract_pipeline_service import request_qwen
        source = dict(evidence_id='E1', text='Does AX-203 require repair?', evidence_kind='expert_question',
                      source_role='question', review_status='draft', author='expert')
        calls = []
        def chat(base, messages, **kwargs):
            calls.append(messages)
            return '{"supported":true,"reason":"The source is explicitly a saved question."}'
        with override_settings(VLLM_BASE_URL='http://gateway.test/v1'), \
             patch('ai_extract.services.contract_pipeline_service._chat', side_effect=chat):
            request_qwen('verify_answer', dict(evidence=[source], answer='Saved question [E1].', query='Question'))
        prompt = str(calls)
        self.assertIn('expert_question', prompt)
        self.assertIn('draft', prompt)
        self.assertIn('questions are not factual evidence', prompt)

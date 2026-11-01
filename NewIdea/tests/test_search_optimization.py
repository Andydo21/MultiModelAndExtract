"""Verify inference call counts, scoped lookup and durable progress."""
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from unittest.mock import Mock, patch
from django.test import SimpleTestCase, TestCase, override_settings
from contracts.tests import test_intelligence as fixtures
from ai_extract.models import LayoutExtractionRun, Clause
from ai_extract.services.contract_pipeline_service import HybridBackend
from NewIdea.services.knowledge_search_service import SearchBackend, search_knowledge
from NewIdea.services.knowledge_answer_service import generate_checked_answer
from NewIdea.services.search_job_service import reserve_search, update_progress, read_job


class SearchReuseTests(SimpleTestCase):
    def test_question_parts_share_one_dense_embedding_batch(self):
        backend=SearchBackend()
        with patch.object(HybridBackend,'embedding',return_value=[[1.0],[2.0]]) as inference:
            backend.prime_embeddings(['Whole question','Part question','Whole question'])
            self.assertEqual(backend.embedding(['Whole question']),[[1.0]])
            self.assertEqual(backend.embedding(['Part question']),[[2.0]])
            inference.assert_called_once_with(['Whole question','Part question'])

    def test_parallel_documents_share_one_query_embedding(self):
        for method, argument, value in [('embedding', ['camera'], [[1.0, 0.0]]),
                                        ('colpali_embedding', 'camera', [[1.0]*128])]:
            with self.subTest(method=method):
                backend = SearchBackend()
                callers, entered, release = Barrier(4), Event(), Event()
                def model(arg):
                    entered.set()
                    if not release.wait(3):
                        raise AssertionError('Embedding did not finish')
                    return value
                def caller():
                    callers.wait(timeout=3)
                    return getattr(backend, method)(argument)
                with patch.object(HybridBackend, method, side_effect=model) as inference:
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        tasks = [pool.submit(caller) for _ in range(4)]
                        try:
                            self.assertTrue(entered.wait(3))
                        finally:
                            release.set()
                        self.assertEqual([task.result(timeout=3) for task in tasks], [value]*4)
                inference.assert_called_once_with(argument)

    def test_colpali_embeds_once_but_queries_each_version_with_its_own_scope(self):
        backend = SearchBackend()
        client = Mock()
        client.query_points.return_value = SimpleNamespace(points=[])
        versions = [SimpleNamespace(pk=i, contract_id=i+10, contract=SimpleNamespace(company_id=i+20)) for i in (1, 2)]
        with patch.object(HybridBackend, 'colpali_embedding', return_value=[[1.0]*128]) as embedding, \
                patch('ai_extract.services.contract_pipeline_service.PlanAQdrantClient.get', return_value=client):
            for version in versions:
                backend.colpali('camera', version, version.pk+30)
        embedding.assert_called_once_with('camera')
        self.assertEqual(client.query_points.call_count, 2)
        self.assertNotEqual(client.query_points.call_args_list[0].kwargs['query_filter'],
                            client.query_points.call_args_list[1].kwargs['query_filter'])

    def test_embedding_failure_is_reused_only_within_one_search(self):
        for method, argument in [('colpali_embedding', 'camera'), ('embedding', ['camera'])]:
            with self.subTest(method=method), patch.object(HybridBackend, method, side_effect=RuntimeError('offline')) as model:
                backend = SearchBackend()
                for _ in range(3):
                    with self.assertRaises(RuntimeError):
                        getattr(backend, method)(argument)
                self.assertEqual(model.call_count, 1)
                with self.assertRaises(RuntimeError):
                    getattr(SearchBackend(), method)(argument)
                self.assertEqual(model.call_count, 2)

    def test_generation_reports_phases_without_emitting_unverified_answer(self):
        stages = []
        calls = []
        def model(task, payload):
            calls.append(task)
            if task == 'verify_answer':
                return dict(supported=True)
            return dict(answer='Payment within 90 days [E99].' if len(calls) == 1 else 'Payment within 30 days [E1].')
        result = generate_checked_answer(dict(query='payment', evidence=[dict(evidence_id='E1', text='Payment within 30 days.')]),
                                         request_model=model, progress=stages.append)
        self.assertEqual(stages, ['generating', 'repairing_citations'])
        self.assertNotIn('verify_answer',calls)
        self.assertEqual(result['generation_status'], 'generated')


class SearchOptimizationIntegrationTests(TestCase):
    setUp = fixtures.ContractIntelligenceTests.setUp

    @override_settings(A3_HYBRID_ENABLED=True, A2_ANSWER_SOURCE_LIMIT=1)
    def test_only_admitted_document_expands_graph(self):
        self.user.is_superuser=True
        self.user.save(update_fields=['is_superuser'])
        Clause.objects.create(version=self.foreign_version,clause_title='Payment',clause_content='Payment within 45 days.')
        from ai_extract.services.contract_pipeline_service import expand_retrieval_context
        def rerank(query, candidates):
            return sorted([dict(c,neural_logit=4 if c['version_id']==self.new_version.pk else 1)
                           for c in candidates],key=lambda c:c['neural_logit'],reverse=True)
        with patch.object(SearchBackend,'dense_knowledge',return_value=[]), \
             patch.object(SearchBackend,'colpali_knowledge',return_value=[]), \
             patch.object(SearchBackend,'rerank',side_effect=rerank), \
             patch('NewIdea.services.knowledge_search_service.expand_retrieval_context',wraps=expand_retrieval_context) as expand:
            result=search_knowledge(self.user,'payment')
        self.assertEqual([c.args[2].pk for c in expand.call_args_list],[self.new_version.pk])
        self.assertEqual(result['context_coverage']['source_selection']['selected_seeds'],1)
        self.assertGreater(result['context_coverage']['source_selection']['omitted_seeds'],0)
        self.assertTrue(result['context_coverage']['partial'])

    @override_settings(A3_HYBRID_ENABLED=True)
    def test_corpus_subquestions_recover_a_detail_without_selecting_a_document(self):
        Clause.objects.create(version=self.new_version,clause_title='Control',
            clause_content='Two robot arms share one controller.')
        Clause.objects.create(version=self.new_version,clause_title='Compatibility',
            clause_content='Compatible controller RC8A.')
        facet='Which controller is compatible?'
        ranked_queries=[]
        def rerank(query,candidates):
            ranked_queries.append(query)
            target='RC8A' if query.endswith('Focused question: '+facet) else 'Two robot arms'
            return sorted([dict(c,neural_logit=3 if target in c['text'] else -5) for c in candidates],
                          key=lambda c:c['neural_logit'],reverse=True)
        with patch('NewIdea.services.knowledge_search_service.plan_retrieval_questions',
                   return_value=dict(status='ready',queries=[facet])), \
             patch.object(SearchBackend,'dense_knowledge',return_value=[]), \
             patch.object(SearchBackend,'colpali_knowledge',return_value=[]), \
             patch.object(SearchBackend,'rerank',side_effect=rerank), \
             patch('NewIdea.services.knowledge_answer_service.answer_from_sources',return_value={}):
            result=search_knowledge(self.user,'robot controller compatibility',generate=True)
        self.assertIn('robot controller compatibility\nFocused question: '+facet,ranked_queries)
        self.assertNotIn(facet,ranked_queries, 'A facet must retain the original subject at reranking')
        self.assertTrue(any('RC8A' in c['text'] for c in result['citations']))
        detail=next(c for c in result['citations'] if 'RC8A' in c['text'])
        self.assertEqual(detail['retrieval_hints'],[dict(part=1,rank=1)])
        self.assertTrue(any('Two robot arms' in c['text'] for c in result['citations']))
        self.assertEqual(result['context_coverage']['versions_scanned'],result['context_coverage']['versions_total'])

    @override_settings(A3_HYBRID_ENABLED=True)
    def test_model_ranked_candidates_are_not_rejected_by_fixed_zero_threshold(self):
        with patch('NewIdea.services.knowledge_search_service.plan_retrieval_questions',
                   return_value=dict(status='ready',queries=['payment'])), \
             patch.object(SearchBackend,'dense_knowledge',return_value=[]), \
             patch.object(SearchBackend,'colpali_knowledge',return_value=[]), \
             patch.object(SearchBackend,'rerank',side_effect=lambda q,c:[dict(s,neural_logit=-4) for s in c]), \
             patch('NewIdea.services.knowledge_answer_service.request_qwen',return_value=dict(answer='Payment is required [E1].')) as model:
            result=search_knowledge(self.user,'payment',generate=True)
        self.assertEqual(result['generation_status'],'generated')
        self.assertTrue(result['citations'])
        self.assertGreater(result['context_coverage']['versions_deeply_searched'],0)
        self.assertTrue(model.called)

    @override_settings(A3_HYBRID_ENABLED=True)
    def test_reranker_outage_is_not_reported_as_missing_document_evidence(self):
        with patch('NewIdea.services.knowledge_search_service.plan_retrieval_questions',
                   return_value=dict(status='ready',queries=['payment'])), \
             patch.object(SearchBackend,'dense_knowledge',return_value=[]), \
             patch.object(SearchBackend,'colpali_knowledge',return_value=[]), \
             patch.object(SearchBackend,'rerank',side_effect=RuntimeError('offline')), \
             patch('NewIdea.services.knowledge_answer_service.answer_from_sources') as answer:
            result=search_knowledge(self.user,'payment',generate=True)
        self.assertEqual(result['generation_status'],'retrieval_unavailable')
        self.assertEqual(result['generation_diagnostics']['state'],'retrieval_unavailable')
        answer.assert_not_called()

    def test_corpus_sources_and_completed_runs_use_prefetched_data(self):
        from contracts.models import ContractContext
        from NewIdea.repositories import contract_repository
        from ai_extract.repositories import layout_repository
        run = LayoutExtractionRun.objects.create(version=self.new_version, status='COMPLETED')
        Clause.objects.filter(version=self.foreign_version).delete()
        ContractContext.objects.create(version=self.foreign_version, context_type='raw_text',
            content='EVP2 Runtime uses the controller and camera.', source='test')
        self.user.is_superuser = True
        versions, _ = contract_repository.latest_visible_versions(self.user)
        with self.assertNumQueries(0):
            sources = {version.pk: contract_repository.version_sources(version) for version in versions}
            completed = {version.pk: layout_repository.latest_run(version, completed=True) for version in versions}
        self.assertIn('60 days', sources[self.new_version.pk][0]['text'])
        self.assertIn('Runtime', sources[self.foreign_version.pk][0]['text'])
        self.assertEqual(completed[self.new_version.pk].pk, run.pk)

    @override_settings(A3_HYBRID_ENABLED=True)
    def test_multiple_documents_rerank_once_before_graph_expansion(self):
        Clause.objects.create(version=self.foreign_version, clause_title='Payment', clause_content='Payment within 45 days.')
        for version in (self.new_version, self.foreign_version):
            LayoutExtractionRun.objects.create(version=version, status='COMPLETED', dense_index_status='COMPLETED')
        self.user.is_superuser = True
        self.user.save(update_fields=['is_superuser'])
        stages = []
        from ai_extract.services.contract_pipeline_service import expand_retrieval_context
        def rerank(query, candidates):
            stages.append('rerank')
            self.assertEqual({c['version_id'] for c in candidates}, {self.new_version.pk, self.foreign_version.pk})
            return candidates
        def expand(*args,**kwargs):
            self.assertEqual(stages[0], 'rerank')
            stages.append('graph')
            return expand_retrieval_context(*args)
        with patch.object(SearchBackend, 'dense_knowledge', return_value=[]), \
                patch.object(SearchBackend, 'dense', return_value=[]), \
                patch.object(SearchBackend, 'colpali', return_value=[]), \
                patch.object(SearchBackend, 'rerank', side_effect=rerank) as rank, \
                patch('NewIdea.services.knowledge_search_service.expand_retrieval_context', side_effect=expand):
            result = search_knowledge(self.user, 'payment')
        rank.assert_called_once()
        self.assertEqual(stages, ['rerank', 'graph', 'graph'])
        self.assertEqual({c['version_id'] for c in result['citations']}, {self.new_version.pk, self.foreign_version.pk})

    def test_progress_is_persisted_and_still_access_controlled(self):
        receipt = reserve_search(self.user, 'payment', self.new_version, generated=True)
        update_progress(receipt, 'verifying')
        visible = read_job(self.user, receipt.pk)
        self.assertEqual(visible['result']['progress']['stage'], 'verifying')
        self.assertNotIn('answer', visible['result']['progress'])
        from contracts.models import KnowledgeSearchLog
        with self.assertRaises(KnowledgeSearchLog.DoesNotExist):
            read_job(self.outsider, receipt.pk)

    @override_settings(A3_HYBRID_ENABLED=True)
    def test_corpus_streams_overlap_without_requerying_documents(self):
        Clause.objects.create(version=self.foreign_version, clause_title='Payment', clause_content='Payment within 45 days.')
        for version in (self.new_version, self.foreign_version):
            LayoutExtractionRun.objects.create(version=version, status='COMPLETED',
                dense_index_status='COMPLETED', colpali_index_status='COMPLETED')
        self.user.is_superuser = True
        self.user.save(update_fields=['is_superuser'])
        from NewIdea.repositories.contract_repository import latest_visible_versions
        versions, _ = latest_visible_versions(self.user)
        expected = [v.pk for v in versions if v.pk in {self.new_version.pk, self.foreign_version.pk}]
        overlap = Barrier(2)
        def retrieve(*args, **kwargs):
            # This would time out if the two corpus streams ran sequentially.
            overlap.wait(timeout=3)
            return []
        with patch.object(SearchBackend, 'dense_knowledge', side_effect=retrieve) as dense, \
             patch.object(SearchBackend, 'colpali_knowledge', side_effect=retrieve) as visual, \
             patch.object(SearchBackend, 'dense') as per_document, \
             patch.object(SearchBackend, 'colpali') as per_page, \
             patch.object(SearchBackend, 'rerank', side_effect=lambda q,c:c):
            result = search_knowledge(self.user, 'payment')
        dense.assert_called_once()
        visual.assert_called_once()
        per_document.assert_not_called()
        per_page.assert_not_called()
        self.assertEqual([d['version_id'] for d in result['context_coverage']['version_details']], expected)
        self.assertEqual({c['version_id'] for c in result['citations']}, set(expected))


class DurableCorpusTests(TestCase):
    setUp = fixtures.ContractIntelligenceTests.setUp

    def corpus(self):
        from NewIdea.repositories.contract_repository import latest_visible_versions, completed_extraction_run, IndexedCorpus
        versions, _ = latest_visible_versions(self.user, limit=None, load_sources=False)
        from NewIdea.services.knowledge_search_service import indexed_corpus
        return indexed_corpus(versions, {v.pk: completed_extraction_run(v) for v in versions}, [])

    def test_warm_lookup_avoids_source_reconstruction_and_preserves_bm25(self):
        from shared.retrieval import lexical_search
        corpus = self.corpus()
        expected = lexical_search('payment 60', corpus[self.new_version.pk]['leaves'])
        with patch('NewIdea.repositories.contract_repository.source_records', side_effect=AssertionError('unexpected source scan')):
            warm = self.corpus()
            actual = warm.keyword_search('payment 60')
        self.assertEqual(actual, expected)
        self.assertEqual(warm.rebuilt, 0)
        self.assertEqual(warm.loaded, {})
        self.assertEqual(set(warm.snapshots), {self.new_version.pk})

    def test_edit_and_reextract_replace_old_postings(self):
        from ai_extract.models import LayoutExtractionPage, LayoutExtractionBlock, RetrievalPassage
        corpus = self.corpus()
        clause = Clause.objects.get(version=self.new_version)
        clause.clause_content = 'Payment within 90 days.'
        clause.save()
        changed = self.corpus()
        self.assertEqual(changed.rebuilt, 1)
        self.assertFalse(changed.keyword_search('60'))
        self.assertTrue(changed.keyword_search('90'))
        run = LayoutExtractionRun.objects.create(version=self.new_version, status='RUNNING')
        page = LayoutExtractionPage.objects.create(run=run, page_number=1)
        LayoutExtractionBlock.objects.create(page=page, block_order=0, text='New delivery terms', bbox=[])
        run.status = 'COMPLETED'
        run.save()
        replaced = self.corpus()
        self.assertFalse(replaced.keyword_search('90'))
        self.assertTrue(replaced.keyword_search('delivery'))
        self.assertEqual(RetrievalPassage.objects.count(), replaced.snapshots[self.new_version.pk].passage_count)

    def test_deleted_source_invalidates_snapshot(self):
        self.corpus()
        Clause.objects.filter(version=self.new_version).delete()
        self.assertFalse(self.corpus().keyword_search('payment'))


class InferenceCacheTests(SimpleTestCase):
    @override_settings(A2_INFERENCE_CACHE_ENABLED=True, A2_INFERENCE_CACHE_VERSION='cache-test')
    def test_successful_embeddings_reused_across_backend_instances(self):
        from django.core.cache import cache
        cache.clear()
        with patch.object(HybridBackend, '_embedding_uncached', return_value=[[1.0]]) as infer:
            self.assertEqual(HybridBackend().embedding(['same']), [[1.0]])
            self.assertEqual(HybridBackend().embedding(['same', 'same']), [[1.0], [1.0]])
            infer.assert_called_once_with(['same'])
        cache.clear()

    @override_settings(A2_INFERENCE_CACHE_ENABLED=True, A2_INFERENCE_CACHE_VERSION='cache-test')
    def test_failed_embeddings_are_not_cached_across_requests(self):
        from django.core.cache import cache
        cache.clear()
        with patch.object(HybridBackend, '_embedding_uncached', side_effect=[RuntimeError('offline'), [[1.0]]]) as infer:
            with self.assertRaises(RuntimeError):
                HybridBackend().embedding(['same'])
            self.assertEqual(HybridBackend().embedding(['same']), [[1.0]])
            self.assertEqual(infer.call_count, 2)
        cache.clear()


class BatchRerankTests(SimpleTestCase):
    @override_settings(A2_BATCH_RERANK=True, RAG_INFERENCE_URL='http://inference')
    def test_duplicate_pairs_share_inference_without_merging_source_ids(self):
        a, b = dict(chunk_id='a', text='same passage'), dict(chunk_id='b', text='same passage')
        reply = Mock()
        reply.json.return_value = dict(model='BAAI/bge-reranker-v2-m3', scores=[2.0, 3.0])
        with patch('ai_extract.services.contract_pipeline_service.requests.post', return_value=reply) as http:
            results = HybridBackend().rerank_many([('one', [a,b]), ('two', [b]), ('empty',[])])
        self.assertEqual(len(http.call_args.kwargs['json']['pairs']), 2)
        self.assertEqual([r['chunk_id'] for r in results[0]], ['a','b'])
        self.assertEqual(results[1][0]['neural_logit'], 3.0)
        self.assertEqual(results[2], [])


class ReviewReuseTests(SimpleTestCase):
    def review(self, memo, claims, sources, complete=False):
        from ai_extract.services.contract_pipeline_service import _verify_cited_claims
        return _verify_cited_claims('http://model/v1', dict(query='Explain both',
            citation_claims=claims, check_completeness=complete, _verified_claims=memo), sources, {}, {}, [], [])

    def test_only_new_claims_send_sources_but_final_review_sees_whole_answer(self):
        import json
        a = dict(text='One robot.', evidence_ids=['E1'])
        b = dict(text='Two controllers.', evidence_ids=['E2'])
        sources = [dict(evidence_id='E1',text='one robot source',version_id=1),
                   dict(evidence_id='E2',text='two controllers source',version_id=1)]
        memo = {}
        with patch('ai_extract.services.contract_pipeline_service._chat', side_effect=[
                json.dumps(dict(supported=[True],missing_parts=[])),
                json.dumps(dict(supported=[True],missing_parts=['Explanation missing'],conflicting_check_ids=[]))]) as chat:
            self.review(memo,[a],sources)
            result = self.review(memo,[a,b],sources,True)
        prompt = chat.call_args.args[1][1]['content']
        self.assertNotIn('one robot source',prompt)
        self.assertIn('two controllers source',prompt)
        self.assertIn('One robot.',prompt)
        self.assertEqual(result['verification_input']['reused_claims'],1)
        self.assertEqual(len(result['claim_checks']),2)
        self.assertFalse(result['completeness']['complete'])

    def test_changed_source_version_and_claim_are_rechecked(self):
        import json
        claim = dict(text='One robot.', evidence_ids=['E1'])
        source = dict(evidence_id='E1', text='one robot source', version_id=1)
        memo = {}
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(dict(supported=[True],missing_parts=[]))) as chat:
            self.review(memo,[claim],[source])
            self.review(memo,[claim],[source])
            self.assertEqual(chat.call_count,1)
            self.review(memo,[claim],[dict(source,version_id=2)])
            self.review(memo,[dict(claim,text='Two robots.')],[source])
            self.assertEqual(chat.call_count,3)

    def test_conflicting_retained_claim_is_invalidated(self):
        import json
        claim = dict(text='One robot.', evidence_ids=['E1'])
        source = dict(evidence_id='E1',text='one robot source')
        memo = {}
        with patch('ai_extract.services.contract_pipeline_service._chat',side_effect=[
                json.dumps(dict(supported=[True],missing_parts=[])),
                json.dumps(dict(supported=[],missing_parts=[],conflicting_check_ids=['0']))]):
            self.review(memo,[claim],[source])
            result = self.review(memo,[claim],[source],True)
        self.assertFalse(result['supported'])
        self.assertFalse(result['claim_checks'][0]['supported'])
        self.assertEqual(memo,{})


class ParallelQuestionTests(SimpleTestCase):
    @override_settings(A2_SEARCH_WORKERS=2)
    def test_independent_queries_overlap_and_keep_input_order(self):
        from NewIdea.services.knowledge_search_service import collect_questions
        barrier = Barrier(2)
        def collect(query):
            barrier.wait(timeout=3)
            return [query+' result']
        self.assertEqual(collect_questions(['a','b'],collect),[('a','a result'),('b','b result')])


class PromptDeduplicationTests(SimpleTestCase):
    def test_sharing_sees_repeated_values_inside_distinct_table_rows(self):
        import json
        from NewIdea.services.knowledge_answer_service import _share_graph_values
        condition='Only after approval and within the specified operating temperature.'
        metadata=dict(condition=condition,polarity='negated',review_status='unreviewed')
        original=dict(rows=[[i,condition] for i in range(50)],
                      metadata=[dict(metadata) for _ in range(20)])
        packed=_share_graph_values(original)
        def expand(value):
            if isinstance(value,dict) and set(value)=={'shared'}:
                return expand(packed['shared_values'][value['shared']])
            if isinstance(value,dict):return {k:expand(v) for k,v in value.items()}
            if isinstance(value,list):return [expand(v) for v in value]
            return value
        restored=expand({k:v for k,v in packed.items() if k not in {'shared_values','encoding'}})
        self.assertEqual(restored,original)
        self.assertLess(len(json.dumps(packed)),len(json.dumps(original)))

    def test_shared_graph_values_are_lossless_and_keep_conditions(self):
        import json
        from copy import deepcopy
        from NewIdea.services.knowledge_answer_service import _share_graph_values
        condition='Only after approval and within the specified operating temperature.'
        original=dict(nodes=[dict(id=i,label=str(i),condition=condition) for i in range(20)],
            relations=[dict(from_id=i,to_id=i+1,review_status='unreviewed',condition=condition,polarity='negated') for i in range(19)])
        snapshot=deepcopy(original)
        packed=_share_graph_values(original)
        def expand(value):
            if isinstance(value,dict) and set(value)=={'shared'}:return packed['shared_values'][value['shared']]
            if isinstance(value,dict):return {k:expand(v) for k,v in value.items()}
            if isinstance(value,list):return [expand(v) for v in value]
            return value
        restored=expand({k:v for k,v in packed.items() if k not in {'shared_values','encoding'}})
        self.assertEqual(restored,original)
        self.assertEqual(original,snapshot)
        self.assertLess(len(json.dumps(packed)),len(json.dumps(original)))

    def test_contained_text_has_one_copy_with_separate_document_provenance(self):
        from ai_extract.services.contract_pipeline_service import _prompt_sources
        text='A source statement with its conditions. '*10
        sources=[dict(evidence_id='E1',text='Header '+text+' End',version_id=1),
                 dict(evidence_id='E2',text=text,version_id=2)]
        prompt=_prompt_sources(sources)
        self.assertEqual(prompt.count(text),1)
        self.assertIn('span [7,',prompt)
        self.assertIn('"version_id":1',prompt)
        self.assertIn('"version_id":2',prompt)

    def test_repair_sources_preserve_direct_dependency_without_reading_unrelated_text(self):
        from NewIdea.services.knowledge_answer_service import focus_answer_sources
        sources=[dict(evidence_id='E1',text='Already answered'),dict(evidence_id='E2',text='Missing fact'),
                 dict(evidence_id='E3',text='Only under these conditions'),dict(evidence_id='E4',text='Unrelated condition')]
        graph=dict(nodes=[dict(id=i,evidence_ids=['E'+str(i)]) for i in range(1,5)],edges=[
            dict(from_id=2,to_id=3,type='DEPENDS_ON'),dict(from_id=3,to_id=4,type='REFERENCES')])
        payload=dict(evidence=sources,graph_context=graph)
        result=focus_answer_sources(payload,[dict(id='P2',source_ids=['E2'])])
        self.assertEqual([s['evidence_id'] for s in result['evidence']],['E2','E3'])
        self.assertEqual(payload['evidence'],sources)

    def test_all_verified_claims_are_visible_when_no_support_checks_remain(self):
        import json
        from ai_extract.services.contract_pipeline_service import _verify_cited_claims
        memo={}
        source=dict(evidence_id='E1',text='The source gives the explanation.')
        payload=dict(query='Explain',citation_claims=[dict(text='The explanation.',evidence_ids=['E1'],part_ids=['P1'])],
            _verified_claims=memo,answer_parts=[dict(id='P1',question='Explain')])
        with patch('ai_extract.services.contract_pipeline_service._chat',side_effect=[
            json.dumps(dict(supported=[True],missing_parts=[],missing_part_ids=[])),
            json.dumps(dict(supported=[],missing_parts=[],missing_part_ids=[],conflicting_check_ids=[]))]) as chat:
            _verify_cited_claims('http://model',payload,[source],{}, {}, [], [])
            result=_verify_cited_claims('http://model',dict(payload,check_completeness=True),[source],{}, {}, [], [])
        sent=json.loads(chat.call_args.args[1][1]['content'].split('CONTEXT:\n',1)[1])
        self.assertEqual(sent['checks'],[])
        self.assertEqual(sent['answer_claims'][0]['text'],'The explanation.')
        self.assertEqual(sent['answer_claims'][0]['support'],'verified')
        self.assertTrue(result['completeness']['complete'])


class RetrievalGroupingIntegrityTests(SimpleTestCase):
    def test_intent_comes_from_the_same_model_plan_without_an_extra_call(self):
        import json
        from ai_extract.services.contract_pipeline_service import _request_openai_compatible
        from NewIdea.services.knowledge_answer_service import needs_source_images
        data=dict(queries=['Describe the arrangement.'],groups=[[0]],
                  intent=dict(scope='focused',needs_images=True))
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(data)) as model:
            result=_request_openai_compatible('http://model','retrieval_plan',dict(query='この配置を説明してください'))
        self.assertEqual(result['intent'],data['intent'])
        self.assertTrue(needs_source_images(result,[dict(text='Caption',image_path='figure.png')]))
        self.assertFalse(needs_source_images(result,[dict(text='No image')]))
        self.assertEqual(model.call_count,1)

    def test_invalid_model_intent_is_not_used_as_a_keyword_fallback(self):
        import json
        from ai_extract.services.contract_pipeline_service import _request_openai_compatible,QwenGatewayError
        data=dict(queries=['Question'],groups=[[0]],intent=dict(scope='focused',needs_images='yes'))
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(data)):
            with self.assertRaises(QwenGatewayError):
                _request_openai_compatible('http://model','retrieval_plan',dict(query='Question'))

    def test_atomic_scopes_survive_even_when_queries_share_a_retrieval(self):
        import json
        from ai_extract.services.contract_pipeline_service import _request_openai_compatible
        data=dict(queries=['Setup requirements?','Operation requirements?'],groups=[[0,1]],facets=[
            dict(query_index=0,subject='Application',attribute='requirements',phase='setup'),
            dict(query_index=1,subject='Application',attribute='requirements',phase='operation')])
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(data)) as chat:
            result=_request_openai_compatible('http://model','retrieval_plan',dict(query='Requirements in both phases?'))
        self.assertEqual(len(result['queries']),2)
        self.assertEqual(result['seed_group_count'],1)
        self.assertEqual([p['query_scope']['phase'] for p in result['answer_parts']],['setup','operation'])
        self.assertEqual([p['question'] for p in result['answer_parts']],data['queries'])
        self.assertEqual(chat.call_count,1)

    def test_comparison_subjects_share_search_without_sharing_answer_ownership(self):
        import json
        from ai_extract.services.contract_pipeline_service import _request_openai_compatible
        questions=[dict(question='Capacity of Alpha?',subject='Alpha',attribute='capacity',phase=''),
            dict(question='Capacity of Beta?',subject='Beta',attribute='capacity',phase='')]
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(
                dict(queries=questions,groups=[[0,1]],intent=dict(scope='focused',needs_images=False)))):
            result=_request_openai_compatible('http://model','retrieval_plan',dict(query='Compare capacities'))
        self.assertEqual(len(result['queries']),1)
        self.assertEqual([p['query_scope']['subject'] for p in result['answer_parts']],['Alpha','Beta'])
        self.assertEqual(len({p['retrieval_query'] for p in result['answer_parts']}),1)

    def test_duplicate_optional_facets_do_not_discard_valid_questions(self):
        import json
        from ai_extract.services.contract_pipeline_service import _request_openai_compatible
        data=dict(queries=['First fact?','Second fact?'],groups=[[0],[1]],facets=[
            dict(query_index=0,subject='A',attribute='value',phase='setup'),
            dict(query_index=0,subject='B',attribute='value',phase='operation')])
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(data)):
            result=_request_openai_compatible('http://model','retrieval_plan',dict(query='Both facts?'))
        self.assertEqual(result['queries'],data['queries'])
        self.assertEqual(len(result['answer_parts']),2)
        self.assertEqual(result['answer_parts'][0]['query_scope'],{})

    def test_grouping_preserves_all_member_questions_including_conditions(self):
        import json
        from ai_extract.services.contract_pipeline_service import _request_openai_compatible
        questions=['Compare loads of Alpha and Beta.','What is the maximum load?','Which temperature conditions apply?']
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(dict(queries=questions,groups=[[0,1],[2]]))):
            result=_request_openai_compatible('http://model','retrieval_plan',dict(query='All requested facts'))
        self.assertEqual(result['queries'],[' '.join(questions[:2]),questions[2]])

    def test_grouping_cannot_silently_drop_a_question(self):
        import json
        from ai_extract.services.contract_pipeline_service import _request_openai_compatible,QwenGatewayError
        for groups in ([[0]],[[0,1],[1]],[[0,3]]):
            with self.subTest(groups=groups),patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(dict(queries=['Fact?','Condition?'],groups=groups))):
                with self.assertRaises(QwenGatewayError):
                    _request_openai_compatible('http://model','retrieval_plan',dict(query='Fact and condition?'))

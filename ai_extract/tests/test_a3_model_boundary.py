"""A3 projects Surya/GLiNER; Qwen is only an answer-time dependency."""
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
from django.test import SimpleTestCase,override_settings
from ai_extract.services.document_structure_service import analyze_document_structure
from ai_extract.services.document_graph_service import projection
from ai_extract.services.contract_pipeline_service import structural_retrieve


class A3ModelBoundaryTests(SimpleTestCase):
    def setUp(self):
        index = patch('NewIdea.services.knowledge_search_service.refresh_snapshot')
        index.start()
        self.addCleanup(index.stop)

    def tree(self):
        return analyze_document_structure([
            dict(chunk_id='a',text='Power isolation procedure.'),
            dict(chunk_id='b',text='Unrelated marketing address.')],'Manual')

    def test_retired_qwen_links_are_ignored_by_graph_and_retrieval(self):
        tree=self.tree()
        a,b=tree['leaves']
        tree['document_relations']=dict(status='complete',links=[dict(
            from_id=a['chunk_id'],to_id=b['chunk_id'],kind='REFERENCES',
            from_quote=a['text'],to_quote=b['text'],confidence=.95,direction_supported=True)])
        run=NS(pk=7,version_id=3,version=NS(contract=NS(pk=2,company_id=1)))
        _,_,edges=projection(run,tree)
        self.assertFalse(any(e['origin']=='qwen_document_pair' for e in edges))
        hits,_=structural_retrieve('Power',tree,seeds=[a])
        self.assertEqual([h['chunk_id'] for h in hits],[a['chunk_id']])

    @override_settings(A3_DOCUMENT_RELATIONS_ENABLED=True)
    def test_finish_never_calls_qwen_even_with_retired_setting(self):
        from ai_extract.tests.test_page_pipeline import PagePipelineTests
        from ai_extract.services.layout_service import PageExtractionSession
        from ai_extract.services.contract_pipeline_service import IndexResult
        run=PagePipelineTests().run_object()
        run.structure=self.tree();run.total_pages=2
        indexer=Mock();indexer.index_dense.return_value=IndexResult('COMPLETED',2)
        with patch('ai_extract.services.contract_pipeline_service.PlanACollectionIndexer',return_value=indexer), \
             patch('ai_extract.services.contract_pipeline_service.request_qwen',side_effect=AssertionError('A3 called Qwen')) as qwen, \
             patch('ai_extract.services.document_graph_service.sync_graph',return_value=dict(status='ready')), \
             patch('ai_extract.repositories.layout_repository.save_run'):
            with PageExtractionSession(run) as session:
                session.records=run.structure['leaves']
                session.finish()
        qwen.assert_not_called()
        self.assertEqual(run.status,'COMPLETED')
        self.assertNotIn('document_relations',run.structure)

    def test_graph_failure_preserves_sources_and_reports_incomplete_index(self):
        from ai_extract.tests.test_page_pipeline import PagePipelineTests
        from ai_extract.services.layout_service import PageExtractionSession
        from ai_extract.services.contract_pipeline_service import IndexResult
        run=PagePipelineTests().run_object();run.structure=self.tree();run.total_pages=2
        indexer=Mock();indexer.index_dense.return_value=IndexResult('COMPLETED',2)
        with patch('ai_extract.services.contract_pipeline_service.PlanACollectionIndexer',return_value=indexer), \
             patch('ai_extract.services.document_graph_service.sync_graph',return_value=dict(status='unavailable')), \
             patch('ai_extract.repositories.layout_repository.save_run'):
            with PageExtractionSession(run) as session:
                session.records=run.structure['leaves']
                with patch.object(session.progress,'emit') as emit:
                    session.finish()
        self.assertEqual(run.status,'COMPLETED')
        self.assertEqual(len(run.structure['leaves']),2)
        self.assertIn('Neo4j index unavailable',run.error_message)
        self.assertTrue(any(c.args[:2]==('pipeline','completed_with_index_errors') for c in emit.call_args_list))

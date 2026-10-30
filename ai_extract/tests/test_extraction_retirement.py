from types import SimpleNamespace
from unittest.mock import Mock
from django.test import SimpleTestCase,TestCase
from contracts.models import Company,Contract,ContractVersion,ContractContext
from ai_extract.models import LayoutExtractionRun,LayoutExtractionPage
from ai_extract.repositories.layout_repository import retire_old_page_contexts,persist_pages
from ai_extract.services.contract_pipeline_service import PlanACollectionIndexer,DENSE_COLLECTION,COLPALI_COLLECTION


class VectorRetirementTests(SimpleTestCase):
    def test_retired_entity_vectors_are_removed_only_after_same_document_source_index_is_ready(self):
        from ai_extract.services.contract_pipeline_service import ENTITY_COLLECTION
        indexer=self.indexer();run=self.run_object()
        indexer.client.get_collections.return_value.collections.append(SimpleNamespace(name=ENTITY_COLLECTION))
        run.structure['graph_index']=dict(status='ready')
        result=indexer.prune_old_run_points(run)
        request=indexer.client.delete.call_args_list[-1].kwargs
        self.assertEqual(request['collection_name'],ENTITY_COLLECTION)
        self.assertEqual(result['entity'],'completed')
        conditions=request['points_selector'].must
        self.assertEqual({c.key:c.match.value for c in conditions if c.match},dict(
            source_app='contract_dl',company_id=1,contract_id=2,version_id=3))
        self.assertEqual(next(c.range.lte for c in conditions if c.key=='contract_dl_run_id'),20)
        indexer.client.delete.reset_mock();run.structure['graph_index']['status']='unavailable'
        indexer.prune_old_run_points(run)
        self.assertFalse(any(c.kwargs['collection_name']==ENTITY_COLLECTION for c in indexer.client.delete.call_args_list))

    def run_object(self):
        return SimpleNamespace(pk=20,status='COMPLETED',version_id=3,dense_index_status='COMPLETED',
            colpali_index_status='FAILED',structure=dict(leaves=[dict(layout_type='figure')]),
            version=SimpleNamespace(contract=SimpleNamespace(pk=2,company_id=1)))

    def indexer(self):
        indexer=PlanACollectionIndexer.__new__(PlanACollectionIndexer)
        indexer.client=Mock()
        indexer.client.get_collections.return_value=SimpleNamespace(collections=[
            SimpleNamespace(name=DENSE_COLLECTION),SimpleNamespace(name=COLPALI_COLLECTION)])
        return indexer

    def test_only_older_vectors_in_the_exact_tenant_contract_version_are_deleted(self):
        indexer=self.indexer();run=self.run_object()
        self.assertEqual(indexer.prune_old_run_points(run),dict(dense='completed',colpali='skipped'))
        request=indexer.client.delete.call_args_list[0].kwargs
        self.assertEqual(request['collection_name'],DENSE_COLLECTION)
        conditions=request['points_selector'].must
        self.assertEqual({c.key:c.match.value for c in conditions if c.match},dict(
            source_app='contract_dl',company_id=1,contract_id=2,version_id=3))
        self.assertEqual(next(c.range.lt for c in conditions if c.key=='contract_dl_run_id'),20)

    def test_retry_cleanup_preserves_current_leaf_vectors_and_excludes_other_runs(self):
        indexer=self.indexer();run=self.run_object()
        run.structure['leaves']=[dict(chunk_id='block:9',text='Current source')]
        indexer.prune_old_run_points(run)
        query=indexer.client.delete.call_args_list[1].kwargs['points_selector']
        self.assertEqual(next(c.match.value for c in query.must if c.key=='contract_dl_run_id'),20)
        from uuid import uuid5,NAMESPACE_URL
        self.assertEqual(query.must_not[0].has_id,[str(uuid5(NAMESPACE_URL,'contract-dl:dense:20:block:9'))])

    def test_failed_extraction_or_failed_branch_never_deletes_previous_vectors(self):
        indexer=self.indexer();run=self.run_object();run.status='FAILED'
        indexer.prune_old_run_points(run)
        indexer.client.delete.assert_not_called()
        run.status='COMPLETED';run.dense_index_status='FAILED';run.colpali_index_status='PARTIAL'
        indexer.prune_old_run_points(run)
        indexer.client.delete.assert_not_called()

    def test_empty_visual_replacement_retires_obsolete_images_but_missing_gateway_does_not(self):
        indexer=self.indexer();run=self.run_object();run.dense_index_status='FAILED';run.colpali_index_status='SKIPPED'
        indexer.prune_old_run_points(run)
        indexer.client.delete.assert_not_called()
        run.structure['leaves']=[]
        self.assertEqual(indexer.prune_old_run_points(run)['colpali'],'completed')
        self.assertEqual(indexer.client.delete.call_args.kwargs['collection_name'],COLPALI_COLLECTION)


class PageRetirementTests(TestCase):
    def setUp(self):
        company=Company.objects.create(company_name='Retirement',tax_code='RETIRE')
        contract=Contract.objects.create(company=company,contract_code='RETIRE',title='Source')
        self.version=ContractVersion.objects.create(contract=contract)
        self.run=LayoutExtractionRun.objects.create(version=self.version,status='COMPLETED')

    def test_page_retry_replaces_blocks_and_never_changes_another_run(self):
        old=LayoutExtractionRun.objects.create(version=self.version,status='FAILED')
        old_page=LayoutExtractionPage.objects.create(run=old,page_number=1,extracted_text='Old source')
        block=SimpleNamespace(order=1,layout_type='paragraph',text='First',bbox=[0,0,10,10],confidence=1)
        page=SimpleNamespace(page_number=1,width=100,height=100,image_path='p.png',text='First',blocks=[block])
        first,_=persist_pages(self.run,SimpleNamespace(pages=[page]),incremental=True)
        block.text=page.text='Replacement'
        second,_=persist_pages(self.run,SimpleNamespace(pages=[page]),incremental=True)
        self.assertEqual(self.run.pages.count(),1)
        self.assertEqual(self.run.pages.get().blocks.count(),1)
        self.assertEqual([r['text'] for r in second],['Replacement'])
        self.assertNotEqual(first[0]['chunk_id'],second[0]['chunk_id'])
        old_page.refresh_from_db();self.assertEqual(old_page.extracted_text,'Old source')

    def test_obsolete_page_contexts_are_retired_without_removing_other_sources(self):
        LayoutExtractionPage.objects.create(run=self.run,page_number=1)
        for source in ('plan-a:page:1','plan-a:page:3','manual-source'):
            ContractContext.objects.create(version=self.version,context_type='raw_text',source=source,content=source)
        self.assertEqual(retire_old_page_contexts(self.run),1)
        self.assertEqual(set(ContractContext.objects.values_list('source',flat=True)),{'plan-a:page:1','manual-source'})

    def test_failed_or_superseded_run_cannot_retire_page_contexts(self):
        context=ContractContext.objects.create(version=self.version,context_type='raw_text',source='plan-a:page:9',content='Old')
        self.run.status='FAILED';self.assertEqual(retire_old_page_contexts(self.run),0)
        self.run.status='COMPLETED'
        LayoutExtractionRun.objects.create(version=self.version,status='RUNNING')
        self.assertEqual(retire_old_page_contexts(self.run),0)
        context.refresh_from_db();self.assertEqual(context.content,'Old')

    def test_batch_persistence_does_not_erase_unrelated_raw_context(self):
        manual=ContractContext.objects.create(version=self.version,context_type='raw_text',source='manual-source',content='Keep')
        page=SimpleNamespace(page_number=1,width=100,height=100,image_path='p.png',text='Page',blocks=[])
        persist_pages(self.run,SimpleNamespace(pages=[page]))
        manual.refresh_from_db();self.assertEqual(manual.content,'Keep')

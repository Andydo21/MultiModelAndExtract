"""Verify page concurrency and failure isolation without calling real models."""
from contextlib import nullcontext
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock, patch

from django.test import SimpleTestCase, TestCase, override_settings

from ai_extract.services.contract_pipeline_service import IndexResult
from ai_extract.services.layout_service import PageExtractionSession


class WorkflowLeaseTests(SimpleTestCase):
    def lease(self,available=True):
        database=Mock(vendor='postgresql');lease=Mock(settings_dict={})
        cursor=Mock();cursor.fetchone.return_value=(available,)
        lease.cursor.return_value=MagicMock(__enter__=Mock(return_value=cursor),__exit__=Mock(return_value=False))
        database.copy.return_value=lease
        return database,lease,cursor

    def test_keepalive_runs_during_model_work_and_connection_always_closes(self):
        from ai_extract.repositories import layout_repository as repository
        database,lease,cursor=self.lease();heartbeat=Event()
        cursor.execute.side_effect=lambda sql,*args:heartbeat.set() if sql=='SELECT 1' else None
        with patch('django.db.connections',{'default':database}),patch.object(repository,'WORKFLOW_HEARTBEAT_SECONDS',.01):
            with repository.workflow_slot() as available:
                self.assertTrue(available)
                self.assertTrue(heartbeat.wait(2))
                lease.close.assert_not_called()
        lease.close.assert_called_once()

    def test_broken_cleanup_cannot_mask_success_or_original_work_error(self):
        from django.db import OperationalError
        from ai_extract.repositories import layout_repository as repository
        for failure in (False,True):
            database,lease,cursor=self.lease();lease.rollback.side_effect=OperationalError('SSL EOF')
            with patch('django.db.connections',{'default':database}),self.assertLogs(repository.__name__,level='WARNING'):
                if failure:
                    with self.assertRaisesRegex(ValueError,'work failed'):
                        with repository.workflow_slot():raise ValueError('work failed')
                else:
                    with repository.workflow_slot() as available:self.assertTrue(available)
            lease.close.assert_called_once()

    def test_lost_lease_refuses_further_writes_instead_of_reconnecting_lease(self):
        from django.db import OperationalError
        from ai_extract.repositories import layout_repository as repository
        database,lease,cursor=self.lease();closed=Event();lease.close.side_effect=closed.set
        def query(sql,*args):
            if sql=='SELECT 1':raise OperationalError('connection lost')
        cursor.execute.side_effect=query
        with patch('django.db.connections',{'default':database}),patch.object(repository,'WORKFLOW_HEARTBEAT_SECONDS',.01):
            with repository.workflow_slot():
                self.assertTrue(closed.wait(2))
                with self.assertRaisesRegex(RuntimeError,'refusing further writes'):repository.ensure_write_connection()
        database.copy.assert_called_once()

    def test_busy_lock_returns_false_without_a_heartbeat(self):
        from ai_extract.repositories import layout_repository as repository
        database,lease,cursor=self.lease(False)
        with patch('django.db.connections',{'default':database}):
            with repository.workflow_slot() as available:self.assertFalse(available)
        self.assertFalse(any(call.args[0]=='SELECT 1' for call in cursor.execute.call_args_list))
        lease.close.assert_called_once()


class WriteConnectionTests(SimpleTestCase):
    def test_cell_ocr_disagreement_is_kept_with_source_and_warning(self):
        from tempfile import TemporaryDirectory
        from PIL import Image
        from ai_extract.services.layout_service import refine_page_tables
        from ai_extract.services.surya_inference_service import TABLE_CELL_POLICY
        from ai_extract.services.inference_runtime_service import PipelineProgress
        with TemporaryDirectory() as directory:
            Image.new('RGB',(200,100),'white').save(Path(directory)/'page.png')
            block=SimpleNamespace(text='32 mm',layout_type='table',bbox=[0,0,1000,1000],
                structure_metadata=dict(text_source='surya_ocr',table_cells=[dict(text='32 mm')]))
            page=SimpleNamespace(page_number=1,blocks=[block],text=block.text,image_path='page.png')
            result=dict(table_structure_status='cell_aware',table_structure_policy=TABLE_CELL_POLICY,
                table_rows=[['33 mm']],table_cells=[dict(row=1,column=1,rowspan=1,colspan=1,text='33 mm',bbox=[0,0,200,100])],
                table_structure_review=dict(rows=1,columns=1,image_size=[200,100],source_quotations=1,
                    source_quotations_covered=0,missing_source_quotations=['32 mm'],structure_warnings=['source_quotations_not_covered_by_cell_ocr']))
            with override_settings(MEDIA_ROOT=directory,RAG_INFERENCE_URL='http://inference'), \
                 patch('ai_extract.repositories.page_inference_cache_repository.read',return_value=None), \
                 patch('ai_extract.repositories.page_inference_cache_repository.write'), \
                 patch('requests.post',return_value=Mock(status_code=200,json=Mock(return_value=result))):
                counts=refine_page_tables(page,PipelineProgress(1))
            self.assertEqual(counts['completed'],1)
            self.assertEqual(block.text,'33 mm')
            self.assertEqual(block.structure_metadata['table_original_text'],'32 mm')
            self.assertEqual(block.structure_metadata['text_source'],'surya_table_cell_ocr')
            self.assertEqual(block.structure_metadata['table_rows'],[['33 mm']])
            self.assertEqual(block.structure_metadata['table_structure_review']['missing_source_quotations'],['32 mm'])

    def test_successful_cell_reconstruction_preserves_source_for_ner_and_embedding(self):
        from tempfile import TemporaryDirectory
        from PIL import Image
        from ai_extract.services.layout_service import refine_page_tables
        from ai_extract.services.surya_inference_service import TABLE_CELL_POLICY
        from ai_extract.services.inference_runtime_service import PipelineProgress
        with TemporaryDirectory() as directory:
            Image.new('RGB',(200,100),'white').save(Path(directory)/'page.png')
            original='Original OCR\nwith source line breaks'
            block=SimpleNamespace(text=original,layout_type='table',bbox=[0,0,1000,1000],
                structure_metadata=dict(text_source='surya_ocr',table_cells=[dict(text=original)]))
            page=SimpleNamespace(page_number=1,blocks=[block],text=original,image_path='page.png')
            result=dict(table_structure_status='cell_aware',table_structure_policy=TABLE_CELL_POLICY,
                table_rows=[[original]],table_cells=[dict(row=1,column=1,rowspan=1,colspan=1,text=original,bbox=[0,0,200,100])],
                table_structure_review=dict(rows=1,columns=1,image_size=[200,100],source_quotations=1,source_quotations_covered=1))
            with override_settings(MEDIA_ROOT=directory,RAG_INFERENCE_URL='http://inference'), \
                 patch('ai_extract.repositories.page_inference_cache_repository.read',return_value=None), \
                 patch('ai_extract.repositories.page_inference_cache_repository.write'), \
                 patch('requests.post',return_value=Mock(status_code=200,json=Mock(return_value=result))):
                counts=refine_page_tables(page,PipelineProgress(1))
            self.assertEqual(counts,dict(completed=1,unresolved=0))
            self.assertEqual((block.text,page.text),(original,original))
            self.assertTrue(block.structure_metadata['table_source_text_retained'])

    def test_non_table_markup_is_not_promoted_and_source_text_is_preserved(self):
        from ai_extract.services.layout_service import refine_page_tables
        from ai_extract.services.inference_runtime_service import PipelineProgress
        for kind in ('list_group','page_header','page_footer','paragraph'):
            block=SimpleNamespace(text='Original source',layout_type=kind,
                structure_metadata=dict(text_source='surya_ocr',table_rows=[['Candidate']],table_cells=[]))
            page=SimpleNamespace(page_number=1,blocks=[block],text=block.text)
            with self.subTest(kind=kind),patch('requests.post') as post:
                counts=refine_page_tables(page,PipelineProgress(1))
            post.assert_not_called()
            self.assertEqual(counts,dict(completed=0,unresolved=0))
            self.assertEqual(block.text,'Original source')
            self.assertNotIn('table_rows',block.structure_metadata)
            self.assertEqual(block.structure_metadata['table_markup_status'],'ignored_non_table_layout')

    def test_physical_table_rejects_duplicate_cells_holes_and_outside_boxes(self):
        from copy import deepcopy
        from ai_extract.services.layout_service import validate_table_cells
        valid=dict(table_structure_status='cell_aware',table_rows=[['A','B']],
            table_structure_review=dict(rows=1,columns=2),table_cells=[
                dict(row=1,column=1,rowspan=1,colspan=1,text='A',bbox=[0,0,100,100]),
                dict(row=1,column=2,rowspan=1,colspan=1,text='B',bbox=[100,0,200,100])])
        validate_table_cells(valid,(200,100))
        duplicate=deepcopy(valid);duplicate['table_cells'].append(duplicate['table_cells'][0])
        missing=deepcopy(valid);missing['table_cells'].pop()
        outside=deepcopy(valid);outside['table_cells'][1]['bbox'][2]=201
        for malformed in (duplicate,missing,outside):
            with self.subTest(malformed=malformed),self.assertRaises(ValueError):
                validate_table_cells(malformed,(200,100))

    def test_failed_physical_reconstruction_retains_source_without_candidate_grid(self):
        from ai_extract.services.layout_service import refine_page_tables
        from ai_extract.services.inference_runtime_service import PipelineProgress
        block=SimpleNamespace(text='Original OCR',layout_type='table',bbox=[0,0,1000,1000],
            structure_metadata=dict(text_source='surya_ocr',structure_image_path='../outside.png',
                table_rows=[['Candidate']],table_cells=[dict(text='Candidate')]))
        page=SimpleNamespace(page_number=1,blocks=[block],text=block.text)
        with override_settings(MEDIA_ROOT=Path('/private/media')):
            counts=refine_page_tables(page,PipelineProgress(1))
        self.assertEqual(counts,dict(completed=0,unresolved=1))
        self.assertEqual(block.text,'')
        self.assertEqual(block.structure_metadata['table_original_text'],'Original OCR')
        self.assertEqual(block.structure_metadata['table_structure_status'],'unresolved')
        self.assertNotIn('table_cells',block.structure_metadata)

    def test_extraction_entry_defers_colpali_by_default(self):
        from ai_extract.services.contract_pipeline_service import LayoutExtractionService
        service=LayoutExtractionService();version=object()
        with patch('ai_extract.repositories.layout_repository.workflow_slot',return_value=nullcontext(True)), \
             patch.object(service,'_extract_version') as extract:
            service.extract_version(version)
        extract.assert_called_once_with(version,force_ocr=False,defer_colpali=True)
    def test_entity_extraction_runs_before_identity_without_reference_resolution(self):
        from ai_extract.services.layout_service import prepare_document
        from ai_extract.services.inference_runtime_service import PipelineProgress
        tree=dict(source_fingerprint='source',leaves=[])
        run=SimpleNamespace(id=1,total_pages=2,structure=tree)
        events=[]
        def reading(run):
            events.append('reading')
            tree['reading_analysis']=dict(status='complete',source_ids=[])
        def extract(tree):
            events.append('entities')
            return dict(status='complete',entities=[],relations=[],source_relations=[],
                source_coverage=dict(completed_ids=[]))
        def identity(run,tree):
            events.append('identity')
            self.assertIn('document_relations',tree['knowledge_graph'])
            return dict(status='complete',mentions=0,linked=0,unresolved=0,canonical_entities=0)
        with patch('ai_extract.services.document_structure_service.prepare_reading',side_effect=reading), \
             patch('ai_extract.services.gliner_entity_service.extract_graph_entities',side_effect=extract), \
             patch('ai_extract.services.gliner_entity_service.enrich_document_relations') as references, \
             patch('ai_extract.services.gliner_identity_classifier.resolve_identities',side_effect=identity), \
             patch('ai_extract.services.layout_service.layout_repository.save_run'), \
             patch('ai_extract.services.document_graph_service.sync_graph') as sync:
            prepare_document(run,PipelineProgress(1))
        self.assertEqual(events,['reading','entities','identity'])
        self.assertNotIn('document_context_relations',tree['pregraph_analysis']['stages'])
        self.assertEqual(tree['knowledge_graph']['document_relations']['status'],'disabled')
        references.assert_not_called()
        sync.assert_not_called()

    def test_unavailable_reference_provider_does_not_block_entity_pipeline(self):
        from ai_extract.services.layout_service import prepare_document
        from ai_extract.services.inference_runtime_service import PipelineProgress
        tree=dict(source_fingerprint='source',leaves=[],pregraph_analysis=dict(status='complete'),
            identity_resolution=dict(status='complete'))
        run=SimpleNamespace(id=1,total_pages=1,structure=tree)
        def reading(run):tree['reading_analysis']=dict(status='complete',source_ids=[])
        knowledge=dict(status='complete',entities=[],relations=[],source_coverage=dict(completed_ids=[]))
        with patch('ai_extract.services.document_structure_service.prepare_reading',side_effect=reading), \
             patch('ai_extract.services.gliner_entity_service.extract_graph_entities',return_value=knowledge), \
             patch('ai_extract.services.gliner_entity_service.enrich_document_relations',side_effect=RuntimeError('unused')) as references, \
             patch('ai_extract.services.gliner_identity_classifier.resolve_identities',return_value=dict(
                 status='complete',mentions=0,linked=0,unresolved=0,canonical_entities=0)) as identity, \
             patch('ai_extract.services.layout_service.layout_repository.save_run'):
            prepare_document(run,PipelineProgress(1))
        references.assert_not_called()
        identity.assert_called_once()
        self.assertEqual(tree['pregraph_analysis']['status'],'complete')

    def test_stale_connection_reopens_before_a_write_without_replaying_it(self):
        from ai_extract.repositories.layout_repository import ensure_write_connection
        database=Mock(vendor='postgresql',in_atomic_block=False,connection=object())
        database.is_usable.return_value=False
        with patch('django.db.connections',{'default':database}):
            ensure_write_connection()
        database.close.assert_called_once()
        database.ensure_connection.assert_called_once()
        database.cursor.assert_not_called()

    def test_active_transaction_is_never_reconnected(self):
        from ai_extract.repositories.layout_repository import ensure_write_connection
        database=Mock(vendor='postgresql',in_atomic_block=True,connection=object())
        with patch('django.db.connections',{'default':database}):
            ensure_write_connection()
        database.is_usable.assert_not_called()
        database.close.assert_not_called()


@override_settings(MEDIA_ROOT=Path('/test-media'))
class PagePipelineTests(SimpleTestCase):
    def test_deferred_visual_branch_does_not_submit_background_work(self):
        with PageExtractionSession(self.run_object(),defer_colpali=True) as session:
            with patch.object(session.executor,'submit') as submit:
                session._submit_visual(1)
                self.assertEqual(session.visual_tasks[1].result().status,'SKIPPED')
                self.assertEqual(session.visual_tasks[1].result().error,'')
                submit.assert_not_called()

    def test_duplicate_page_is_rejected_before_persistence(self):
        with PageExtractionSession(self.run_object()) as session:
            session.completed_pages.add(1)
            with patch('ai_extract.repositories.layout_repository.persist_pages') as persist:
                with self.assertRaisesRegex(ValueError,'Duplicate extraction page'):
                    session.process_page(SimpleNamespace(page_number=1))
                persist.assert_not_called()
            with patch('ai_extract.repositories.layout_repository.persist_rendered_page') as persist:
                with self.assertRaisesRegex(ValueError,'Duplicate extraction page'):
                    session.start_page_image(1,Path('/test-media/1.png'),10,10)
                persist.assert_not_called()

    @override_settings(A3_KNOWLEDGE_ENABLED=True,A3_ENTITY_PROVIDER='gliner',A3_RELATED_PASSAGES_ENABLED=True)
    def test_dense_embedding_precedes_related_reranking_and_graph_publication(self):
        events=[];run=self.run_object();run.structure=dict(leaves=[],structure_status='native_structure')
        def prepare(run,progress,**options):
            self.assertTrue(options['defer_related_passages']);events.append('entities')
            run.structure['knowledge_graph']=dict(status='complete',schema='ner',entities=[])
        def related(run,progress):
            events.append('related');run.structure['related_passages']=dict(status='complete',links=[])
        indexer=Mock();indexer.index_dense.side_effect=lambda *a,**k:(events.append('dense') or IndexResult('COMPLETED',0))
        with patch('ai_extract.services.layout_service.prepare_document',side_effect=prepare), \
             patch('ai_extract.services.layout_service.prepare_related_passages',side_effect=related), \
             patch('ai_extract.repositories.layout_repository.save_run'), \
             patch('ai_extract.services.contract_pipeline_service.PlanACollectionIndexer',return_value=indexer), \
             patch('ai_extract.services.document_graph_service.sync_graph',side_effect=lambda run:(events.append('graph') or dict(status='ready'))):
            with PageExtractionSession(run) as session:session.finish()
        self.assertEqual(events,['entities','dense','related','graph'])
        indexer.index_entities.assert_not_called()
    def test_deferred_visual_branch_does_not_prepare_crops_or_call_colpali(self):
        with PageExtractionSession(self.run_object(),defer_colpali=True) as session, \
             patch('ai_extract.services.layout_extractor.visual_sources') as sources, \
             patch('ai_extract.services.contract_pipeline_service.PlanACollectionIndexer') as indexer:
            result=session._index_visual(1)
        self.assertEqual(result.status,'SKIPPED')
        sources.assert_not_called();indexer.assert_not_called()

    def setUp(self):
        index = patch('NewIdea.services.knowledge_search_service.refresh_snapshot')
        index.start()
        self.addCleanup(index.stop)


    def run_object(self):
        return SimpleNamespace(pk=7, id=7, version_id=3, version=SimpleNamespace(contract=SimpleNamespace(title='Source', id=2, company_id=1, contract_code='C')),
                               structure={}, total_pages=0, total_blocks=0)

    def test_structure_retains_previous_pages_without_rereading_old_blocks(self):
        observed = []
        def structure(records, title):
            observed.append([dict(record) for record in records])
            return dict(leaves=[], structure_status='native_structure')
        with patch('ai_extract.services.layout_service.transaction.atomic', side_effect=nullcontext), \
             patch('ai_extract.repositories.layout_repository.persist_pages',
                   side_effect=[([dict(chunk_id='block:1', text='Heading', page_number=1)], 1),
                                ([dict(chunk_id='block:2', text='Continued content', page_number=2)], 1)]) as persist, \
             patch('ai_extract.repositories.layout_repository.save_run'), \
             patch('ai_extract.services.document_structure_service.analyze_document_structure', side_effect=structure), \
             patch('ai_extract.services.document_graph_service.infer_relations', return_value={'status': 'disabled'}), \
             patch('ai_extract.services.document_graph_service.sync_graph', return_value={'status': 'ready'}):
            with PageExtractionSession(self.run_object()) as session:
                session._index_visual = lambda number: IndexResult('SKIPPED')
                for number in (1, 2):
                    session.process_page(SimpleNamespace(page_number=number, blocks=[object()]))
        self.assertEqual([[r['chunk_id'] for r in rows] for rows in observed],
                         [['block:1'], ['block:1', 'block:2']])
        self.assertTrue(all(call.kwargs == dict(incremental=True, cumulative=False)
                            for call in persist.call_args_list))

    def test_colpali_overlaps_page_structure_but_dense_waits_for_complete_document(self):
        started, structure_done = Event(), Event()
        def visual(page_number):
            started.set()
            if not structure_done.wait(3):
                raise AssertionError('Structure branch did not overlap ColPali')
            return IndexResult('COMPLETED', 1)
        def structure(*args):
            self.assertTrue(started.wait(3))
            structure_done.set()
            return dict(leaves=[], structure_status='native_structure')
        indexer = Mock()
        indexer.index_dense.return_value = IndexResult('COMPLETED', 4)
        with patch('ai_extract.services.layout_service.transaction.atomic', side_effect=nullcontext), \
             patch('ai_extract.repositories.layout_repository.persist_rendered_page'), \
             patch('ai_extract.repositories.layout_repository.persist_pages', return_value=([], 0)), \
             patch('ai_extract.repositories.layout_repository.save_run'), \
             patch('ai_extract.services.document_structure_service.analyze_document_structure', side_effect=structure), \
             patch('ai_extract.services.document_graph_service.infer_relations', return_value={'status': 'disabled'}), \
             patch('ai_extract.services.document_graph_service.sync_graph', return_value={'status': 'ready'}), \
             patch('ai_extract.services.contract_pipeline_service.PlanACollectionIndexer', return_value=indexer):
            run = self.run_object()
            with PageExtractionSession(run) as session:
                session._index_visual = visual
                for number in (1, 2):
                    started.clear()
                    structure_done.clear()
                    session.start_page_image(number, Path('/test-media') / f'{number}.png', 100, 200)
                    self.assertFalse(started.is_set(), 'ColPali must wait for confirmed Surya figures')
                    with self.assertRaisesRegex(RuntimeError, 'previous page'):
                        session.start_page_image(number + 1, Path('/test-media/other.png'), 100, 200)
                    session.process_page(SimpleNamespace(page_number=number, blocks=[]))
                    self.assertFalse(session.visual_tasks)
                    self.assertEqual(session.page_states[-1]['dense'], 'DEFERRED')
                    self.assertEqual(session.page_states[-1]['colpali'], 'COMPLETED')
                    indexer.index_dense.assert_not_called()
                session.finish()
        indexer.index_dense.assert_called_once_with(run, replace_existing=False)
        self.assertEqual(run.total_pages, 2)
        self.assertEqual((run.dense_points_count, run.colpali_points_count), (4, 2))

    def test_failure_in_one_index_does_not_erase_the_other(self):
        indexer = Mock()
        indexer.index_dense.return_value = IndexResult('FAILED', error='embedding unavailable')
        with patch('ai_extract.services.layout_service.transaction.atomic', side_effect=nullcontext), \
             patch('ai_extract.repositories.layout_repository.persist_pages', return_value=([], 0)), \
             patch('ai_extract.repositories.layout_repository.save_run'), \
             patch('ai_extract.services.document_structure_service.analyze_document_structure',
                   return_value=dict(leaves=[], structure_status='native_structure')), \
             patch('ai_extract.services.document_graph_service.infer_relations', return_value={'status': 'disabled'}), \
             patch('ai_extract.services.document_graph_service.sync_graph', return_value={'status': 'ready'}), \
             patch('ai_extract.services.contract_pipeline_service.PlanACollectionIndexer', return_value=indexer):
            run = self.run_object()
            with PageExtractionSession(run) as session:
                session._index_visual = lambda number: IndexResult('COMPLETED', 1)
                session.process_page(SimpleNamespace(page_number=1, blocks=[]))
                session.finish()
        self.assertEqual(run.dense_index_status, 'FAILED')
        self.assertEqual(run.colpali_index_status, 'COMPLETED')
        self.assertEqual(run.colpali_points_count, 1)

    def test_visual_failure_continues_to_next_page_and_retries_only_at_finish(self):
        calls = []
        def visual(number):
            calls.append(number)
            if calls == [1]:
                return IndexResult('FAILED', error='temporary connection failure')
            return IndexResult('COMPLETED', 1, point_ids=(f'p{number}',))
        indexer = Mock()
        def dense(*args, **kwargs):
            self.assertEqual(calls, [1, 2], 'Text indexing must not wait for the final visual retry')
            return IndexResult('COMPLETED', 4)
        indexer.index_dense.side_effect = dense
        with patch('ai_extract.services.layout_service.transaction.atomic', side_effect=nullcontext), \
             patch('ai_extract.repositories.layout_repository.persist_pages', return_value=([], 0)) as persist, \
             patch('ai_extract.repositories.layout_repository.save_run'), \
             patch('ai_extract.services.document_structure_service.analyze_document_structure',
                   return_value=dict(leaves=[], structure_status='native_structure')) as structure, \
             patch('ai_extract.services.document_graph_service.infer_relations', return_value={'status': 'empty'}), \
             patch('ai_extract.services.document_graph_service.sync_graph', return_value={'status': 'ready'}), \
             patch('ai_extract.services.contract_pipeline_service.PlanACollectionIndexer', return_value=indexer):
            run = self.run_object()
            with PageExtractionSession(run) as session:
                session._index_visual = visual
                session.process_page(SimpleNamespace(page_number=1, blocks=[]))
                self.assertEqual(run.colpali_index_status, 'FAILED')
                session.process_page(SimpleNamespace(page_number=2, blocks=[]))
                self.assertEqual(calls, [1, 2])
                session.finish()
                self.assertEqual(session.page_states[0]['colpali_attempts'], 2)
                self.assertEqual(session.page_states[1]['colpali_attempts'], 1)
                self.assertEqual(session.page_states[0]['colpali_error'], '')
        self.assertEqual(calls, [1, 2, 1])
        self.assertEqual(persist.call_count, 2)
        self.assertEqual(structure.call_count, 2)
        indexer.index_dense.assert_called_once_with(run, replace_existing=False)
        self.assertEqual(run.colpali_index_status, 'COMPLETED')
        self.assertEqual(run.colpali_points_count, 2)
        self.assertEqual(run.colpali_index_error, '')
        self.assertEqual(run.structure['processing']['colpali_missing_pages'], [])

    def test_partial_retry_keeps_old_vectors_without_double_counting_or_looping(self):
        run = self.run_object()
        indexer = Mock()
        indexer.index_dense.return_value = IndexResult('COMPLETED', 2)
        with patch('ai_extract.repositories.layout_repository.save_run'), \
             patch('ai_extract.services.document_graph_service.sync_graph', return_value={'status': 'ready'}), \
             patch('ai_extract.services.contract_pipeline_service.PlanACollectionIndexer', return_value=indexer):
            with PageExtractionSession(run) as session:
                session.records = [dict(page_number=1, text='source')]
                run.structure = dict(leaves=session.records)
                session.page_states = [dict(page_number=1, colpali_attempts=1),
                                       dict(page_number=2, colpali_attempts=1),
                                       dict(page_number=3, colpali_attempts=1)]
                session.colpali_results = {1: IndexResult('COMPLETED', 1, point_ids=('good',)),
                    2: IndexResult('PARTIAL', 2, 'first error', ('a', 'b')),
                    3: IndexResult('SKIPPED')}
                retry = Mock(return_value=IndexResult('PARTIAL', 2, 'still unavailable', ('b', 'c')))
                session._index_visual = retry
                session.finish()
        retry.assert_called_once_with(2)
        self.assertEqual(run.colpali_points_count, 4)  # good + a, b, c
        self.assertEqual(run.colpali_index_status, 'PARTIAL')
        self.assertEqual(run.colpali_index_error, 'still unavailable')
        self.assertEqual(run.structure['processing']['colpali_missing_pages'], [2])
        self.assertEqual(session.page_states[1]['colpali_attempts'], 2)

    def test_all_failed_ocr_still_retries_visuals_without_claiming_ocr_success(self):
        run = self.run_object()
        with patch('ai_extract.repositories.layout_repository.save_run'), \
             patch('ai_extract.services.contract_pipeline_service.PlanACollectionIndexer') as indexer:
            with PageExtractionSession(run) as session:
                session.failed_pages = [dict(page_number=1, stage='surya', error_type='RuntimeError')]
                session.page_states = [dict(page_number=1, structure='FAILED', colpali_attempts=1)]
                session.colpali_results = {1: IndexResult('FAILED', error='temporary network failure')}
                retry = Mock(return_value=IndexResult('COMPLETED', 1, point_ids=('p1',)))
                session._index_visual = retry
                session.finish()
        retry.assert_called_once_with(1)
        indexer.assert_not_called()
        self.assertEqual(run.status, 'FAILED')
        self.assertEqual(run.colpali_index_status, 'COMPLETED')
        self.assertEqual(session.page_states[0]['structure'], 'FAILED')

    def test_health_does_not_wait_for_ocr_lock(self):
        from concurrent.futures import ThreadPoolExecutor
        from PIL import Image
        import base64, io
        from ai_extract.views.inference_views import register_contract_rag
        routes = {}
        app = SimpleNamespace(get=lambda path: lambda fn: routes.setdefault(path, fn),
                              post=lambda path: lambda fn: routes.setdefault(path, fn))
        entered, release = Event(), Event()
        def ocr(*args):
            entered.set()
            if not release.wait(3):
                raise AssertionError('Health blocked behind OCR')
            return {'layout': []}
        buffer = io.BytesIO()
        Image.new('RGB', (2, 2)).save(buffer, 'PNG')
        with patch('ai_extract.services.surya_inference_service.create_surya_predictor', return_value=object()), \
             patch('ai_extract.services.surya_inference_service.surya_page', side_effect=ocr), \
             patch.dict('sys.modules', fastapi=SimpleNamespace(HTTPException=RuntimeError)):
            register_contract_rag(app)
            with ThreadPoolExecutor(max_workers=2) as pool:
                task = pool.submit(routes['/rag/extract_page'],
                    {'image_base64': base64.b64encode(buffer.getvalue()).decode()})
                try:
                    self.assertTrue(entered.wait(2))
                    health = pool.submit(routes['/rag/health']).result(timeout=1)
                    self.assertEqual(health['model_family'], 'contract-a3')
                finally:
                    release.set()
                task.result(timeout=2)

    def test_failed_ocr_is_recorded_and_all_failed_document_is_not_completed(self):
        from concurrent.futures import Future
        run = self.run_object()
        result = Future()
        result.set_result(IndexResult('COMPLETED', 1))
        with patch('ai_extract.repositories.layout_repository.save_run'), \
             patch('ai_extract.services.contract_pipeline_service.PlanACollectionIndexer') as indexer:
            with PageExtractionSession(run) as session:
                session.expected_pages = 1
                session.page_states = [dict(page_number=1)]
                session.visual_tasks[1] = result
                session.fail_page(1, RuntimeError())
                self.assertFalse(session.visual_tasks)
                self.assertEqual(session.page_states[0]['structure'], 'FAILED')
                self.assertEqual(run.colpali_points_count, 1)
                session.finish()
        self.assertEqual(run.status, 'FAILED')
        self.assertEqual(run.structure['processing']['failed_pages'][0]['page_number'], 1)
        indexer.assert_not_called()

    def test_prepared_pdf_visual_branch_does_not_reopen_neon(self):
        run = self.run_object()
        page = SimpleNamespace(id=24, page_number=14, image_path='p14.png')
        visuals = [SimpleNamespace(page=page, block=None, image_path='p14.png')]
        indexer = Mock()
        indexer.index_colpali.return_value = IndexResult('COMPLETED', 1)
        with PageExtractionSession(run) as session, \
             patch('ai_extract.repositories.layout_repository.indexed_run') as read, \
             patch('ai_extract.services.layout_extractor.visual_sources') as sources, \
             patch('ai_extract.services.contract_pipeline_service.PlanACollectionIndexer', return_value=indexer):
            session.visual_inputs[14] = (run, visuals)
            result = session._index_visual(14)
        self.assertEqual(result.status, 'COMPLETED')
        read.assert_not_called()
        sources.assert_not_called()
        indexer.index_colpali.assert_called_once_with(run, page_number=14, visuals=visuals)

    def test_incomplete_document_retains_sources_but_never_publishes_final_graph(self):
        indexer = Mock()
        indexer.index_dense.return_value = IndexResult('COMPLETED', 2)
        with patch('ai_extract.repositories.layout_repository.save_run'), \
             patch('ai_extract.services.document_graph_service.sync_graph', return_value={'status': 'ready'}) as publish, \
             patch('ai_extract.services.contract_pipeline_service.PlanACollectionIndexer', return_value=indexer):
            run = self.run_object()
            with PageExtractionSession(run) as session:
                session.records = [dict(page_number=1, text='good source'), dict(page_number=3, text='other source')]
                session.failed_pages = [dict(page_number=2, stage='surya', error_type='RuntimeError')]
                session.colpali_results = {1: IndexResult('COMPLETED', 3)}
                run.structure = dict(leaves=session.records)
                run.total_pages = 2
                session.finish()
        self.assertEqual(run.status, 'FAILED')
        self.assertEqual(run.total_pages, 2)
        indexer.index_dense.assert_not_called()
        publish.assert_not_called()
        self.assertEqual(run.colpali_points_count, 3)
        self.assertIn('Failed pages: 2', run.error_message)
        self.assertEqual(run.structure['processing']['failed_pages'][0]['page_number'], 2)


class RenderedPagePersistenceTests(TestCase):
    def test_ocr_fills_rendered_page_without_duplicate_or_changing_old_run(self):
        from contracts.models import Company, Contract, ContractVersion
        from ai_extract.models import LayoutExtractionRun, LayoutExtractionPage, LayoutExtractionBlock
        from ai_extract.repositories.layout_repository import persist_rendered_page, persist_pages
        company = Company.objects.create(company_name='Test', tax_code='PAGE')
        contract = Contract.objects.create(company=company, contract_code='PAGE', title='Test')
        version = ContractVersion.objects.create(contract=contract)
        old_run = LayoutExtractionRun.objects.create(version=version)
        old_page = LayoutExtractionPage.objects.create(run=old_run, page_number=1, extracted_text='Old source')
        run = LayoutExtractionRun.objects.create(version=version)
        rendered = persist_rendered_page(run, 1, 'page.png', 100, 200)
        block = SimpleNamespace(order=1, text='OCR source', layout_type='paragraph', bbox=[0,0,100,100], confidence=1)
        page = SimpleNamespace(page_number=1, width=100, height=200, image_path='page.png', text='OCR source', blocks=[block])
        records, count = persist_pages(run, SimpleNamespace(pages=[page]), incremental=True)
        rendered.refresh_from_db()
        old_page.refresh_from_db()
        self.assertEqual(run.pages.count(), 1)
        self.assertEqual(rendered.extracted_text, 'OCR source')
        self.assertEqual(count, 1)
        self.assertEqual(records[0]['text'], 'OCR source')
        self.assertEqual(old_page.extracted_text, 'Old source')

        # Returning only the new page must not SELECT any existing block rows.
        from django.db import connection
        from django.test.utils import CaptureQueriesContext
        second = SimpleNamespace(page_number=2, width=100, height=200, image_path='two.png',
                                 text='Second page', blocks=[block])
        with CaptureQueriesContext(connection) as queries:
            current, count = persist_pages(run, SimpleNamespace(pages=[second]), incremental=True, cumulative=False)
        self.assertEqual(count, 1)
        self.assertEqual([r['page_number'] for r in current], [2])
        self.assertNotEqual(current[0]['chunk_id'], records[0]['chunk_id'])
        self.assertFalse(any('SELECT' in q['sql'].upper() and LayoutExtractionBlock._meta.db_table in q['sql']
                             for q in queries))


class ProgressLogTests(SimpleTestCase):
    def test_slow_step_emits_heartbeat_and_stops_it_when_step_completes(self):
        from ai_extract.services.inference_runtime_service import PipelineProgress
        waiting = Event()
        records = []
        progress = PipelineProgress('contract_run_7')
        def emit(stage, event, **fields):
            records.append((stage, event, fields))
            if event == 'waiting':
                waiting.set()
        with patch.object(progress, 'emit', side_effect=emit):
            with progress.stage('surya', heartbeat_seconds=.01, page=2, total_pages=14):
                self.assertTrue(waiting.wait(1))
        self.assertEqual(records[0][1], 'started')
        self.assertEqual(records[-1][1], 'completed')
        self.assertEqual(records[-1][2]['page'], 2)
        self.assertGreaterEqual(records[-1][2]['elapsed_s'], .01)

    def test_failed_step_logs_error_type_and_does_not_swallow_exception(self):
        import json
        from ai_extract.services.inference_runtime_service import PipelineProgress
        with self.assertLogs('ai_extract.pipeline', level='INFO') as logs:
            with self.assertRaises(RuntimeError):
                with PipelineProgress(7).stage('colpali', page=2):
                    raise RuntimeError('Sensitive source content must not enter logs')
        record = json.loads(logs.records[-1].getMessage().removeprefix('A3 '))
        self.assertEqual(record['run'], 7)
        self.assertEqual(record['event'], 'failed')
        self.assertEqual(record['error_type'], 'RuntimeError')
        self.assertNotIn('Sensitive source', str(logs.output))

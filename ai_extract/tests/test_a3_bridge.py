from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from types import SimpleNamespace

from django.test import SimpleTestCase, override_settings
from PIL import Image

from ai_extract.services.contract_pipeline_service import A3ContractPipeline
from ai_extract.services.contract_pipeline_service import build_structure, clauses_from_structure
from ai_extract.services.contract_pipeline_service import IndexResult, LayoutExtractionService, PlanACollectionIndexer


class A3BridgeTests(SimpleTestCase):
    def test_dense_retry_keeps_source_structure_and_visual_branch_unchanged(self):
        structure={'leaves':[{'text':'Existing source'}]}
        run=SimpleNamespace(structure=structure,colpali_index_status='FAILED',colpali_points_count=0)
        indexer=Mock();indexer.index_dense.return_value=IndexResult('COMPLETED',121)
        with patch('ai_extract.services.contract_pipeline_service.PlanACollectionIndexer',return_value=indexer), \
             patch('ai_extract.repositories.layout_repository.indexed_run',return_value=run), \
             patch('ai_extract.repositories.layout_repository.save_run') as save:
            LayoutExtractionService()._index_plan_a_collections(run,dense_only=True)
        indexer.index_colpali.assert_not_called()
        indexer.index_dense.assert_called_once_with(run,replace_existing=False)
        self.assertIs(run.structure,structure)
        self.assertEqual(run.dense_index_status,'COMPLETED')
        self.assertEqual(run.colpali_index_status,'FAILED')
        self.assertTrue(all('structure' not in call.args[1] for call in save.call_args_list))

    def test_reindex_preserves_success_when_other_branch_raises(self):
        for failed_branch in ('dense', 'colpali'):
            with self.subTest(failed_branch=failed_branch):
                indexer = Mock()
                indexer.index_dense.return_value = IndexResult('COMPLETED', 7)
                indexer.index_colpali.return_value = IndexResult('COMPLETED', 3)
                getattr(indexer, 'index_' + failed_branch).side_effect = RuntimeError('model offline')
                run = SimpleNamespace(structure={})
                with patch('ai_extract.services.contract_pipeline_service.PlanACollectionIndexer', return_value=indexer), \
                     patch('ai_extract.repositories.layout_repository.indexed_run', return_value=run), \
                     patch('ai_extract.repositories.layout_repository.save_run'):
                    LayoutExtractionService()._index_plan_a_collections(run)
                successful_branch = 'colpali' if failed_branch == 'dense' else 'dense'
                self.assertEqual(getattr(run, failed_branch + '_index_status'), 'FAILED')
                self.assertEqual(getattr(run, failed_branch + '_index_error'), 'model offline')
                self.assertEqual(getattr(run, successful_branch + '_index_status'), 'COMPLETED')
                self.assertGreater(getattr(run, successful_branch + '_points_count'), 0)

    def test_page_index_summary_retains_partial_points_and_failure_details(self):
        summary = IndexResult.combine([
            IndexResult('COMPLETED', 2), IndexResult('FAILED', error='page 2 timed out'),
            IndexResult('PARTIAL', 1, 'page 3 image missing'),
        ])
        run = SimpleNamespace()
        summary.apply_to(run, 'colpali')
        self.assertEqual(run.colpali_index_status, 'PARTIAL')
        self.assertEqual(run.colpali_points_count, 3)
        self.assertIn('page 2 timed out', run.colpali_index_error)
        self.assertIn('page 3 image missing', run.colpali_index_error)

    def test_page_index_summary_distinguishes_no_images_from_failure(self):
        self.assertEqual(IndexResult.combine([]).status, 'SKIPPED')
        self.assertEqual(IndexResult.combine([IndexResult('SKIPPED')]).status, 'SKIPPED')
        self.assertEqual(IndexResult.combine([IndexResult('FAILED')]).status, 'FAILED')
        self.assertEqual(IndexResult.combine([IndexResult('COMPLETED', 1)]).status, 'COMPLETED')

    def test_colpali_retains_acknowledged_batch_if_last_write_fails(self):
        client = Mock()
        client.upsert.side_effect = [None, RuntimeError('last batch unavailable')]
        response = Mock()
        response.json.return_value = {'embeddings': [[1.0] * 128]}
        run = SimpleNamespace(id=8, version_id=9)
        with TemporaryDirectory() as directory, override_settings(MEDIA_ROOT=Path(directory)), \
             patch('ai_extract.services.contract_pipeline_service.PlanAQdrantClient.get', return_value=client), \
             patch.object(PlanACollectionIndexer, '_colpali_url', return_value='http://test/colpali'), \
             patch.object(PlanACollectionIndexer, '_ensure_collection'), \
             patch.object(PlanACollectionIndexer, '_payload', return_value={}), \
             patch('ai_extract.services.contract_pipeline_service.requests.post', return_value=response):
            Image.new('RGB', (2, 2)).save(Path(directory) / 'page.png')
            page = SimpleNamespace(id=42, page_number=1)
            visuals = [SimpleNamespace(page=page, block=SimpleNamespace(pk=i, layout_type='figure', text=''), image_path='page.png')
                       for i in range(6)]
            result = PlanACollectionIndexer().index_colpali(run, page_number=1, visuals=visuals)
        self.assertEqual(result.status, 'PARTIAL')
        self.assertEqual(result.indexed_count, 5)
        acknowledged = client.upsert.call_args_list[0].kwargs['points']
        self.assertEqual(set(result.point_ids), {str(point.id) for point in acknowledged})
        self.assertIn('last batch unavailable', result.error)

    def test_uses_a3_and_preserves_page_image_bbox_and_text(self):
        with TemporaryDirectory() as directory, override_settings(MEDIA_ROOT=Path(directory)):
            output = Path(directory) / 'extracted_images'
            output.mkdir()
            Image.new('RGB', (300, 600)).save(output / 'pdf_page_contract_run_8_p2.png')
            fake = SimpleNamespace(surya_layout_model=object(), extract_document=lambda doc: [
                dict(page_number=2, bbox=[0, 0, 1000, 500], text='Payment 30 days',
                     layout_type='table', confidence=0.87)])
            with patch('ai_extract.services.layout_extractor.SuryaExtractor', return_value=fake):
                result = A3ContractPipeline().extract(Path(directory) / 'contract.pdf', document_key='contract_run_8')
            page = result.pages[0]
            self.assertEqual((page.page_number, page.width, page.height), (2, 300, 600))
            self.assertEqual(page.blocks[0].bbox, [0, 0, 1000, 500])
            self.assertEqual(page.text, 'Payment 30 days')
            self.assertEqual(page.image_path, 'extracted_images/pdf_page_contract_run_8_p2.png')
            self.assertTrue(fake.strict)

    def test_missing_surya_cannot_report_success(self):
        fake = SimpleNamespace(surya_layout_model=None)
        with patch('ai_extract.services.layout_extractor.SuryaExtractor', return_value=fake):
            with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                A3ContractPipeline().extract(Path('contract.pdf'), document_key='run')

    def test_office_content_does_not_claim_physical_page_image(self):
        fake = SimpleNamespace(surya_layout_model=None, extract_document=lambda doc: [dict(text='Article 1', bbox=[])])
        with patch('ai_extract.services.layout_extractor.SuryaExtractor', return_value=fake):
            result = A3ContractPipeline().extract(Path('contract.docx'), document_key='run')
        self.assertEqual(result.pages[0].image_path, '')
        self.assertEqual(result.pages[0].blocks[0].bbox, [])

    def test_a3_clause_materialization_spans_pages(self):
        tree = build_structure([dict(chunk_id='h1',page_number=1,text='Article 1. Payment',layout_type='title',
                                     structure_metadata=dict(heading_level=1,structure_kind='article')),
                                dict(chunk_id='p1',page_number=1,text='Pay within'),
                                dict(chunk_id='p2',page_number=2,text='30 days.'),
                                dict(chunk_id='h2',page_number=2,text='Article 2. Term',layout_type='title',
                                     structure_metadata=dict(heading_level=1,structure_kind='article')),
                                dict(chunk_id='p3',page_number=2,text='One year.')])
        clauses = clauses_from_structure(tree)
        self.assertEqual(len(clauses), 2)
        self.assertIn('30 days', clauses[0]['content'])
        self.assertEqual(clauses[0]['end_page'], 2)

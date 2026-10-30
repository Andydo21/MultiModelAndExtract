"""Figure-only visual indexing and source identity across Qdrant, graph and A2."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
from contextlib import nullcontext
from copy import deepcopy

from PIL import Image
from django.test import SimpleTestCase, override_settings

from ai_extract.services.layout_extractor import visual_sources, colpali_point_id
from ai_extract.services.contract_pipeline_service import hydrate, IndexResult, PlanACollectionIndexer
from ai_extract.services.document_graph_service import projection
from ai_extract.tests.test_document_graph import fixture
from ai_extract.tests.test_hybrid_retrieval import version
from shared.retrieval import LexicalOnlyReranker


class ColPaliFigureTests(SimpleTestCase):
    def test_colpali_only_cleanup_preserves_dense_collection(self):
        from ai_extract.services.contract_pipeline_service import DENSE_COLLECTION,COLPALI_COLLECTION
        run=NS(pk=7,status='COMPLETED',version_id=3,version=NS(contract=NS(pk=2,company_id=1)),
               structure=dict(leaves=[dict(layout_type='figure')],colpali_point_ids=['image-id']),
               colpali_index_status='COMPLETED',colpali_points_count=1,dense_index_status='COMPLETED')
        indexer=object.__new__(PlanACollectionIndexer);indexer.client=Mock()
        indexer.client.get_collections.return_value=NS(collections=[NS(name=n) for n in (DENSE_COLLECTION,COLPALI_COLLECTION)])
        result=indexer.prune_old_run_points(run,branches=('colpali',))
        self.assertEqual(result,dict(colpali='completed'))
        self.assertTrue(indexer.client.delete.called)
        self.assertTrue(all(c.kwargs['collection_name']==COLPALI_COLLECTION for c in indexer.client.delete.call_args_list))
    def test_empty_or_duplicate_visual_inputs_skip_remote_and_vector_store_work(self):
        indexer=object.__new__(PlanACollectionIndexer);indexer.client=Mock()
        run=NS(id=7);visual=NS(page=NS(page_number=1),block=NS(pk=10,layout_type='figure'))
        with patch.object(indexer,'_colpali_url',return_value='http://visual.test'), \
             patch('ai_extract.services.contract_pipeline_service.requests.post') as post:
            self.assertEqual(indexer.index_colpali(run,visuals=[]).status,'SKIPPED')
            failed=indexer.index_colpali(run,visuals=[visual,visual])
        self.assertEqual(failed.status,'FAILED')
        self.assertIn('Duplicate ColPali source block IDs',failed.error)
        self.assertEqual(indexer.client.mock_calls,[])
        post.assert_not_called()
    @override_settings(COLPALI_ENABLED=False)
    def test_disabled_colpali_skips_without_remote_or_database_calls(self):
        indexer = object.__new__(PlanACollectionIndexer)
        with patch('ai_extract.services.contract_pipeline_service.requests.post') as post:
            result = indexer.index_colpali(NS(id=7), page_number=1)
        self.assertEqual(result.status, 'SKIPPED')
        self.assertEqual(indexer._colpali_url(), '')
        post.assert_not_called()

    def run_with(self, blocks, path='page.png'):
        page = NS(id=8, pk=8, page_number=1, image_path=path, blocks=Mock())
        page.blocks.all.return_value = blocks
        run = NS(id=7, pk=7, pages=Mock())
        run.pages.prefetch_related.return_value.order_by.return_value = [page]
        run.pages.filter.return_value = run.pages
        return run, page

    def test_png_is_exact_figure_crop_and_text_table_blocks_are_not_indexed(self):
        blocks = [NS(pk=10,layout_type='figure',bbox=[200,100,600,900]),
                  NS(pk=11,layout_type='paragraph'), NS(pk=12,layout_type='table')]
        run, _ = self.run_with(blocks)
        with TemporaryDirectory() as directory, override_settings(MEDIA_ROOT=directory):
            image = Image.new('RGB',(100,100),'red')
            image.paste('blue',(20,10,60,90)); image.save(Path(directory)/'page.png')
            sources = list(visual_sources(run))
            self.assertEqual(len(sources),1)
            self.assertEqual(sources[0].block.pk,10)
            with Image.open(Path(directory)/sources[0].image_path) as crop:
                self.assertEqual(crop.format,'PNG')
                self.assertEqual(crop.size,(40,80))
                self.assertEqual(crop.getpixel((0,0)),(0,0,255))

    def test_text_only_page_has_no_full_page_fallback(self):
        run, _ = self.run_with([NS(pk=1,layout_type='paragraph')])
        self.assertEqual(list(visual_sources(run)),[])

    def test_full_page_input_cannot_reach_remote_embed(self):
        indexer = object.__new__(PlanACollectionIndexer)
        indexer.client = Mock()
        with patch.object(indexer,'_colpali_url',return_value='http://test'), \
             patch.object(indexer,'_ensure_collection'), \
             patch('ai_extract.services.contract_pipeline_service.requests.post') as post:
            result = indexer.index_colpali(NS(id=7),page_number=1,
                visuals=[NS(page=NS(page_number=1),block=None,image_path='page.png')])
        self.assertEqual(result.status,'FAILED')
        post.assert_not_called()

    def test_written_point_keeps_figure_bbox_and_graph_identity(self):
        block = NS(id=10,pk=10,layout_type='figure',bbox=[200,100,600,900],
                   text='',structure_metadata={})
        run, page = self.run_with([block])
        run.version_id = 3
        run.version = NS(contract=NS(id=2,company_id=1,title='Manual',contract_code='C'))
        indexer = object.__new__(PlanACollectionIndexer)
        indexer.client = Mock()
        response = Mock()
        response.json.return_value = {'embeddings':[[1.0]*128]}
        with TemporaryDirectory() as directory, override_settings(MEDIA_ROOT=directory,MEDIA_URL='/media/'), \
             patch.object(indexer,'_colpali_url',return_value='http://test'), \
             patch.object(indexer,'_ensure_collection'), \
             patch('ai_extract.services.contract_pipeline_service.requests.post',return_value=response):
            Image.new('RGB',(100,100)).save(Path(directory)/'page.png')
            sources = list(visual_sources(run))
            result = indexer.index_colpali(run,page_number=1,visuals=sources)
        point = indexer.client.upsert.call_args.kwargs['points'][0]
        self.assertEqual(result.status,'COMPLETED')
        self.assertEqual(result.point_ids,(colpali_point_id(7,8,10),))
        self.assertEqual(point.payload['visual_block_id'],10)
        self.assertEqual(point.payload['bbox'],block.bbox)
        self.assertEqual(point.payload['visual_node_id'],'layout-block:1:block:10')
        self.assertEqual(point.payload['image_path'],sources[0].image_path)
        self.assertEqual(point.payload['visual_scope'],'figure')
        self.assertEqual(point.payload['image_id'],point.id)
        self.assertEqual(len(point.payload['image_sha256']),64)

    def test_backfill_reuses_matching_image_vector_and_reembeds_changed_image_with_same_id(self):
        block=NS(id=10,pk=10,layout_type='figure',bbox=[0,0,1000,1000],text='',structure_metadata={})
        run,page=self.run_with([block]);run.version_id=3
        run.version=NS(contract=NS(id=2,company_id=1,title='Manual',contract_code='C'))
        indexer=object.__new__(PlanACollectionIndexer);indexer.client=Mock()
        indexer.client.retrieve.return_value=[]
        response=Mock();response.json.return_value={'embeddings':[[1.]*128]}
        with TemporaryDirectory() as directory,override_settings(MEDIA_ROOT=directory), \
             patch.object(indexer,'_colpali_url',return_value='http://test'), \
             patch.object(indexer,'_ensure_collection'), \
             patch('ai_extract.services.contract_pipeline_service.requests.post',return_value=response) as post:
            path=Path(directory)/'page.png';Image.new('RGB',(8,8),'red').save(path)
            first=indexer.index_colpali(run,visuals=list(visual_sources(run)),skip_existing=True)
            point=indexer.client.upsert.call_args.kwargs['points'][0]
            indexer.client.retrieve.return_value=[NS(id=point.id,payload=deepcopy(point.payload))]
            post.reset_mock();indexer.client.upsert.reset_mock()
            second=indexer.index_colpali(run,visuals=list(visual_sources(run)),skip_existing=True)
            self.assertEqual(first.point_ids,second.point_ids)
            self.assertEqual(second.status,'COMPLETED')
            self.assertTrue(second.verified_inventory)
            post.assert_not_called();indexer.client.upsert.assert_not_called()
            Image.new('RGB',(8,8),'blue').save(path)
            third=indexer.index_colpali(run,visuals=list(visual_sources(run)),skip_existing=True)
            self.assertEqual(third.point_ids,first.point_ids)
            post.assert_called_once()
            updated=indexer.client.upsert.call_args.kwargs['points'][0]
            self.assertNotEqual(updated.payload['image_sha256'],point.payload['image_sha256'])

    def test_deferred_job_only_embeds_images_and_links_acknowledged_ids(self):
        from ai_extract.services.layout_service import backfill_colpali
        block=NS(id=10,pk=10,layout_type='figure',bbox=[0,0,1000,1000],text='',structure_metadata={})
        run,page=self.run_with([block]);run.version_id=3
        run.version=NS(contract=NS(id=2,company_id=1,title='Manual',contract_code='C'))
        run.status='COMPLETED';run.error_message='Original OCR warning'
        run.structure=dict(source_fingerprint='source',knowledge_graph=dict(entities=['unchanged']))
        run.colpali_points_count=0;run.colpali_index_status='SKIPPED'
        run.dense_index_status='COMPLETED';run.dense_points_count=7
        image_id=colpali_point_id(run.id,page.id,block.pk)
        indexer=Mock();indexer.index_colpali.return_value=IndexResult('COMPLETED',1,point_ids=(image_id,))
        with TemporaryDirectory() as directory,override_settings(MEDIA_ROOT=directory,COLPALI_ENABLED=True), \
             patch('ai_extract.services.layout_service.layout_repository.workflow_slot',return_value=nullcontext(True)), \
             patch('ai_extract.services.layout_service.layout_repository.indexed_run',return_value=run), \
             patch('ai_extract.services.layout_service.layout_repository.save_run') as save, \
             patch('ai_extract.services.contract_pipeline_service.PlanACollectionIndexer',return_value=indexer), \
             patch('ai_extract.services.document_graph_service.sync_graph',return_value=dict(status='ready')), \
             patch('NewIdea.services.knowledge_search_service.refresh_snapshot'), \
             patch('ai_extract.services.gliner_entity_service.extract_graph_entities') as entities, \
             patch('ai_extract.services.contract_pipeline_service._BgeM3Embedding.embed_batch') as dense:
            Image.new('RGB',(8,8),'red').save(Path(directory)/'page.png')
            result=backfill_colpali(run)
            indexer.index_colpali.return_value=IndexResult('FAILED',error='Endpoint offline')
            retried=backfill_colpali(run)
            retained=deepcopy(run.structure)
            indexer.index_colpali.return_value=IndexResult('FAILED',error='Known vector missing; model offline',verified_inventory=True)
            missing=backfill_colpali(run)
        self.assertEqual(result['status'],'COMPLETED')
        self.assertEqual(run.status,'COMPLETED');self.assertEqual(run.error_message,'Original OCR warning')
        self.assertEqual(run.dense_points_count,7)
        self.assertEqual(run.structure['knowledge_graph'],dict(entities=['unchanged']))
        self.assertEqual(run.structure['colpali_images'][0]['image_id'],image_id)
        self.assertEqual(retained['colpali_images'][0]['state'],'indexed')
        self.assertEqual(retained['colpali_point_ids'],[image_id])
        self.assertEqual(retried['status'],'PARTIAL')
        self.assertEqual(retried['points'],1)
        self.assertEqual(missing['points'],0)
        self.assertEqual(missing['status'],'FAILED')
        self.assertEqual(run.structure['colpali_point_ids'],[])
        self.assertEqual(run.structure['colpali_images'][0]['state'],'pending')
        self.assertTrue(indexer.index_colpali.call_args.kwargs['skip_existing'])
        indexer.prune_old_run_points.assert_called_once_with(run,branches=('colpali',))
        entities.assert_not_called();dense.assert_not_called()
        self.assertTrue(all(not any(f.startswith('dense_') or f=='status' for f in c.args[1]) for c in save.call_args_list))

    def test_graph_link_only_exists_for_acknowledged_figure_point(self):
        run, tree = fixture()
        leaf = tree['leaves'][0]
        leaf.update(layout_type='figure',image_path='crop.png',page_image_path='page.png')
        block = NS(pk=10,layout_type='figure')
        fake, page = self.run_with([block])
        run.pages = fake.pages
        point_id = colpali_point_id(run.pk,page.pk,block.pk)
        for ids in ([],[point_id]):
            tree['colpali_point_ids'] = ids
            _,nodes,edges = projection(run,tree)
            links = [e for e in edges if e['type']=='HAS_EMBEDDING']
            self.assertEqual(len(links),len(ids))
            if ids:
                vector = next(n for n in nodes if n['category']=='visual_embedding')
                self.assertEqual(vector['point_id'],point_id)
                self.assertEqual(vector['storage'],'qdrant')
                self.assertTrue(links[0]['from_key'].endswith('layout-block:1:block:10'))

    def test_same_page_figures_do_not_hydrate_other_regions_or_legacy_page_hits(self):
        leaves = [dict(chunk_id=f'leaf{i}',source_chunk_id=f'block:{i}',page_number=1,
                       text='',heading_path=[],image_path='page.png') for i in (10,11,12)]
        scope = dict(source_app='contract_dl',company_id=1,contract_id=2,version_id=3,
                     contract_dl_run_id=7,page_number=1,visual_scope='figure')
        points = [NS(id=str(i),score=.9,payload=dict(scope,visual_block_id=i,image_path=f'crop{i}.png'))
                  for i in (10,11)]
        points.append(NS(id='old',score=.9,payload=dict(scope,visual_scope='page')))
        result = hydrate(points,leaves,version(),7,True)
        self.assertEqual([r['chunk_id'] for r in result],['leaf10','leaf11'])
        self.assertEqual([r['image_path'] for r in result],['crop10.png','crop11.png'])
        fused = LexicalOnlyReranker.reciprocal_rank_fusion({'keyword':[leaves[0]],'colpali':result})
        self.assertEqual(fused[0]['image_path'],'crop10.png')
        self.assertEqual(fused[0]['colpali_point_id'],'10')

    def test_skipped_text_pages_do_not_mark_successful_figure_index_partial(self):
        result = IndexResult.combine([IndexResult('SKIPPED'),IndexResult('COMPLETED',1,point_ids=('x',))])
        self.assertEqual(result.status,'COMPLETED')
        self.assertEqual(result.point_ids,('x',))

    def test_figure_image_to_text_preserves_caption_and_in_image_ocr(self):
        from ai_extract.services.layout_extractor import SuryaExtractor, extraction_metadata
        extractor = SuryaExtractor()
        # Isolated model-output fixture: an explicit source parent owns caption.
        fig_region = NS(region_id='fig:1', label='picture', bbox=[100, 100, 400, 300], confidence=0.95)
        cap_region = NS(region_id='cap:1',parent_region_id='fig:1', label='figure_caption', bbox=[100, 310, 400, 330], confidence=0.98)
        lines = [
            dict(region_id='fig:1', text='Driver 1', bbox=[120, 150, 200, 180]),
            dict(region_id='fig:1', text='Port EtherCAT', bbox=[220, 150, 350, 180]),
            dict(region_id='cap:1', text='Hình 1. Sơ đồ mạch điều khiển', bbox=[100, 310, 400, 330]),
        ]
        img = Image.new('RGB', (1000, 1000), 'white')
        chunks, _ = extractor._chunks_from_surya_regions(
            [fig_region, cap_region], lines, img, page_num=1, doc_id=1,
            image_path='page.png', image_url='/media/page.png', start_chunk_id=1
        )
        fig_chunk = next(c for c in chunks if c['layout_type'] == 'figure')
        self.assertIn('[Hình 1. Sơ đồ mạch điều khiển]', fig_chunk['text'])
        self.assertIn('Port EtherCAT', fig_chunk['text'])
        self.assertEqual(fig_chunk['in_image_text'], ['Driver 1', 'Port EtherCAT'])
        self.assertEqual(fig_chunk['caption_text'], 'Hình 1. Sơ đồ mạch điều khiển')

        meta = extraction_metadata(fig_chunk, Path('.'))
        self.assertTrue(meta['display_only'])
        self.assertEqual(meta['text_source'],'visual_display_label')
        from ai_extract.services.document_structure_service import build_document_structure
        tree=build_document_structure([dict(chunk_id='figure',text=fig_chunk['text'],
            layout_type='figure',structure_metadata=meta)])
        self.assertEqual(tree['leaves'][0]['text'],'')
        self.assertEqual(tree['leaves'][0]['structure_metadata']['display_text'],fig_chunk['text'])
        self.assertEqual(meta['caption_text'], 'Hình 1. Sơ đồ mạch điều khiển')
        self.assertEqual(meta['in_image_text'], ['Driver 1', 'Port EtherCAT'])
        del cap_region.parent_region_id
        unlinked,_=extractor._chunks_from_surya_regions([fig_region,cap_region],lines,img,
            page_num=1,doc_id=1,image_path='page.png',image_url='/media/page.png',start_chunk_id=1)
        figure=next(c for c in unlinked if c['layout_type']=='figure')
        self.assertNotIn('caption_text',figure)
        self.assertTrue(any(c['layout_type']=='figure_caption' and 'Hình 1.' in c['text'] for c in unlinked))


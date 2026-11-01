"""Source lineage, template escaping and access scope for the visualization."""
from unittest.mock import patch
from django.test import TestCase, RequestFactory
from django.contrib.auth import get_user_model
from contracts.models import Company, Contract, ContractVersion
from ai_extract.models import LayoutExtractionRun, LayoutExtractionPage, LayoutExtractionBlock
from ai_extract.services.document_structure_service import build_document_structure
from ai_extract.services.graph_visualization_service import tree_payload, page_payload
from ai_extract.services.layout_service import inspect_page
from ai_extract.views.layout_views import page_inspector
from ai_extract.services.contract_pipeline_service import _request_openai_compatible, structural_retrieve


class VisualizationTests(TestCase):
    def setUp(self):
        self.company = Company.objects.create(company_name='Preview', tax_code='UI')
        self.user = get_user_model().objects.create_user(username='ui-reader', company=self.company)
        self.contract = Contract.objects.create(company=self.company, contract_code='UI-1', title='Maintenance manual')
        self.version = ContractVersion.objects.create(contract=self.contract)
        self.run = LayoutExtractionRun.objects.create(version=self.version, status='COMPLETED')
        self.page = LayoutExtractionPage.objects.create(run=self.run, page_number=1, image_path='page.png')
        self.block = LayoutExtractionBlock.objects.create(page=self.page, block_order=1, text='Maintenance', bbox=[20,20,800,100])
        self.unknown = LayoutExtractionBlock.objects.create(page=self.page, block_order=2, text='Unrecognized source', bbox=[20,120,800,200])
        records = [dict(chunk_id=f'block:{b.pk}', text=b.text, bbox=b.bbox, page_number=1, image_path='page.png') for b in (self.block,self.unknown)]
        records[0]['layout_type']='title'
        self.run.structure = build_document_structure(records, self.contract.title)
        observation=dict(id='model-heading',kind='section_header',text='Maintenance',level=1,bbox=[20,20,800,100])
        self.run.structure['nodes'][1]['model_evidence']=observation
        self.run.structure['leaves'][0]['structure_metadata']['granite_observation']=observation
        self.run.structure.update(structure_method='granite_observations_with_source_layout',
            analysis=dict(visual_pages=[dict(page_number=1,items=[observation])]))
        self.run.save(update_fields=['structure'])

    def test_observed_boxes_and_source_matches_are_persisted(self):
        payload = page_payload(self.run, self.page)
        self.assertEqual(payload['observations'][0]['items'][0]['bbox'], [20,20,800,100])
        self.assertTrue(payload['blocks'][0]['classified'])
        self.assertFalse(payload['blocks'][1]['classified'])
        self.assertEqual(payload['blocks'][0]['units'][0]['granite']['id'], 'model-heading')

    def test_partial_page_preview_pins_run_and_cannot_cross_version(self):
        partial = LayoutExtractionRun.objects.create(version=self.version, status='FAILED')
        page = LayoutExtractionPage.objects.create(run=partial, page_number=1, extracted_text='Retained source')
        context = inspect_page(self.user, self.contract.pk, 1, self.version.pk, run_id=partial.pk)
        self.assertEqual(context['page'].pk, page.pk)
        other_version = ContractVersion.objects.create(contract=self.contract)
        with self.assertRaises(LayoutExtractionRun.DoesNotExist):
            inspect_page(self.user, self.contract.pk, 1, other_version.pk, run_id=partial.pk)

    def test_tree_links_pin_exact_version_and_source_block(self):
        payload = tree_payload(self.run.structure, self.contract.pk, self.version.pk)
        self.assertTrue(payload['leaves'][0]['source_url'].endswith(f'?version_id={self.version.pk}#block-{self.block.pk}'))
        self.assertNotIn('visual_pages', payload['analysis'])

    def test_old_data_does_not_invent_granite_observations(self):
        self.run.structure = {}
        payload = page_payload(self.run, self.page)
        self.assertEqual(payload['observations'], [])
        self.assertFalse(payload['blocks'][0]['classified'])

    def test_html_has_overlay_controls_and_escaped_source_json(self):
        self.block.text = '<script>alert(1)</script>'
        self.block.save(update_fields=['text'])
        self.run.structure['nodes'][1]['model_evidence']['text'] = '</script><script>alert(1)</script>'
        self.run.save(update_fields=['structure'])
        request = RequestFactory().get('/', {'version_id':self.version.pk})
        request.user = self.user
        html = page_inspector(request, self.contract.pk, 1).content.decode()
        self.assertIn('id="show-granite"',html)
        self.assertIn('id="only-uncertain"',html)
        self.assertIn('id="page-visualization"',html)
        self.assertNotIn('<script>alert(1)</script>',html)
        self.assertIn('\\u003C/script\\u003E',html)

    def test_other_company_cannot_read_visualization(self):
        other = Company.objects.create(company_name='Other',tax_code='UI-other')
        outsider = get_user_model().objects.create_user(username='outsider-ui',company=other)
        with self.assertRaises(Contract.DoesNotExist):
            inspect_page(outsider,self.contract.pk,1,self.version.pk)

    def test_generation_metadata_reports_confirmed_input_and_combine_source_ids(self):
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value='Maintenance [E1]'):
            result = _request_openai_compatible('http://unused','contract_qa',dict(query='Maintenance',evidence=[dict(evidence_id='E1',text='Maintenance')]))
            combined = _request_openai_compatible('http://unused','contract_qa',dict(query='Maintenance',evidence=[],review_stage='combine',allowed_ids=['E1'],findings='Maintenance [E1]'))
        self.assertEqual(result['generation_input']['evidence_ids'],['E1'])
        self.assertEqual(result['generation_input']['delivery'],'confirmed_gateway_response')
        self.assertEqual(result['generation_input']['images'],[])
        self.assertEqual(combined['generation_input']['derived_ids'],['E1'])
        self.assertEqual(combined['generation_input']['evidence_ids'],[])

    def test_trace_keeps_seed_ranks_when_expanding_structure(self):
        leaves=self.run.structure['leaves']
        selected, _=structural_retrieve('Maintenance',self.run.structure,seeds=[dict(leaves[0],rrf_score=.04,rerank_score=2)])
        self.assertEqual(selected[0]['rrf_score'],.04)
        self.assertEqual(selected[0]['expansion_reason'],'query_match')

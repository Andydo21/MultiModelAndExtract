"""A3 persistence -> NewIdea search, with model/index boundaries mocked."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from django.contrib.auth import get_user_model
from django.test import TestCase, Client, override_settings
from contracts.models import Company, Contract, ContractVersion, ContractFile
from contracts.services.document_encryption_service import encrypt_pdf
from ai_extract.models import LayoutExtractionRun, LayoutExtractionPage, LayoutExtractionBlock
from ai_extract.services.contract_pipeline_service import LayoutExtractionService, IndexResult


class ContractSearchTests(TestCase):
    def setUp(self):
        self.company=Company.objects.create(company_name='A',tax_code='A')
        other=Company.objects.create(company_name='B',tax_code='B')
        self.user=get_user_model().objects.create_user(username='searcher',company=self.company)
        self.contract=Contract.objects.create(company=self.company,contract_code='HD-A3',title='Supply',contract_type='Supply')
        self.version=ContractVersion.objects.create(contract=self.contract,version_number=1)
        self.foreign=Contract.objects.create(company=other,contract_code='FOREIGN',title='Private foreign contract')
        self.foreign_version=ContractVersion.objects.create(contract=self.foreign)
        self.client.force_login(self.user)

    def post(self, **updates):
        data=dict(query='payment',contract_id=self.contract.pk,version_id=self.version.pk)
        data.update(updates)
        return self.client.post('/api/newidea/ask/',json.dumps(data),content_type='application/json')

    def persist_source(self, version=None, text='Article 1. Payment\nPay within 15 days.'):
        run=LayoutExtractionRun.objects.create(version=version or self.version,status='COMPLETED',dense_index_status='COMPLETED',colpali_index_status='COMPLETED')
        page=LayoutExtractionPage.objects.create(run=run,page_number=1,extracted_text=text)
        block=LayoutExtractionBlock.objects.create(page=page,block_order=1,text=text,bbox=[10,20,900,400])
        return run,block

    def test_extracted_file_is_searchable_without_copying_into_factory_database(self):
        with TemporaryDirectory() as directory, override_settings(MEDIA_ROOT=Path(directory)):
            path=Path(directory)/'contract.txt'
            path.write_bytes(encrypt_pdf(b'Article 1. Payment\nThe buyer pays 7300 USD within 15 days.'))
            ContractFile.objects.create(version=self.version,file_name='contract.txt',file_path='contract.txt')
            def extract(doc):
                return [dict(page_number=1,text=Path(doc.file.path).read_text(),
                             bbox=[10,20,900,400],layout_type='paragraph')]
            extractor=SimpleNamespace(surya_layout_model=None,extract_document=extract)
            indexer=SimpleNamespace(index_dense=lambda run:IndexResult('COMPLETED',1),
                                    index_colpali=lambda run:IndexResult('SKIPPED'))
            with patch('ai_extract.services.layout_extractor.SuryaExtractor',return_value=extractor), \
                 patch('ai_extract.services.contract_pipeline_service.PlanACollectionIndexer',return_value=indexer), \
                 patch('ai_extract.services.contract_pipeline_service.request_qwen',side_effect=RuntimeError('Model offline')):
                run=LayoutExtractionService().extract_version(self.version)
            self.assertEqual(run.status,'COMPLETED')
            response=self.post()
            self.assertEqual(response.status_code,200)
            body=response.json()
            self.assertEqual(body['search_engine'],'newidea_a3')
            self.assertIn('7300 USD',' '.join(c['text'] for c in body['citations']))
            # Plain paragraph OCR supplies no heading level. The removed
            # article-name rules must not invent structure from this text.
            self.assertEqual(body['context_coverage']['structure_status'], 'unstructured')
            self.assertEqual(body['citations'][0]['version_id'],self.version.pk)
            self.assertIn('version_id='+str(self.version.pk),body['citations'][0]['source_url'])

    @override_settings(A3_HYBRID_ENABLED=True)
    def test_newidea_uses_dense_and_colpali_of_the_same_completed_a3_run(self):
        run,block=self.persist_source()
        point=SimpleNamespace(score=.9,payload=dict(source_app='contract_dl',company_id=self.company.pk,
            contract_id=self.contract.pk,version_id=self.version.pk,contract_dl_run_id=run.pk,
            chunk_id=block.pk,page_number=1))
        with patch('ai_extract.services.contract_pipeline_service.HybridBackend.dense',return_value=[point]) as dense, \
             patch('ai_extract.services.contract_pipeline_service.HybridBackend.colpali',return_value=[point]) as visual, \
             patch('ai_extract.services.contract_pipeline_service.HybridBackend.rerank',side_effect=lambda q,c:c):
            body=self.post().json()
        self.assertEqual(body['context_coverage']['retrieval_mode'],'hybrid')
        self.assertEqual(body['context_coverage']['fusion_weights'],dict(keyword=1.0,dense=1.0,colpali=1.0))
        self.assertEqual(dense.call_args.args[1].pk,self.version.pk)
        self.assertEqual(visual.call_args.args[2],run.pk)

    def test_other_company_contract_is_rejected_before_retrieval(self):
        with patch('NewIdea.services.contract_search_service.retrieve_hybrid') as retrieval:
            self.assertEqual(self.post(contract_id=self.foreign.pk).status_code,404)
        retrieval.assert_not_called()

    def test_version_of_other_contract_is_rejected(self):
        self.assertEqual(self.post(version_id=self.foreign_version.pk).status_code,404)

    def test_anonymous_and_missing_contract_are_rejected(self):
        self.client.logout()
        self.assertEqual(self.post().status_code,401)
        self.client.force_login(self.user)
        self.assertEqual(self.post(contract_id=None).status_code,400)

    def test_workspace_links_to_newidea_endpoint_and_filters_contracts(self):
        response=self.client.get(f'/newidea/contracts/{self.contract.pk}/')
        self.assertContains(response,'NewIdea')
        self.assertContains(response,f'/api/newidea/contracts/{self.contract.pk}/search/')
        self.assertNotContains(response,'Private foreign contract')
        self.assertEqual(self.client.get(f'/newidea/contracts/{self.foreign.pk}/').status_code,404)

    def test_gateway_outage_retains_sources_and_explains_fallback(self):
        from ai_extract.services.contract_pipeline_service import QwenGatewayError
        self.persist_source()
        with patch('ai_extract.services.contract_pipeline_service.request_qwen', side_effect=QwenGatewayError('offline')):
            response = self.post(generate=True)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['generation_status'], 'source_excerpt')
        self.assertEqual(data['generation_diagnostics']['state'], 'gateway_unavailable')
        self.assertTrue(data['citations'])

    def test_selected_version_does_not_mix_old_and_new_source(self):
        self.persist_source(text='Article 1. Payment\nPayment within 15 days.')
        newer=ContractVersion.objects.create(contract=self.contract,version_number=2)
        self.persist_source(newer,'Article 1. Payment\nPayment within 60 days.')
        body=self.post(version_id=newer.pk).json()
        self.assertIn('60 days',' '.join(c['text'] for c in body['citations']))
        self.assertNotIn('15 days',' '.join(c['text'] for c in body['citations']))

    def test_newidea_search_retains_csrf_protection(self):
        client=Client(enforce_csrf_checks=True);client.force_login(self.user)
        response=client.post('/api/newidea/ask/',json.dumps(dict(query='payment',contract_id=self.contract.pk)),
                             content_type='application/json')
        self.assertEqual(response.status_code,403)

    def test_pipeline_rejects_extraction_of_other_company_contract(self):
        from ai_extract.views.layout_views import api_run_layout_extraction
        from django.test import RequestFactory
        request=RequestFactory().post('/extract/');request.user=self.user
        from django.http import Http404
        with patch('ai_extract.services.layout_service.LayoutExtractionService') as extraction:
            with self.assertRaises(Http404):
                api_run_layout_extraction(request,self.foreign.pk)
        extraction.assert_not_called()

    def test_semantic_concern_is_visible_without_hiding_generated_answer(self):
        self.persist_source(text='Article 1. Payment\nThe contract price is 120000000 VND.')
        def model(task, payload):
            if task == 'verify_answer':
                return {'supported': False, 'reason': 'Price does not establish a penalty.'}
            return dict(answer='The late payment penalty is 120000000 VND [E1].',answer_notes=[dict(
                text='Price does not establish a penalty.',evidence_ids=['E1'])])
        with patch('ai_extract.services.contract_pipeline_service.request_qwen', side_effect=model):
            result=self.post(query='payment penalty',generate=True).json()
        self.assertEqual(result['generation_status'],'generated')
        self.assertIn('The late payment penalty is',result['answer'])
        self.assertNotIn('semantic_check',result['citation_check'])
        self.assertIn('Price does not establish a penalty',
                      result['generation_diagnostics']['review_warning']['reason'])

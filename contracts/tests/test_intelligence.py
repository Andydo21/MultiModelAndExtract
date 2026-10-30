import json
from unittest.mock import patch
from django.test import TestCase, Client
from django.contrib.auth import get_user_model
from contracts.models import Company, Role, Contract, ContractVersion, ContractContext, ContractKnowledgeCase, AuditLog
from ai_extract.models import Clause, LayoutExtractionRun, LayoutExtractionPage, LayoutExtractionBlock


class ContractIntelligenceTests(TestCase):
    def setUp(self):
        self.company = Company.objects.create(company_name='Company A', tax_code='A')
        self.other = Company.objects.create(company_name='Company B', tax_code='B')
        self.manager_role = Role.objects.create(role_name='MANAGER')
        self.user = get_user_model().objects.create_user(username='author', company=self.company)
        self.manager = get_user_model().objects.create_user(username='manager', company=self.company, role=self.manager_role)
        self.outsider = get_user_model().objects.create_user(username='outsider', company=self.other, role=self.manager_role)
        self.contract = Contract.objects.create(company=self.company, contract_code='HD-001', title='Supply contract', contract_type='Supply')
        self.version = ContractVersion.objects.create(contract=self.contract, version_number=1)
        self.new_version = ContractVersion.objects.create(contract=self.contract, version_number=2)
        self.foreign = Contract.objects.create(company=self.other, contract_code='HD-002', title='Foreign contract')
        self.foreign_version = ContractVersion.objects.create(contract=self.foreign)
        clause = Clause.objects.create(version=self.version, clause_title='Payment', clause_content='Payment within 30 days.')
        Clause.objects.create(version=self.new_version, clause_title='Payment', clause_content='Payment within 60 days.')
        self.fields = dict(title='Ambiguous payment deadline', clause_type='Payment', issue='payment deadline unclear',
            reasoning='Clause omitted trigger date', resolution='Specify trigger and deadline',
            verification='Reviewer compared revised clause', applicability='Supply contracts with same trigger', source_reference=f'clause:{clause.pk}')
        self.client.force_login(self.user)

    def write_case(self, **changes):
        return self.client.post('/api/contract-knowledge/',
            json.dumps(dict(self.fields, version_id=self.version.pk, **changes)), content_type='application/json')

    def action(self, action, pk, **values):
        return self.client.post('/api/contract-knowledge/', json.dumps(dict(action=action, id=pk, **values)), content_type='application/json')

    def search(self, contract=None, **values):
        return self.client.post(f'/api/contracts/{(contract or self.contract).pk}/intelligence/search/',
            json.dumps({'query': 'payment', **values}), content_type='application/json')

    def test_search_defaults_to_latest_version(self):
        body = self.search().json()
        self.assertEqual(body['version_id'], self.new_version.pk)
        self.assertIn('60 days', body['citations'][0]['text'])
        self.assertNotIn('30 days', body['answer'])

    def test_explicit_version_preserved_in_sources(self):
        body = self.search(version_id=self.version.pk).json()
        self.assertIn('30 days', body['citations'][0]['text'])
        self.assertIn(f'version_id={self.version.pk}', body['citations'][0]['source_url'])

    def test_cross_company_contract_and_version_are_denied(self):
        self.assertEqual(self.search(contract=self.foreign).status_code, 404)
        self.assertEqual(self.search(version_id=self.foreign_version.pk).status_code, 404)
        self.assertEqual(self.client.get(f'/contracts/{self.foreign.pk}/intelligence/').status_code, 404)

    def test_anonymous_cannot_search_or_write(self):
        self.client.logout()
        self.assertEqual(self.search().status_code, 401)
        self.assertEqual(self.write_case().status_code, 401)

    def test_companyless_user_has_no_default_access(self):
        self.user.company = None
        self.user.save()
        self.assertEqual(self.search().status_code, 404)

    @patch('ai_extract.services.contract_pipeline_service.HybridBackend.rerank_many',
           side_effect=lambda groups: [[dict(c, neural_logit=2.0) for c in candidates] for _, candidates in groups])
    def test_case_review_uses_existing_manager_role(self, reranker):
        response = self.write_case()
        self.assertEqual(response.status_code, 200)
        pk = response.json()['case']['id']
        self.assertEqual(self.action('approve', pk).status_code, 403)
        self.assertEqual(self.client.get('/api/contract-knowledge/?q=payment').json()['cases'], [])
        self.assertEqual(len(self.client.get('/api/contract-knowledge/').json()['cases']), 1)
        self.client.force_login(self.manager)
        self.assertEqual(self.action('approve', pk).status_code, 200)
        self.client.force_login(self.user)
        self.assertEqual(len(self.search().json()['similar_cases']), 1)
        self.assertEqual(AuditLog.objects.filter(action='A2_CASE_APPROVE').count(), 1)

    def test_case_requires_verification_and_source(self):
        fields = dict(self.fields, verification='')
        response = self.client.post('/api/contract-knowledge/', json.dumps(dict(fields, version_id=self.version.pk)), content_type='application/json')
        pk = response.json()['case']['id']
        self.client.force_login(self.manager)
        self.assertEqual(self.action('approve', pk).status_code, 400)

    def test_revision_keeps_approved_history(self):
        pk = self.write_case().json()['case']['id']
        self.client.force_login(self.manager)
        self.action('approve', pk)
        revised = self.action('revise', pk, resolution='New wording').json()['case']
        self.assertEqual(revised['revision'], 2)
        self.assertEqual(ContractKnowledgeCase.objects.get(pk=pk).status, 'approved')
        second_reviewer = get_user_model().objects.create_user(username='second-reviewer', company=self.company, role=self.manager_role)
        self.client.force_login(second_reviewer)
        self.action('approve', revised['id'])
        self.assertEqual(ContractKnowledgeCase.objects.get(pk=pk).status, 'deprecated')
        self.assertTrue(AuditLog.objects.filter(action='A2_CASE_SUPERSEDED').exists())

    def test_other_company_cannot_read_or_modify_case(self):
        pk = self.write_case().json()['case']['id']
        self.client.force_login(self.outsider)
        self.assertEqual(self.client.get('/api/contract-knowledge/').json()['cases'], [])
        self.assertEqual(self.action('approve', pk).status_code, 404)

    def test_raw_context_fallback_excludes_generated_advice(self):
        Clause.objects.filter(version=self.new_version).delete()
        ContractContext.objects.create(version=self.new_version, context_type='advice', source='AI', content='Payment 999 days')
        ContractContext.objects.create(version=self.new_version, context_type='raw_text', source='Original', content='Payment 60 days')
        body = self.search().json()
        self.assertEqual(len(body['citations']), 1)
        self.assertNotIn('999', body['answer'])

    def test_unknown_citation_falls_back_to_source(self):
        with patch('ai_extract.services.contract_pipeline_service.request_qwen', return_value={'answer':'Incorrect [E999]'}):
            body = self.search(generate=True).json()
        self.assertEqual(body['generation_status'], 'source_excerpt')
        self.assertNotIn('Incorrect', body['answer'])

    def test_valid_generation_and_timing(self):
        with patch('ai_extract.services.contract_pipeline_service.request_qwen', side_effect=lambda task,payload:
                   {'supported':True,'reason':'Payment term is stated in fixture evidence'} if task=='verify_answer'
                   else {'answer':'Payment 60 days [E1]'}):
            body = self.search(generate=True).json()
        self.assertEqual(body['generation_status'], 'generated')
        self.assertGreaterEqual(body['latency_ms'], body['retrieval_latency_ms'])

    def test_page_block_links_keep_version(self):
        run = LayoutExtractionRun.objects.create(version=self.version, status='COMPLETED')
        page = LayoutExtractionPage.objects.create(run=run, page_number=2)
        LayoutExtractionBlock.objects.create(page=page, block_order=1, text='Payment 30 days', bbox=[0,0,100,100])
        body = self.search(version_id=self.version.pk).json()
        block = next(c for c in body['citations'] if c['chunk_id'].startswith('block:'))
        self.assertIn(f'version_id={self.version.pk}', block['source_url'])
        self.assertEqual(block['page_number'], 2)

    def test_structural_search_expands_cross_page_reference_with_same_version(self):
        run = LayoutExtractionRun.objects.create(version=self.new_version, status='COMPLETED')
        for number, content in [(1, 'Điều 2. Chấm dứt\n1. Được chấm dứt theo Điều 3.'),
                                (2, 'Điều 3. Thông báo\nPhải báo trước 30 ngày.')]:
            page = LayoutExtractionPage.objects.create(run=run, page_number=number)
            LayoutExtractionBlock.objects.create(page=page, block_order=1, text=content,layout_type='title',structure_metadata=dict(heading_level=1,structure_kind='article'))
        from ai_extract.services.document_structure_service import analyze_document_structure
        from NewIdea.repositories.contract_repository import version_sources
        run.structure=analyze_document_structure(version_sources(self.new_version),'Contract')
        a,b=run.structure['leaves'][:2]
        a['graph_related_chunk_ids']=[b['chunk_id']]
        run.save(update_fields=['structure'])
        body = self.search(query='chấm dứt').json()
        self.assertIn('30 ngày', '\n'.join(c['text'] for c in body['citations']))
        self.assertTrue(all(c['version_id']==self.new_version.pk for c in body['citations']))
        self.assertTrue(any(c['heading_path'] for c in body['citations']))
        self.assertEqual(body['context_coverage']['mode'], 'structural')

    def test_whole_review_does_not_require_a_lexical_match(self):
        with patch('ai_extract.services.contract_pipeline_service.request_qwen', return_value={'answer':'Payment terms [E1]'}):
            body = self.search(query='Rà soát toàn bộ', generate=True, mode='whole_contract').json()
        self.assertTrue(body['context_coverage']['whole_contract_review'])
        self.assertEqual(body['generation_status'], 'reviewed_all_extracted_sources')

    def test_structure_persisted_with_a3_metadata(self):
        from types import SimpleNamespace
        from ai_extract.services.contract_pipeline_service import IndexResult
        from ai_extract.services.layout_service import PageExtractionSession
        run = LayoutExtractionRun.objects.create(version=self.new_version)
        block = SimpleNamespace(order=1, layout_type='title',
            text='Article 1. Payment\n1. Pay within 60 days.', bbox=[0,0,1000,1000],
            confidence=0.9, structure_metadata={'parent_chunk_id':'parent_1', 'heading_path':'Payment','heading_level':1,'structure_kind':'clause'})
        result = SimpleNamespace(pages=[SimpleNamespace(page_number=1, width=100, height=100,
            image_path='', text=block.text, blocks=[block])])
        with patch('ai_extract.services.contract_pipeline_service.request_qwen') as qwen, \
             patch.object(PageExtractionSession,'_index_visual',return_value=IndexResult('COMPLETED')), \
             patch('ai_extract.services.contract_pipeline_service.PlanACollectionIndexer') as indexer:
            indexer.return_value.index_dense.return_value=IndexResult('COMPLETED')
            indexer.return_value.prune_old_run_points.return_value=dict(dense='skipped',colpali='skipped')
            with PageExtractionSession(run) as session:
                session.process_page(result.pages[0])
                session.finish()
        qwen.assert_not_called()
        run.refresh_from_db()
        self.assertEqual(run.structure['schema'], 'document-structure-v2')
        self.assertEqual(run.structure['structure_status'], 'structured')
        self.assertTrue(any(n['kind']=='clause' for n in run.structure['nodes']))
        saved = run.pages.first().blocks.first()
        self.assertEqual(saved.structure_metadata['parent_chunk_id'], 'parent_1')
        from NewIdea.repositories.contract_repository import version_sources
        from ai_extract.services.document_structure_service import structure_for_version
        reused = structure_for_version(self.new_version, version_sources(self.new_version), self.contract.title)
        self.assertEqual(reused['schema'], 'document-structure-v2')
        self.assertEqual(reused['leaves'][0]['contract_id'], self.contract.pk)
        self.assertIn('source_url', reused['leaves'][0])
        with patch('ai_extract.services.contract_pipeline_service.request_qwen') as model:
            response = self.search(version_id=self.new_version.pk).json()
        model.assert_not_called()
        self.assertEqual(response['context_coverage']['structure_status'], 'structured')
        self.assertTrue(any(n['kind'] == 'clause' for n in response['structure_outline']))
        self.assertEqual(response['structure_visualization']['status'], 'structured')
        self.assertEqual(response['retrieval_trace']['generation_calls'], [])
        self.assertTrue(response['retrieval_trace']['selected'])

    def test_changed_source_does_not_reuse_model_structure(self):
        from ai_extract.services.document_structure_service import analyze_document_structure, structure_for_version
        from NewIdea.repositories.contract_repository import version_sources
        run = LayoutExtractionRun.objects.create(version=self.new_version, status='COMPLETED')
        page = LayoutExtractionPage.objects.create(run=run, page_number=1)
        block = LayoutExtractionBlock.objects.create(page=page, block_order=1, text='Original source')
        def model(task, payload):
            return {'relations': []} if task == 'document_relations' else {'nodes': [dict(id='u1', parent_id='root', kind='paragraph', is_heading=False, title='')]}
        run.structure = analyze_document_structure(version_sources(self.new_version), 'Manual')
        run.save(update_fields=['structure'])
        block.text = 'Changed source'
        block.save(update_fields=['text'])
        tree = structure_for_version(self.new_version, version_sources(self.new_version), 'Manual')
        self.assertEqual(tree['structure_status'], 'source_changed_requires_reanalysis')
        self.assertEqual(tree['leaves'][0]['text'], 'Changed source')

    def test_clause_rows_use_a3_hierarchy_and_retain_descendants(self):
        from ai_extract.services import ClauseExtractService
        run = LayoutExtractionRun.objects.create(version=self.new_version, status='COMPLETED')
        page = LayoutExtractionPage.objects.create(run=run, page_number=1)
        for order,(text,kind,level) in enumerate([
            ('Article 1. Payment','article',1),('1. Pay within 60 days.','clause',2),
            ('a) Payment must be in USD.','point',3),('Article 2. Term','article',1),
            ('One year.',None,None)],1):
            LayoutExtractionBlock.objects.create(page=page,block_order=order,text=text,
                layout_type='title' if kind else 'paragraph',
                structure_metadata=dict(heading_level=level,structure_kind=kind) if kind else {})
        with patch('ai_extract.services.contract_pipeline_service.request_qwen') as qwen:
            result = ClauseExtractService().extract_version(self.new_version, re_extract=True)
            qwen.assert_not_called()
        self.assertEqual(result['total_clauses'], 2)
        payment = self.new_version.ai_extract_clauses.get(clause_title='Article 1. Payment')
        self.assertIn('60 days', payment.clause_content)
        self.assertIn('USD', payment.clause_content)
        second = ClauseExtractService().extract_version(self.new_version)
        self.assertEqual(second['total_clauses'], 2)
        self.assertEqual(payment.pk, self.new_version.ai_extract_clauses.get(clause_title='Article 1. Payment').pk)

    def test_expert_ai_draft_requires_confirmation_and_does_not_save(self):
        count=ContractKnowledgeCase.objects.count()
        with patch('ai_extract.services.contract_pipeline_service.request_qwen',return_value={'title':'Payment review','issue':'Missing deadline'}):
            response=self.client.post('/api/contract-knowledge/draft/',json.dumps(
                {'version_id':self.new_version.pk,'note':'Expert reviewed missing payment deadline in Article 1.'}),content_type='application/json')
        self.assertEqual(response.status_code,200)
        self.assertTrue(response.json()['requires_expert_confirmation'])
        self.assertFalse(response.json()['saved'])
        self.assertEqual(response.json()['draft']['verification'],'')
        self.assertEqual(ContractKnowledgeCase.objects.count(),count)

    def test_expert_ai_draft_denies_foreign_version_before_model_call(self):
        with patch('ai_extract.services.contract_pipeline_service.request_qwen') as model:
            response=self.client.post('/api/contract-knowledge/draft/',json.dumps(
                {'version_id':self.foreign.latest_version.pk,'note':'Expert reviewed the foreign contract payment deadline.'}),content_type='application/json')
        self.assertEqual(response.status_code,404)
        model.assert_not_called()

    def test_generated_unsupported_number_is_displayed_with_model_warning(self):
        with patch('ai_extract.services.contract_pipeline_service.request_qwen',return_value=dict(
                answer='Payment within 999 days [E1]',answer_notes=[dict(
                    text='The source says 30 days, not 999.',evidence_ids=['E1'])])):
            body=self.search(generate=True).json()
        self.assertEqual(body['generation_status'],'generated')
        self.assertIn('999',body['answer'])
        self.assertIn('30 days, not 999',body['generation_diagnostics']['review_warning']['reason'])
        self.assertNotIn('semantic_check',body['citation_check'])

    def test_workspace_renders_contract_terms(self):
        response = self.client.get(f'/contracts/{self.contract.pk}/intelligence/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'knowledge-form')
        self.assertNotContains(response, 'Máy / revision')
        invalid = self.client.get(f'/contracts/{self.contract.pk}/intelligence/?version_id=invalid')
        self.assertEqual(invalid.status_code, 400)

    def test_write_requires_csrf(self):
        client = Client(enforce_csrf_checks=True)
        client.force_login(self.user)
        self.assertEqual(client.post('/api/contract-knowledge/', '{}', content_type='application/json').status_code, 403)

    def test_evidence_window_retains_tail_and_verbatim_text(self):
        from shared.retrieval import evidence_window
        text = 'Intro. ' * 1000 + 'Payment within 60 days.'
        window = evidence_window('payment', text, 200)
        self.assertLessEqual(len(window), 200)
        self.assertIn('Payment within 60 days.', window)
        self.assertIn(window, text)

    def test_other_contract_type_does_not_become_a_similar_case(self):
        pk = self.write_case().json()['case']['id']
        self.client.force_login(self.manager)
        self.action('approve', pk)
        self.contract.contract_type = 'Lease'
        self.contract.save()
        different = Contract.objects.create(company=self.company, contract_code='HD-003', title='Different', contract_type='Supply')
        ContractVersion.objects.create(contract=different)
        self.assertEqual(self.search(contract=different).json()['similar_cases'], [])

    def test_qwen_prompt_uses_contract_evidence(self):
        from ai_extract.services.contract_pipeline_service import _request_openai_compatible
        with patch('ai_extract.services.contract_pipeline_service._chat', return_value='Payment [E1]') as chat:
            result = _request_openai_compatible('http://unused', 'contract_qa',
                {'query':'payment', 'evidence':[{'evidence_id':'E1', 'text':'Payment 60 days'}]})
        self.assertEqual(result['answer'], 'Payment [E1]')
        messages = chat.call_args.args[1]
        self.assertIn('trợ lý tra cứu tài liệu', messages[0]['content'])
        self.assertIn('[E1]', messages[1]['content'])
        self.assertNotIn('cơ khí', messages[0]['content'])

    def test_newidea_preserves_a3_image_only_block(self):
        from NewIdea.repositories.contract_repository import version_sources
        from ai_extract.services.contract_pipeline_service import build_structure
        run = LayoutExtractionRun.objects.create(version=self.version, status='COMPLETED')
        page = LayoutExtractionPage.objects.create(run=run, page_number=1)
        block = LayoutExtractionBlock.objects.create(page=page, block_order=1, layout_type='figure', text='',
            structure_metadata={'image_path': 'extracted_images/circuit.png', 'source_location': 'Slide 1'})
        sources = version_sources(self.version)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]['source_location'], 'Slide 1')
        tree = build_structure(sources)
        self.assertEqual(tree['leaves'][0]['image_path'], 'extracted_images/circuit.png')
        self.assertEqual(tree['leaves'][0]['source_chunk_id'], f'block:{block.pk}')

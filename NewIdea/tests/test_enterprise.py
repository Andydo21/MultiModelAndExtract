import json
from types import SimpleNamespace
from unittest.mock import patch
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings, Client
from contracts.tests.test_intelligence import ContractIntelligenceTests
from contracts.models import Contract, ContractVersion, ContractKnowledgeCase, ContractKnowledgeInquiry, KnowledgeSearchLog
from ai_extract.models import Clause, LayoutExtractionRun, LayoutExtractionPage, LayoutExtractionBlock


class EnterpriseKnowledgeTests(TestCase):
    setUp = ContractIntelligenceTests.setUp
    write_case = ContractIntelligenceTests.write_case
    action = ContractIntelligenceTests.action
    search = ContractIntelligenceTests.search

    def inquiry(self, **data):
        return self.client.post('/api/newidea/inquiries/', json.dumps(data), content_type='application/json')

    def discovery(self, **data):
        return self.client.post('/api/newidea/search/', json.dumps({'query': 'payment', **data}), content_type='application/json')

    def test_manager_cannot_approve_own_revision(self):
        self.client.force_login(self.manager)
        pk = self.write_case().json()['case']['id']
        self.assertEqual(self.action('approve', pk).status_code, 403)
        self.assertEqual(ContractKnowledgeCase.objects.get(pk=pk).status, 'draft')

    def test_source_must_exist_in_case_version(self):
        for source in ('invented review', 'clause:999999'):
            fields = dict(self.fields, source_reference=source)
            self.client.force_login(self.user)
            pk = self.client.post('/api/contract-knowledge/', json.dumps(dict(fields, version_id=self.version.pk)), content_type='application/json').json()['case']['id']
            self.client.force_login(self.manager)
            self.assertEqual(self.action('approve', pk).status_code, 400)

    def test_foreign_version_evidence_cannot_be_bound(self):
        clause = Clause.objects.create(version=self.foreign_version, clause_title='Secret', clause_content='Payment 999 days')
        fields = dict(self.fields, source_reference=f'clause:{clause.pk}')
        pk = self.client.post('/api/contract-knowledge/', json.dumps(dict(fields, version_id=self.version.pk)), content_type='application/json').json()['case']['id']
        self.client.force_login(self.manager)
        self.assertEqual(self.action('approve', pk).status_code, 400)

    def test_changed_evidence_excludes_previously_approved_case(self):
        pk = self.write_case().json()['case']['id']
        self.client.force_login(self.manager)
        self.assertEqual(self.action('approve', pk).status_code, 200)
        case = ContractKnowledgeCase.objects.get(pk=pk)
        self.assertEqual(len(case.evidence_snapshot['sha256']), 64)
        Clause.objects.filter(version=self.version).update(clause_content='Payment terms replaced')
        self.assertEqual(self.search().json()['similar_cases'], [])
        self.assertEqual(self.client.get('/api/contract-knowledge/?q=payment').json()['cases'], [])

    def test_search_receipt_contains_answer_and_source_hash(self):
        body = self.search().json()
        log = KnowledgeSearchLog.objects.get(pk=body['search_id'])
        self.assertEqual(log.status, 'COMPLETED')
        self.assertEqual(log.version, self.new_version)
        self.assertEqual(log.result['answer'], body['answer'])
        self.assertEqual(len(log.result['sources'][0]['sha256']), 64)

    def test_quota_checked_before_model_or_retrieval(self):
        KnowledgeSearchLog.objects.bulk_create([KnowledgeSearchLog(user=self.user, query='test') for _ in range(30)])
        with patch('NewIdea.services.contract_search_service.retrieve_hybrid') as retrieval:
            response = self.search()
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response['Retry-After'], '60')
        retrieval.assert_not_called()

    def test_long_question_rejected_before_retrieval(self):
        with patch('NewIdea.services.contract_search_service.retrieve_hybrid') as retrieval:
            self.assertEqual(self.search(query='x'*2001).status_code, 400)
        retrieval.assert_not_called()

    @patch('ai_extract.services.contract_pipeline_service.HybridBackend.rerank_many',
           side_effect=lambda groups: [[dict(c, neural_logit=2.0) for c in candidates] for _, candidates in groups])
    def test_expert_workflow_requires_independent_approval(self, reranker):
        receipt = self.search(version_id=self.version.pk).json()['search_id']
        item = self.inquiry(search_id=receipt).json()['inquiry']
        self.assertEqual(self.inquiry(search_id=receipt).json()['inquiry']['id'], item['id'])
        self.assertEqual(ContractKnowledgeInquiry.objects.count(), 1)
        self.assertEqual(self.inquiry(action='assign', id=item['id'], assignee_id=self.manager.pk).status_code, 403)
        self.client.force_login(self.manager)
        self.assertEqual(self.inquiry(action='assign', id=item['id'], assignee_id=self.manager.pk).status_code, 200)
        answered = self.inquiry(action='answer', id=item['id'], **self.fields)
        self.assertEqual(answered.status_code, 200)
        self.assertEqual(answered.json()['inquiry']['status'], 'ANSWERED')
        pk = answered.json()['case']['id']
        self.assertEqual(self.action('approve', pk).status_code, 403)
        reviewer = get_user_model().objects.create_user(username='independent', company=self.company, role=self.manager_role)
        self.client.force_login(reviewer)
        self.assertEqual(self.action('approve', pk).status_code, 200)
        self.assertEqual(ContractKnowledgeInquiry.objects.get(pk=item['id']).status, 'RESOLVED')
        self.client.force_login(self.user)
        self.assertEqual(self.search().json()['similar_cases'][0]['id'], pk)

    def test_receipt_and_inquiry_not_visible_to_foreign_company(self):
        receipt = self.search().json()['search_id']
        item = self.inquiry(search_id=receipt).json()['inquiry']
        self.client.force_login(self.outsider)
        self.assertEqual(self.inquiry(search_id=receipt).status_code, 404)
        self.assertEqual(self.inquiry(action='assign', id=item['id'], assignee_id=self.outsider.pk).status_code, 404)
        self.assertEqual(self.client.get('/api/newidea/inquiries/').json()['inquiries'], [])
        self.assertEqual(self.client.get(f"/api/newidea/inquiries/?id={item['id']}").status_code, 404)

    def test_inquiry_detail_restores_draft_and_original_evidence(self):
        receipt = self.search(version_id=self.version.pk).json()['search_id']
        item = self.inquiry(search_id=receipt).json()['inquiry']
        self.client.force_login(self.manager)
        self.inquiry(action='assign', id=item['id'], assignee_id=self.manager.pk)
        answered = self.inquiry(action='answer', id=item['id'], **self.fields).json()
        body = self.client.get(f"/api/newidea/inquiries/?id={item['id']}").json()
        self.assertEqual(body['case']['id'], answered['case']['id'])
        self.assertEqual(body['case']['resolution'], self.fields['resolution'])
        self.assertEqual(body['search']['sources'], KnowledgeSearchLog.objects.get(pk=receipt).result['sources'])
        self.client.force_login(self.user)
        self.assertEqual(self.client.get(f"/api/newidea/inquiries/?id={item['id']}").status_code, 200)
        colleague = get_user_model().objects.create_user(username='uninvolved', company=self.company)
        self.client.force_login(colleague)
        self.assertEqual(self.client.get(f"/api/newidea/inquiries/?id={item['id']}").status_code, 404)

    def test_inactive_reviewer_cannot_be_assigned(self):
        receipt = self.search().json()['search_id']
        item = self.inquiry(search_id=receipt).json()['inquiry']
        self.client.force_login(self.manager)
        inactive = get_user_model().objects.create_user(username='inactive', company=self.company, role=self.manager_role, is_active=False)
        self.assertEqual(self.inquiry(action='assign', id=item['id'], assignee_id=inactive.pk).status_code, 404)
        reviewers = self.client.get('/api/newidea/inquiries/').json()['reviewers']
        self.assertNotIn(inactive.pk, [r['id'] for r in reviewers])

    def test_company_discovery_finds_other_contract_and_only_latest_version(self):
        contract = Contract.objects.create(company=self.company, contract_code='SECOND', title='Another agreement')
        old = ContractVersion.objects.create(contract=contract, version_number=1)
        new = ContractVersion.objects.create(contract=contract, version_number=2)
        Clause.objects.create(version=old, clause_title='Payment', clause_content='Payment 123 days')
        Clause.objects.create(version=new, clause_title='Payment', clause_content='Payment 77 days')
        Clause.objects.create(version=self.foreign_version, clause_title='Payment', clause_content='Payment SECRET999')
        body = self.discovery().json()
        text = ' '.join(r['text'] for r in body['results'])
        self.assertIn('77 days', text)
        self.assertIn('60 days', text)
        self.assertNotIn('123 days', text)
        self.assertNotIn('30 days', text)
        self.assertNotIn('SECRET999', text)

    @override_settings(A3_HYBRID_ENABLED=True)
    def test_dense_discovery_ignores_forged_payloads_and_finds_synonym(self):
        run = LayoutExtractionRun.objects.create(version=self.new_version, status='COMPLETED', dense_index_status='COMPLETED', colpali_index_status='COMPLETED')
        page = LayoutExtractionPage.objects.create(run=run, page_number=1)
        block = LayoutExtractionBlock.objects.create(page=page, block_order=1, text='Article 1. Payment\nPay within 60 days.')
        payload = dict(source_app='contract_dl', company_id=self.company.pk, contract_id=self.contract.pk,
            version_id=self.new_version.pk, contract_dl_run_id=run.pk, chunk_id=block.pk, page_number=1)
        point = SimpleNamespace(payload=payload, score=.9)
        foreign = SimpleNamespace(payload=dict(payload, company_id=self.other.pk), score=1)
        with patch('NewIdea.services.knowledge_search_service.SearchBackend.dense_knowledge', return_value=[foreign, point]) as dense, \
             patch('NewIdea.services.knowledge_search_service.SearchBackend.dense', return_value=[point]), \
             patch('NewIdea.services.knowledge_search_service.SearchBackend.colpali', return_value=[]), \
             patch('NewIdea.services.knowledge_search_service.SearchBackend.rerank', side_effect=lambda q, c: c):
            body = self.discovery(query='settle invoices').json()
        self.assertEqual({v.pk for v in dense.call_args.args[1]}, {self.new_version.pk})
        self.assertEqual(dense.call_args.args[2], {self.new_version.pk: run.pk})
        self.assertTrue(body['results'])
        self.assertTrue(all(r['version_id'] == self.new_version.pk for r in body['results']))

    def test_global_search_and_inquiries_require_csrf_and_auth(self):
        client = Client(enforce_csrf_checks=True)
        client.force_login(self.user)
        for url in ('/api/newidea/search/', '/api/newidea/inquiries/'):
            self.assertEqual(client.post(url, '{}', content_type='application/json').status_code, 403)
        self.client.logout()
        self.assertEqual(self.discovery().status_code, 401)
        self.assertEqual(self.inquiry(search_id=1).status_code, 401)

    def test_evidence_picker_is_scoped(self):
        self.assertEqual(self.client.get(f'/api/newidea/versions/{self.foreign_version.pk}/sources/').status_code, 404)
        body = self.client.get(f'/api/newidea/versions/{self.version.pk}/sources/').json()
        self.assertEqual(body['sources'][0]['chunk_id'], self.fields['source_reference'])

    def test_queued_search_runs_once_and_preserves_receipt(self):
        from contracts.management.commands.knowledge_worker import process_one
        with patch('ai_extract.services.contract_pipeline_service.request_qwen', side_effect=lambda task, payload:
                   {'supported': True, 'reason': 'Fixture evidence supports payment term'} if task == 'verify_answer'
                   else {'answer': 'Payment within 60 days [E1]'}) as model:
            response = self.client.post('/api/newidea/jobs/', json.dumps(dict(query='payment', version_id=self.new_version.pk)), content_type='application/json')
            self.assertEqual(response.status_code, 202)
            model.assert_not_called()
            pk = response.json()['search_id']
            self.assertEqual(KnowledgeSearchLog.objects.get(pk=pk).status, 'QUEUED')
            self.assertTrue(process_one())
            self.assertFalse(process_one())
        self.assertEqual(KnowledgeSearchLog.objects.count(), 1)
        body = self.client.get(f'/api/newidea/jobs/?id={pk}').json()
        self.assertEqual(body['status'], 'COMPLETED')
        self.assertEqual(body['result']['response']['generation_status'], 'generated')
        self.assertIn('sha256', body['result']['sources'][0])
        self.client.force_login(self.outsider)
        self.assertEqual(self.client.get(f'/api/newidea/jobs/?id={pk}').status_code, 404)

    def test_failed_worker_does_not_leave_running_job(self):
        from contracts.management.commands.knowledge_worker import process_one
        response = self.client.post('/api/newidea/jobs/', json.dumps(dict(query='payment', version_id=self.new_version.pk)), content_type='application/json')
        with patch('contracts.management.commands.knowledge_worker.search_contract', side_effect=RuntimeError('down')):
            self.assertTrue(process_one())
        log = KnowledgeSearchLog.objects.get(pk=response.json()['search_id'])
        self.assertEqual(log.status, 'FAILED')
        self.assertNotIn('down', log.result['error'])

    def test_worker_rechecks_access_after_company_changes(self):
        from contracts.management.commands.knowledge_worker import process_one
        queued = self.client.post('/api/newidea/jobs/', json.dumps(dict(query='payment', version_id=self.new_version.pk)), content_type='application/json')
        self.user.company = self.other
        self.user.save(update_fields=['company'])
        with patch('ai_extract.services.contract_pipeline_service.request_qwen') as model:
            self.assertTrue(process_one())
        model.assert_not_called()
        self.assertEqual(KnowledgeSearchLog.objects.get(pk=queued.json()['search_id']).status, 'FAILED')

    def test_pending_job_limit_prevents_unbounded_queue(self):
        for _ in range(3):
            response = self.client.post('/api/newidea/jobs/', json.dumps(dict(query='payment', version_id=self.new_version.pk)), content_type='application/json')
            self.assertEqual(response.status_code, 202)
        response = self.client.post('/api/newidea/jobs/', json.dumps(dict(query='payment', version_id=self.new_version.pk)), content_type='application/json')
        self.assertEqual(response.status_code, 429)
        self.assertEqual(KnowledgeSearchLog.objects.filter(status='QUEUED').count(), 3)

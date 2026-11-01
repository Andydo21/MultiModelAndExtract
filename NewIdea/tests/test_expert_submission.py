import json
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch
from django.test import SimpleTestCase, TestCase, override_settings
from django.core.files.uploadedfile import SimpleUploadedFile
from contracts.tests.test_intelligence import ContractIntelligenceTests
from contracts.models import KnowledgeSearchLog, ContractFile
from ai_extract.models import LayoutExtractionRun
from NewIdea.services.expert_submission_service import validate_semantics, grounded_concepts
from NewIdea.services.search_job_service import process_next_job
from contracts.services.document_encryption_service import decrypt_pdf


class SemanticGroundingTests(SimpleTestCase):
    def test_reset_graph_does_not_recreate_context_from_saved_relations(self):
        from types import SimpleNamespace
        from NewIdea.services.search_evidence_service import submission_context
        receipt=SimpleNamespace(result={'graph_index': {'status': 'cleared'}})
        context=submission_context(receipt, [])
        self.assertEqual(context['state'], 'cleared')
        self.assertEqual(context['nodes'], [])
        self.assertEqual(context['edges'], [])

    def test_concept_name_is_copied_from_source_not_generated_translation(self):
        concept=grounded_concepts([dict(kind='action',quote='kiểm tra dây tín hiệu')],
                                 'Hãy kiểm tra dây tín hiệu.')[0]
        self.assertEqual(concept['value'],'kiểm tra dây tín hiệu')
    def test_shared_concept_identity_is_literal_and_kind_specific(self):
        first=grounded_concepts([dict(kind='machine',value='AX-203',quote='AX-203')],'Repair AX-203')[0]
        same=grounded_concepts([dict(kind='machine',value='ax-203',quote='ax-203')],'Repair ax-203')[0]
        different=grounded_concepts([dict(kind='machine',value='AX-204',quote='AX-204')],'Repair AX-204')[0]
        self.assertEqual(first['id'],same['id']);self.assertNotEqual(first['id'],different['id'])
        self.assertEqual(first['relation'],'ABOUT_MACHINE')

    def test_invented_machine_solution_and_relation_are_rejected(self):
        for concept in [dict(kind='machine',value='AX-999',quote='AX-999'),
                        dict(kind='solution',value='Replace sensor',quote='AX-203'),
                        dict(kind='ARBITRARY_CYPHER',value='AX-203',quote='AX-203')]:
            with self.assertRaises(ValueError):grounded_concepts([concept],'AX-203')
    def response(self):
        return dict(units=[dict(id='u1',source_id='note',quote='Pay within 30 days.',title='Payment',kind='clause')],
                    relations=[dict(from_id='question',to_id='u1',kind='ASKS_ABOUT')])

    def test_exact_quote_and_root_relation_are_preserved(self):
        units, relations = validate_semantics(self.response(), [dict(id='note',text='Pay within 30 days.')], '0')
        self.assertEqual(units[0]['quote'], 'Pay within 30 days.')
        self.assertEqual(relations[0]['to_id'], units[0]['id'])
        self.assertEqual(relations[0]['from_id'], 'question')

    def test_hallucinated_text_and_unknown_sources_or_relation_endpoints_are_rejected(self):
        for change in ('quote','source','endpoint','kind'):
            response = self.response()
            if change=='quote': response['units'][0]['quote']='Pay within 999 days.'
            if change=='source': response['units'][0]['source_id']='foreign'
            if change=='endpoint': response['relations'][0]['to_id']='invented'
            if change=='kind': response['relations'][0]['kind']='ARBITRARY_CYPHER'
            with self.assertRaises(ValueError):
                validate_semantics(response, [dict(id='note',text='Pay within 30 days.')], '0')


@override_settings(A3_HYBRID_ENABLED=False)
class ExpertSubmissionFlow(TestCase):
    setUp = ContractIntelligenceTests.setUp

    def submit(self, **data):
        return self.client.post('/api/newidea/expert-submissions/', json.dumps(dict(question='When is payment due?',
            note='Pay within 30 days.', **data)), content_type='application/json')

    def model(self, task, payload):
        self.assertEqual(task, 'expert_semantic_extract')
        source = next((s for s in payload['sources'] if 'Pay within 30 days.' in s['text']), None)
        if source is None:return dict(units=[])
        return dict(units=[dict(id='u1',source_id=source['id'],quote='Pay within 30 days.',title='Payment',kind='clause')],
                    relations=[dict(from_id='question',to_id='u1',kind='ASKS_ABOUT')])

    def test_note_only_job_is_persisted_readable_and_processed_without_a3(self):
        response = self.submit();self.assertEqual(response.status_code,202)
        identity = response.json()['search_id']
        self.assertEqual(self.client.get('/api/newidea/jobs/',{'id':identity}).status_code,200)
        with patch('ai_extract.services.contract_pipeline_service.request_qwen',side_effect=self.model), \
             patch('NewIdea.repositories.expert_submission_repository.sync_semantic_graph',return_value=dict(status='ready')) as graph, \
             patch('ai_extract.services.contract_pipeline_service.LayoutExtractionService.extract_version') as extract:
            process_next_job(lambda *a,**k:self.fail('Expert jobs must use expert worker'))
        extract.assert_not_called();graph.assert_called_once()
        log = KnowledgeSearchLog.objects.get(pk=identity)
        self.assertEqual(log.status,'COMPLETED')
        self.assertEqual(log.result['response']['units'][0]['quote'],'Pay within 30 days.')
        self.assertTrue(log.result['response']['requires_expert_review'])

    def test_model_semantic_links_and_source_membership_are_both_preserved(self):
        response=self.submit(answer='Replace the sensor on AX-203.')
        identity=response.json()['search_id']
        def model(task,payload):
            source=next((s for s in payload['sources'] if s['id'].startswith('expert-answer:')),None)
            if source is None:return dict(units=[])
            return dict(units=[dict(id='u1',source_id=source['id'],quote=source['text'],title='Sensor repair',
                kind='recommendation',concepts=[dict(kind='machine',value='AX-203',quote='AX-203'),
                dict(kind='action',value='Replace the sensor',quote='Replace the sensor')])],
                relations=[dict(from_id='u1',to_id='question',kind='ANSWERS')])
        with patch('ai_extract.services.contract_pipeline_service.request_qwen',side_effect=model), \
             patch('NewIdea.repositories.expert_submission_repository.sync_semantic_graph',return_value=dict(status='ready')), \
             patch('NewIdea.repositories.expert_submission_repository.related_units',return_value=[]):
            process_next_job(lambda *a,**k:self.fail('Wrong worker'))
        result=KnowledgeSearchLog.objects.get(pk=identity).result['response']
        self.assertIn(dict(from_id='answer',to_id='question',kind='ANSWERS'),result['relations'])
        self.assertIn(dict(from_id='answer',to_id=result['units'][0]['id'],kind='HAS_UNIT'),result['relations'])
        self.assertEqual(result['units'][0]['concepts'][0]['relation'],'ABOUT_MACHINE')
        self.assertIn(dict(from_id=result['units'][0]['id'],to_id='question',kind='ANSWERS'),result['relations'])
        self.assertNotIn('CONTRADICTS',[r['kind'] for r in result['relations']])

    def test_foreign_document_anonymous_and_company_change_are_blocked(self):
        self.assertEqual(self.submit(version_id=self.foreign_version.pk).status_code,404)
        identity = self.submit().json()['search_id']
        self.user.company=self.other;self.user.save(update_fields=['company'])
        self.assertEqual(self.client.get('/api/newidea/jobs/',{'id':identity}).status_code,404)
        with patch('ai_extract.services.contract_pipeline_service.request_qwen') as model:
            process_next_job(lambda *a,**k:self.fail('Unexpected version job'))
        model.assert_not_called()
        log=KnowledgeSearchLog.objects.get(pk=identity)
        self.assertEqual(log.status,'FAILED');self.assertIn('request',log.result)
        self.client.logout();self.assertEqual(self.submit().status_code,401)

    def test_attachment_is_encrypted_and_worker_uses_a3_and_keeps_document_link(self):
        with TemporaryDirectory() as folder, override_settings(MEDIA_ROOT=Path(folder)):
            upload=SimpleUploadedFile('evidence.txt',b'Pay within 30 days.',content_type='text/plain')
            response=self.client.post('/api/newidea/expert-submissions/',dict(question='When is payment due?',files=upload))
            self.assertEqual(response.status_code,202)
            identity=response.json()['search_id'];version_id=response.json()['attachment_version_ids'][0]
            file=ContractFile.objects.get(version_id=version_id)
            encrypted=next(Path(folder).glob('contracts/expert/*/evidence.txt')).read_bytes()
            self.assertNotEqual(encrypted,b'Pay within 30 days.')
            self.assertEqual(decrypt_pdf(encrypted),b'Pay within 30 days.')
            run=LayoutExtractionRun(version_id=version_id,status='COMPLETED',structure=dict(source_fingerprint='real-source'))
            record=dict(chunk_id='block:7',text='Pay within 30 days.',page_number=1,source_url='/source/')
            with patch('ai_extract.services.contract_pipeline_service.LayoutExtractionService.extract_version',return_value=run) as extract, \
                 patch('NewIdea.repositories.contract_repository.version_sources',return_value=[record]), \
                 patch('ai_extract.services.contract_pipeline_service.request_qwen',side_effect=self.model), \
                 patch('NewIdea.repositories.expert_submission_repository.sync_semantic_graph',return_value=dict(status='ready')):
                process_next_job(lambda *a,**k:self.fail('Unexpected version job'))
            extract.assert_called_once()
            result=KnowledgeSearchLog.objects.get(pk=identity).result['response']
            self.assertEqual(result['documents'][0]['version_id'],version_id)
            self.assertEqual(result['units'][0]['source_chunk_id'],'block:7')

    def test_unsupported_upload_is_rejected_before_saving(self):
        upload=SimpleUploadedFile('bad.exe',b'payload')
        response=self.client.post('/api/newidea/expert-submissions/',dict(question='Question',files=upload))
        self.assertEqual(response.status_code,400)
        self.assertEqual(KnowledgeSearchLog.objects.count(),0)

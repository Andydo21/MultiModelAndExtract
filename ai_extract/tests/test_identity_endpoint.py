from types import SimpleNamespace
from unittest.mock import patch
from django.contrib.auth.models import AnonymousUser
from django.test import RequestFactory, SimpleTestCase
from ai_extract.views.layout_views import api_resolve_entities,api_run_layout_extraction
from ai_extract.management.commands.resolve_document_entities import Command


class IdentityEndpointTests(SimpleTestCase):
    def setUp(self):
        self.request=RequestFactory().post('/api/ai/contracts/1/resolve-entities/')
        self.request.user=SimpleNamespace(is_authenticated=True)

    def test_unauthenticated_request_never_queues_a_job(self):
        self.request.user=AnonymousUser()
        with patch('ai_extract.services.layout_service.queue_identity') as queue:
            self.assertEqual(api_resolve_entities(self.request,1).status_code,302)
            queue.assert_not_called()

    def test_authenticated_post_uses_project_service(self):
        with patch('ai_extract.services.layout_service.queue_identity',
                   return_value=dict(success=True,run_id=7,identity_state='pending')) as queue:
            self.assertEqual(api_resolve_entities(self.request,1).status_code,202)
            queue.assert_called_once_with(self.request.user,1)

    def test_get_never_mutates_the_queue(self):
        request=RequestFactory().get('/api/ai/contracts/1/resolve-entities/')
        request.user=self.request.user
        with patch('ai_extract.services.layout_service.queue_identity') as queue:
            self.assertEqual(api_resolve_entities(request,1).status_code,405)
            queue.assert_not_called()

    def test_missing_source_graph_is_not_reported_as_success(self):
        with patch('ai_extract.services.layout_service.queue_identity',side_effect=ValueError('Graph unavailable')):
            self.assertEqual(api_resolve_entities(self.request,1).status_code,400)

    def test_background_worker_defers_while_extraction_uses_gpu(self):
        with patch('ai_extract.repositories.document_graph_repository.entity_schema'), \
             patch('ai_extract.repositories.layout_repository.extraction_in_progress',return_value=True), \
             patch('ai_extract.repositories.document_graph_repository.claim_entity_job') as claim, \
             patch('ai_extract.management.commands.resolve_document_entities.time.sleep',side_effect=KeyboardInterrupt):
            Command().handle(watch=True,run_id=None)
            claim.assert_not_called()

    def test_fresh_extraction_options_reach_the_real_service(self):
        request=RequestFactory().post('/api/ai/contracts/1/layout-extract/',
            data='{"force_ocr":true,"defer_colpali":true}',content_type='application/json')
        request.user=self.request.user
        with patch('ai_extract.services.layout_service.extract',return_value=dict(success=True)) as extract:
            self.assertEqual(api_run_layout_extraction(request,1).status_code,200)
            extract.assert_called_once_with(request.user,1,force_ocr=True,defer_colpali=True)

    def test_string_flag_cannot_silently_enable_fresh_extraction(self):
        request=RequestFactory().post('/api/ai/contracts/1/layout-extract/',
            data='{"force_ocr":"false"}',content_type='application/json')
        request.user=self.request.user
        with patch('ai_extract.services.layout_service.extract') as extract:
            self.assertEqual(api_run_layout_extraction(request,1).status_code,400)
            extract.assert_not_called()

    def test_busy_workflow_is_an_explicit_retryable_api_response(self):
        from ai_extract.services.contract_pipeline_service import WorkflowBusyError
        with patch('ai_extract.services.layout_service.extract',side_effect=WorkflowBusyError('Resource busy')):
            response=api_run_layout_extraction(self.request,1)
        self.assertEqual(response.status_code,409)
        self.assertEqual(response['Retry-After'],'5')

    def test_worker_does_not_claim_a_job_without_the_shared_resource(self):
        from contextlib import contextmanager
        @contextmanager
        def busy():yield False
        with patch('ai_extract.repositories.layout_repository.workflow_slot',busy), \
             patch('ai_extract.repositories.document_graph_repository.claim_entity_job') as claim:
            self.assertIsNone(Command().process_next())
            claim.assert_not_called()

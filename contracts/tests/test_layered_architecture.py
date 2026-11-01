"""Layer boundaries and endpoint behavior after the multi-app refactor."""
import ast
import json
from pathlib import Path
from unittest.mock import patch
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import resolve, Resolver404, reverse
from contracts.tests import test_intelligence as intelligence_tests
from ai_extract.models import ContractSummary


class LayerBoundaryTests(SimpleTestCase):
    def test_newidea_views_use_services_and_repositories_do_not_call_services(self):
        root=Path(__file__).resolve().parents[2]/'NewIdea'
        for layer in ('views','repositories'):
            for path in (root/layer).glob('*.py'):
                for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                    if not isinstance(node,ast.ImportFrom) or not node.module:continue
                    with self.subTest(file=path.name,module=node.module):
                        if layer=='views':
                            self.assertNotIn('repositories',node.module.split('.'))
                        else:
                            self.assertNotIn('services',node.module.split('.'))
                            self.assertFalse(node.module.startswith('django.http'))

    def test_web_has_one_ai_processing_application(self):
        from django.apps import apps
        root = Path(__file__).resolve().parents[2]
        self.assertTrue(apps.is_installed('ai_extract'))
        self.assertFalse(apps.is_installed('ai_extract.legacy_schema'))
        self.assertFalse((root/'ai_extract/a3/documents').exists())
        self.assertFalse((root/'ai_extract/a3/service.py').exists())
        self.assertTrue((root/'ai_extract/services/contract_pipeline_service.py').is_file())
        for layer in ('services', 'views', 'repositories'):
            self.assertFalse((root/'ai_extract/legacy_schema'/layer).exists())

    @override_settings(ROOT_URLCONF='contracts.tests.integration_urls')
    def test_retired_blockchain_routes_are_removed(self):
        routes = (
            '/api/contracts/1/blockchain/proof/',
            '/api/contracts/1/blockchain/anchor/',
            '/api/contracts/1/blockchain/verify/',
            '/api/contracts/1/blockchain/status/',
            '/api/blockchain/certificates/', '/api/blockchain/sign/',
            '/api/blockchain/history/1/', '/api/blockchain/transaction/old/',
            '/api/blockchain/proof/1/', '/api/blockchain/certificate/1/',
            '/api/blockchain/user/1/', '/api/blockchain/explorer/stats/',
            '/blockchain-explorer/',
            '/developer-test/', '/api/developer-test/run/',
        )
        for route in routes:
            with self.subTest(route=route), self.assertRaises(Resolver404):
                resolve(route)
        self.assertEqual(reverse('contract_ledger'), '/contract-ledger/')
        self.assertEqual(resolve('/api/ledger/verify/').url_name, 'ledger_verify')

    def test_controllers_and_services_do_not_query_models_directly(self):
        root = Path(__file__).resolve().parents[2]
        for app in ('contracts', 'ai_extract', 'NewIdea'):
            for layer in ('views', 'services'):
                for path in (root/app/layer).glob('*.py'):
                    tree = ast.parse(path.read_text(encoding='utf-8'))
                    with self.subTest(file=str(path.relative_to(root))):
                        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == 'objects' for n in ast.walk(tree)))
                        if layer == 'services':
                            self.assertFalse(any(isinstance(n, ast.ImportFrom) and n.module and n.module.startswith('django.http') for n in ast.walk(tree)))
                            self.assertFalse(any(isinstance(n, ast.Name) and n.id == 'request' for n in ast.walk(tree)))
    def test_a3_persistence_uses_repository(self):
        path = Path(__file__).resolve().parents[2]/'ai_extract/services/contract_pipeline_service.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == 'objects' for n in ast.walk(tree)))


@override_settings(ROOT_URLCONF='contracts.tests.integration_urls')
class LayeredEndpointTests(TestCase):
    setUp = intelligence_tests.ContractIntelligenceTests.setUp

    def test_contract_list_uses_existing_service_and_repository(self):
        response = self.client.get('/api/contracts/')
        self.assertEqual(response.status_code, 200)
        text = json.dumps(response.json())
        self.assertIn(self.contract.contract_code, text)
        self.assertNotIn(self.foreign.contract_code, text)

    def test_html_detail_and_api_detail_are_distinct_operations(self):
        with patch('contracts.services.workflow_service.WorkflowService.get_workflow_status', return_value=None):
            page = self.client.get(f'/contracts/{self.contract.pk}/')
            response = self.client.get(f'/api/contracts/{self.contract.pk}/')
        self.assertEqual(page.status_code, 200)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['id'], self.contract.pk)

    def test_cross_company_detail_is_denied_before_business_processing(self):
        with patch('contracts.services.contract_operations_service.contract_service.get_contract_details') as details:
            self.assertEqual(self.client.get(f'/api/contracts/{self.foreign.pk}/').status_code, 403)
        details.assert_not_called()

    def test_legacy_catalog_routes_still_return_json_lists(self):
        for route in ('companies', 'users', 'roles'):
            response = self.client.get(f'/api/identity/{route}/')
            self.assertEqual(response.status_code, 200)
            self.assertIsInstance(response.json(), list)
        self.assertEqual(self.client.get('/api/risks/').status_code, 200)

    def test_invalid_json_is_rejected_by_controller(self):
        self.client.force_login(self.manager)
        response = self.client.post('/api/risks/', '{invalid', content_type='application/json')
        self.assertEqual(response.status_code, 400)

    def test_ai_summary_is_scoped_to_company_before_model_call(self):
        ContractSummary.objects.create(version=self.new_version, summary='Stored summary')
        response = self.client.get(f'/api/ai/contracts/{self.contract.pk}/summary/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['summary'], 'Stored summary')
        with patch('ai_extract.services.extraction_service.SummarizeService') as model:
            response = self.client.post(f'/api/ai/contracts/{self.foreign.pk}/summarize/', '{}', content_type='application/json')
        self.assertEqual(response.status_code, 404)
        model.assert_not_called()

    def test_anonymous_extraction_and_admin_mutation_are_rejected(self):
        self.client.logout()
        self.assertEqual(self.client.get(f'/api/ai/contracts/{self.contract.pk}/summary/').status_code, 401)
        self.assertEqual(self.client.post(f'/api/admin/users/{self.user.pk}/delete/').status_code, 401)

    def test_version_routes_require_login_and_contract_scope(self):
        with patch('contracts.services.contract_operations_service.contract_service.create_new_version') as create:
            self.assertEqual(self.client.get(f'/api/contracts/{self.foreign.pk}/versions/').status_code, 403)
            self.assertEqual(self.client.post(f'/api/contracts/{self.contract.pk}/versions/', {'raw_content': 'Changed'}).status_code, 403)
            self.client.force_login(self.manager)
            self.assertEqual(self.client.post(f'/api/contracts/{self.foreign.pk}/versions/', {'raw_content': 'Changed'}).status_code, 403)
            create.assert_not_called()
        self.client.logout()
        self.assertEqual(self.client.get(f'/api/contracts/{self.contract.pk}/versions/').status_code, 401)

    def test_workflow_scope_is_checked_before_remote_call(self):
        self.client.force_login(self.manager)
        with patch('contracts.services.workflow_gateway_service.workflow_service.push_to_workflow') as push, patch('contracts.services.workflow_gateway_service.workflow_service.get_workflow_status') as status:
            self.assertEqual(self.client.post(f'/api/contracts/{self.foreign.pk}/workflow/', '{}', content_type='application/json').status_code, 403)
            self.assertEqual(self.client.get(f'/api/contracts/{self.foreign.pk}/workflow/status/').status_code, 403)
            push.assert_not_called()
            status.assert_not_called()

    def test_companyless_account_cannot_list_all_contracts(self):
        self.user.company = None
        self.user.save(update_fields=['company'])
        self.assertEqual(self.client.get('/api/contracts/').json(), [])

import json
from django.contrib.auth import get_user_model
from django.test import TestCase, Client, override_settings


class ContractModeTests(TestCase):
    def test_retired_writes_are_not_exposed(self):
        for endpoint in ('expert_answer', 'expert_consult', 'contribute_knowledge', 'clear_chat_history'):
            self.assertEqual(self.client.post(f'/api/{endpoint}/').status_code, 404)

    @override_settings(NEWIDEA_MODE='contracts',CONTRACT_WEB_URL='http://localhost:8000')
    def test_chat_opens_authenticated_main_web_search(self):
        response=self.client.get('/chat/')
        self.assertEqual(response.status_code,302)
        self.assertEqual(response.url,'http://localhost:8000/newidea/')

    @override_settings(NEWIDEA_MODE='contracts')
    def test_old_anonymous_search_cannot_read_contract_indexes(self):
        response=self.client.post('/api/ask/',json.dumps({'query':'payment'}),content_type='application/json')
        self.assertEqual(response.status_code,404)



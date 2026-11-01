import unittest
from unittest.mock import Mock, patch
import requests
from shared.qwen_gateway import chat, is_connected


@patch.dict('os.environ', {'VLLM_BASE_URL': 'https://gateway.test/v1', 'QWEN_MODEL_ID': 'test-model'})
class QwenGatewayTests(unittest.TestCase):
    @patch('shared.qwen_gateway.requests.get')
    def test_models_endpoint_is_used_for_dual_gateway(self, get):
        get.return_value = Mock(ok=True)
        get.return_value.json.return_value = {'data': [{'id': 'test-model'}]}
        self.assertTrue(is_connected('https://gateway.test'))
        self.assertEqual(get.call_args.args[0], 'https://gateway.test/v1/models')

    @patch('shared.qwen_gateway.requests.get')
    def test_wrong_model_does_not_report_healthy(self, get):
        get.return_value = Mock(ok=True)
        get.return_value.json.return_value = {'data': [{'id': 'other-model'}]}
        self.assertFalse(is_connected())

    @patch('shared.qwen_gateway.requests.get', side_effect=requests.ConnectionError('offline'))
    def test_unreachable_gateway_does_not_report_healthy(self, get):
        self.assertFalse(is_connected())

    @patch('shared.qwen_gateway.requests.post')
    def test_invalid_json_is_rejected(self, post):
        post.return_value.json.return_value = {'choices': [{'message': {'content': 'not JSON'}}]}
        with self.assertRaises(ValueError):
            chat('system', 'test', json_output=True)

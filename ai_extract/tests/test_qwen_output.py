"""Raw API output is preserved without mixing concurrent requests."""
import json
import io
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock, patch
from django.test import SimpleTestCase, override_settings
from ai_extract.services.contract_pipeline_service import request_qwen, QwenGatewayError
from NewIdea.services.knowledge_answer_service import generate_checked_answer


def api_response(content, identity='request-one', finish_reason='stop'):
    body=dict(id=identity, model='Qwen-test', usage=dict(prompt_tokens=20,completion_tokens=10,total_tokens=30),
              choices=[dict(finish_reason=finish_reason,message=dict(role='assistant',content=content))])
    response=Mock(status_code=200)
    response.json.return_value=body
    return response, body


@patch('ai_extract.services.contract_pipeline_service._vllm_base_url',return_value='http://unused/v1')
class QwenOutputTests(SimpleTestCase):
    def test_runtime_json_log_contains_decoded_model_content(self, base):
        content = dict(supported=True, reason='Nguồn xác nhận không cần PC.')
        response, original = api_response(json.dumps(content, ensure_ascii=False))
        stream = io.StringIO()
        with redirect_stdout(stream), patch('ai_extract.services.contract_pipeline_service.requests.post', return_value=response):
            request_qwen('verify_answer', dict(evidence=[dict(evidence_id='E1', text='PC')]))
        event = json.loads(stream.getvalue())
        self.assertEqual(event['event'], 'qwen_output')
        self.assertEqual(event['task'], 'verify_answer')
        self.assertEqual(event['outputs'][0]['content'], content)
        self.assertEqual(event['usage'], original['usage'])
        self.assertEqual(event['outputs'][0]['finish_reason'], 'stop')
        self.assertNotIn('messages', event)

    def test_rejected_content_is_also_printed_to_runtime_logs(self, base):
        response, _ = api_response('{"claims":[', finish_reason='length')
        stream = io.StringIO()
        with redirect_stdout(stream), patch('ai_extract.services.contract_pipeline_service.requests.post', return_value=response):
            with self.assertRaises(QwenGatewayError):
                request_qwen('verify_answer', dict(evidence=[]))
        events = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(len(events),4)
        for event in events:
            self.assertEqual(event['outputs'][0]['content'], '{"claims":[')
            self.assertEqual(event['outputs'][0]['finish_reason'], 'length')

    @override_settings(KAGGLE_QWEN_SERVICE_URL='http://unused')
    def test_legacy_output_does_not_create_a_circular_json_reference(self, base):
        base.return_value=''
        response=Mock(status_code=200)
        original=dict(summary='Source summary')
        response.json.return_value=original
        with patch('ai_extract.services.contract_pipeline_service.requests.post',return_value=response):
            result=request_qwen('summarize',dict(clauses=[]))
        self.assertEqual(result['qwen_outputs'],[dict(task='summarize',response=original)])
        self.assertNotIn('qwen_outputs',original)
        self.assertTrue(json.dumps(result))

    def test_exact_api_envelope_and_token_usage_are_preserved(self, base):
        response, original=api_response('{"supported":true,"reason":"source supports it"}')
        with patch('ai_extract.services.contract_pipeline_service.requests.post',return_value=response):
            result=request_qwen('verify_answer',dict(query='PC?',evidence=[dict(evidence_id='E1',text='PC')],answer='PC [E1].'))
        self.assertEqual(result['qwen_outputs'],[dict(task='verify_answer',response=original)])
        self.assertTrue(result['supported'])
        self.assertNotIn('messages',result['qwen_outputs'][0])

    def test_unparseable_model_content_is_kept_for_diagnosis(self, base):
        response, original=api_response('not valid model JSON')
        with patch('ai_extract.services.contract_pipeline_service.requests.post',return_value=response):
            with self.assertRaises(QwenGatewayError) as raised:
                request_qwen('verify_answer',dict(evidence=[dict(evidence_id='E1',text='source')]))
        self.assertEqual(raised.exception.qwen_outputs[0]['response'],original)

    def test_truncated_output_is_visible_but_not_accepted_as_an_answer(self, base):
        response, original=api_response('{"claims":[',finish_reason='length')
        with patch('ai_extract.services.contract_pipeline_service.requests.post',return_value=response):
            result=generate_checked_answer(dict(query='PC?',evidence=[dict(evidence_id='E1',text='PC')],answer_format='cited_claims'))
        self.assertEqual(result['generation_diagnostics']['state'],'output_limit_reached')
        # Every bounded retry remains visible; none is accepted as a complete answer.
        self.assertEqual(result['qwen_outputs'],[dict(task='contract_qa',stage='failed',response=original)]*4)
        self.assertEqual(result['generation_status'],'source_excerpt')

    def test_concurrent_calls_never_share_output_lists(self, base):
        barrier=Barrier(2)
        def post(url, **kwargs):
            key='one' if '[one]' in kwargs['json']['messages'][-1]['content'] else 'two'
            barrier.wait(timeout=5)
            return api_response(key,identity=key)[0]
        with patch('ai_extract.services.contract_pipeline_service.requests.post',side_effect=post):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results=list(pool.map(lambda key: request_qwen('summarize',dict(clauses=[dict(title=key,content='Source')])),('one','two')))
        for key, result in zip(('one','two'),results):
            self.assertEqual(result['summary'],key)
            self.assertEqual([r['response']['id'] for r in result['qwen_outputs']],[key])


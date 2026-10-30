from unittest.mock import patch
from copy import deepcopy
from django.test import SimpleTestCase, override_settings
from ai_extract.services.contract_pipeline_service import QwenContextLimitError, QwenGatewayError, QwenServerError
from NewIdea.services.knowledge_answer_service import request_answer_with_context_retry


@override_settings(A2_MAP_REDUCE_ENABLED=True)
class ContextRetryTests(SimpleTestCase):
    def test_server_error_splits_request_and_preserves_sources(self):
        calls=[]
        def model(task,data):
            ids=[s['evidence_id'] for s in data['evidence']]
            calls.append(ids)
            if len(ids)>2: raise QwenServerError('HTTP 500')
            return dict(answer='Fact',citation_claims=[dict(text='Fact',evidence_ids=[ids[0]])])
        result=request_answer_with_context_retry(model,self.payload(),synthesize=False)
        self.assertEqual(calls[0],['E0','E1','E2','E3'])
        self.assertEqual(set(sum(calls[1:],[])),{'E0','E1','E2','E3'})
        self.assertEqual(result['generation_input']['unread_source_ids'],[])

    def test_server_error_splits_are_bounded(self):
        calls=[]
        def model(task,data):
            calls.append(data)
            raise QwenServerError('HTTP 500')
        with self.assertRaises(QwenServerError):
            request_answer_with_context_retry(model,self.payload())
        self.assertLessEqual(len(calls),7)

    def payload(self):
        return dict(query='Explain all parts',evidence=[dict(evidence_id=f'E{i}',text=f'Passage {i}') for i in range(4)],
            graph_context=dict(nodes=[],edges=[]))

    @override_settings(A2_MAP_REDUCE_ENABLED=False)
    def test_disabled_experiment_keeps_sequential_pending_part_fallback(self):
        payload=self.payload();payload['answer_parts']=[dict(id='P1',question='First?')]
        delivered=[]
        def model(task,data):
            if len(data['evidence'])>2:
                error=QwenContextLimitError(16384,17000,8192);error.preflight=True
                raise error
            delivered.extend(s['evidence_id'] for s in data['evidence'])
            return dict(answer='Fact',citation_claims=[dict(text='Fact',evidence_ids=['E0'],part_ids=['P1'])])
        result=request_answer_with_context_retry(model,payload)
        self.assertEqual(delivered,['E0','E1'])
        self.assertEqual(result['generation_input']['unread_source_ids'],['E2','E3'])
        self.assertEqual(result['generation_input']['context_retry'],'shorter_output_then_source_batches')

    def test_retries_with_less_output_without_removing_sources(self):
        payload=self.payload();original=deepcopy(payload);calls=[]
        def model(task,data):
            calls.append(data)
            if len(calls)==1:raise QwenContextLimitError(16384,14000,8192)
            return dict(answer='complete')
        self.assertEqual(request_answer_with_context_retry(model,payload)['answer'],'complete')
        self.assertEqual(calls[1]['max_output_tokens'],2048)
        self.assertEqual(calls[1]['evidence'],original['evidence'])
        self.assertEqual(payload,original)

    def test_oversized_input_reads_every_passage_then_merges_real_claims(self):
        payload=self.payload();original=deepcopy(payload);delivered=[]
        def model(task,data):
            if task=='verify_answer':
                return dict(claim_checks=[dict(check_id=str(i),supported=True) for i,_ in enumerate(data['citation_claims'])])
            if len(data['evidence'])>2:raise QwenContextLimitError(16384,16385,8192)
            delivered.extend(data['evidence'])
            return dict(answer='Batch answer',citation_claims=[dict(text=s['text'],evidence_ids=[s['evidence_id']]) for s in data['evidence']])
        result=request_answer_with_context_retry(model,payload)
        self.assertCountEqual(delivered,original['evidence'])
        self.assertEqual(len(result['citation_claims']),4)
        self.assertEqual(result['generation_input']['context_retry'],'parallel_source_maps')
        self.assertEqual(payload,original)

    def test_batch_merge_keeps_claims_without_per_batch_verification_calls(self):
        payload=self.payload()
        def model(task,data):
            if task=='verify_answer':
                self.fail('Batch processing must defer review until the merged answer')
            if len(data['evidence'])>2:raise QwenContextLimitError(16384,16385,8192)
            return dict(answer='Batch answer',citation_claims=[dict(text=s['text'],evidence_ids=[s['evidence_id']]) for s in data['evidence']])
        result=request_answer_with_context_retry(model,payload)
        self.assertEqual([c['evidence_ids'] for c in result['citation_claims']],[['E0'],['E1'],['E2'],['E3']])

    def test_single_oversized_passage_is_never_silently_cut(self):
        payload=self.payload();payload['evidence']=payload['evidence'][:1]
        def model(task,data):raise QwenContextLimitError(16384,16385,8192)
        with self.assertRaises(QwenContextLimitError):request_answer_with_context_retry(model,payload)

    def test_reads_later_sources_even_when_an_earlier_map_answers_all_parts(self):
        payload=self.payload();payload['answer_parts']=[dict(id='P1',question='Explain')]
        delivered=[]
        def model(task,data):
            if task=='verify_answer':return dict(claim_checks=[dict(check_id='0',supported=True)])
            if len(data['evidence'])>2:raise QwenContextLimitError(16384,16385,8192)
            delivered.extend(data['evidence'])
            source=data['evidence'][0]
            return dict(answer='Supported answer',citation_claims=[dict(text=source['text'],evidence_ids=[source['evidence_id']],part_ids=['P1'])])
        result=request_answer_with_context_retry(model,payload)
        self.assertCountEqual([s['evidence_id'] for s in delivered],['E0','E1','E2','E3'])
        self.assertEqual(result['generation_input']['unread_source_ids'],[])

    def test_map_cannot_attach_a_claim_to_another_documents_evidence(self):
        def model(task,data):
            if len(data['evidence'])>1:
                error=QwenContextLimitError(16384,17000,8192);error.preflight=True
                raise error
            return dict(answer='Claim',citation_claims=[dict(text='Claim',evidence_ids=['E99'])])
        payload=dict(query='Facts?',evidence=[dict(evidence_id=f'E{i}',text='Fact',document_id=f'D{i}')
            for i in range(2)])
        with self.assertRaisesMessage(QwenGatewayError,'outside its assigned source group'):
            request_answer_with_context_retry(model,payload)

    def test_failed_map_keeps_successful_claims_and_reports_unread_sources(self):
        sources=[dict(evidence_id=f'E{i}',text=f'Fact {i}',document_id=f'D{i}') for i in range(2)]
        def model(task,data):
            if len(data['evidence'])>1:
                error=QwenContextLimitError(16384,17000,8192);error.preflight=True
                raise error
            source=data['evidence'][0]
            if source['evidence_id']=='E1':raise QwenGatewayError('timeout')
            return dict(answer='Fact 0',citation_claims=[dict(text='Fact 0',evidence_ids=['E0'])])
        result=request_answer_with_context_retry(model,dict(query='Facts?',evidence=sources))
        self.assertEqual([c['evidence_ids'] for c in result['citation_claims']],[['E0']])
        self.assertEqual(result['generation_input']['processed_source_ids'],['E0'])
        self.assertEqual(result['generation_input']['unread_source_ids'],['E1'])
        self.assertEqual(result['answer_notes'][0]['evidence_ids'],['E1'])

    @override_settings(A2_MAP_WORKERS=2)
    def test_recursive_splits_do_not_create_more_concurrent_requests(self):
        import time
        from threading import Lock
        active=peak=0
        lock=Lock()
        def model(task,data):
            nonlocal active,peak
            if len(data['evidence'])>1:
                error=QwenContextLimitError(16384,17000,8192);error.preflight=True
                raise error
            with lock:
                active+=1;peak=max(peak,active)
            try:
                time.sleep(.02)
                source=data['evidence'][0]
                return dict(answer='Fact',citation_claims=[dict(text=source['text'],evidence_ids=[source['evidence_id']])])
            finally:
                with lock:active-=1
        sources=[dict(evidence_id=f'E{i}',text=f'Fact {i}',document_id=f'D{i%3}') for i in range(9)]
        result=request_answer_with_context_retry(model,dict(query='Facts?',evidence=sources))
        self.assertEqual(peak,2)
        self.assertEqual(len(result['citation_claims']),9)
        self.assertEqual(result['generation_input']['unread_source_ids'],[])


@override_settings(A2_MAP_REDUCE_ENABLED=True)
class TokenPreflightTests(SimpleTestCase):
    @override_settings(QWEN_TOKEN_PREFLIGHT=False)
    def test_short_completion_at_context_boundary_triggers_source_split(self):
        from ai_extract.services.contract_pipeline_service import _chat
        from unittest.mock import Mock
        reply=Mock(status_code=200)
        reply.json.return_value=dict(usage=dict(prompt_tokens=16208,completion_tokens=176,total_tokens=16384),
                                     choices=[dict(finish_reason='length',message=dict(content='partial'))])
        with patch('ai_extract.services.contract_pipeline_service.requests.post',return_value=reply):
            with self.assertRaises(QwenContextLimitError):
                _chat('http://model/v1',[dict(role='user',content='Full source text')],max_tokens=8192)

    @override_settings(QWEN_TOKEN_PREFLIGHT=False)
    def test_gateway_input_length_overflow_is_recognized(self):
        from ai_extract.services.contract_pipeline_service import _chat
        from unittest.mock import Mock
        reply=Mock(status_code=400,text="Input length (17115) exceeds model's maximum context length (16384).")
        reply.json.return_value={'error':{'message':reply.text}}
        with patch('ai_extract.services.contract_pipeline_service.requests.post',return_value=reply):
            with self.assertRaises(QwenContextLimitError) as error:
                _chat('http://model/v1',[dict(role='user',content='Full source text')],max_tokens=8192)
        self.assertEqual(error.exception.capacity,16384)
        self.assertEqual(error.exception.prompt_tokens,17115)

    @override_settings(QWEN_TOKEN_PREFLIGHT=True)
    def test_budget_is_reduced_before_generation_without_removing_source_text(self):
        from ai_extract.services.contract_pipeline_service import _chat
        from unittest.mock import Mock
        reply = Mock(status_code=200)
        reply.json.return_value = dict(choices=[dict(finish_reason='stop',message=dict(content='Done'))])
        messages=[dict(role='user',content='Full source text')]
        with patch('ai_extract.services.contract_pipeline_service._chat_token_budget',return_value=(15000,16384)), \
             patch('ai_extract.services.contract_pipeline_service.requests.post',return_value=reply) as http:
            self.assertEqual(_chat('http://model/v1',messages,max_tokens=8192),'Done')
        body=http.call_args.kwargs['json']
        self.assertEqual(body['max_tokens'],1320)
        self.assertEqual(body['messages'],messages)

    @override_settings(QWEN_TOKEN_PREFLIGHT=True)
    def test_oversized_input_never_reaches_generation_endpoint(self):
        from ai_extract.services.contract_pipeline_service import _chat
        with patch('ai_extract.services.contract_pipeline_service._chat_token_budget',return_value=(17000,16384)), \
             patch('ai_extract.services.contract_pipeline_service.requests.post') as http:
            with self.assertRaises(QwenContextLimitError) as error:
                _chat('http://model/v1',[dict(role='user',content='Full source text')],max_tokens=8192)
        self.assertTrue(error.exception.preflight)
        http.assert_not_called()

    def test_preflight_overflow_splits_sources_without_repeating_full_prompt(self):
        calls=[]
        payload=dict(query='Both facts',evidence=[dict(evidence_id='E1',text='first'),dict(evidence_id='E2',text='second')])
        def model(task,data):
            calls.append((task,len(data['evidence'])))
            if len(data['evidence'])>1:
                error=QwenContextLimitError(16384,17000,8192)
                error.preflight=True
                raise error
            return dict(answer='No source-supported fact',citation_claims=[],insufficient_evidence=True)
        request_answer_with_context_retry(model,payload)
        self.assertEqual(calls,[('contract_qa',2),('contract_qa',1),('contract_qa',1)])


@override_settings(A2_MAP_REDUCE_ENABLED=True)
class FocusedBatchTests(SimpleTestCase):
    def test_reduce_receives_original_cited_passages_and_marks_maps_as_unverified(self):
        sources=[dict(evidence_id=f'E{i}',text=f'Original passage {i}',document_id=f'D{i//2}') for i in range(4)]
        reductions=[]
        def model(task,data):
            if data.get('mapped_findings'):
                reductions.append(data)
                return dict(answer='Synthesis',citation_claims=[dict(text='Supported synthesis',evidence_ids=['E0','E2'])])
            if len(data['evidence'])>2:
                error=QwenContextLimitError(16384,17000,8192);error.preflight=True
                raise error
            source=data['evidence'][0]
            return dict(answer='Map',citation_claims=[dict(text='Mapped '+source['text'],evidence_ids=[source['evidence_id']])])
        result=request_answer_with_context_retry(model,dict(query='Compare?',evidence=sources,check_completeness=True))
        self.assertEqual(len(reductions),1)
        self.assertEqual(reductions[0]['evidence'],[sources[0],sources[2]])
        self.assertEqual(len(reductions[0]['mapped_findings']),2)
        self.assertIn('unverified proposals',reductions[0]['instruction'])
        self.assertEqual(result['generation_input']['reduce'],'model_synthesis_with_original_sources')
        self.assertEqual(result['citation_claims'][0]['text'],'Supported synthesis')

    def test_maps_route_parts_using_their_source_bindings(self):
        sources=[dict(evidence_id='E'+str(i),text='Source '+str(i)) for i in range(4)]
        payload=dict(query='Explain both',evidence=sources,graph_context={},answer_parts=[
            dict(id='P1',question='First?',source_ids=['E0']),
            dict(id='P2',question='Second?',source_ids=['E2'])],coverage={})
        batches=[]
        def model(task,data):
            if task=='verify_answer':return dict(claim_checks=[dict(check_id='0',supported=True)])
            if len(data['evidence'])>2:raise QwenContextLimitError(16384,17000,8192)
            batches.append(data)
            part=data['answer_parts'][0]['id']
            return dict(answer='Answer',citation_claims=[dict(text='Supported '+part,evidence_ids=[data['evidence'][0]['evidence_id']],part_ids=[part])])
        result=request_answer_with_context_retry(model,payload)
        self.assertCountEqual([[p['id'] for p in batch['answer_parts']] for batch in batches],[['P1'],['P2']])
        self.assertEqual([p['status'] for p in result['answer_part_coverage']],['answered','answered'])

    @override_settings(A2_MAP_WORKERS=2)
    def test_document_maps_overlap_and_preserve_source_order_at_reduce(self):
        from threading import Barrier
        barrier=Barrier(2)
        sources=[dict(evidence_id=f'E{i}',text=f'Fact {i}',document_id=f'D{i%2}') for i in range(4)]
        delivered=[]
        def model(task,data):
            if len(data['evidence'])>2:
                error=QwenContextLimitError(16384,17000,8192);error.preflight=True
                raise error
            barrier.wait(timeout=5)  # A sequential map cannot pass this barrier.
            delivered.append(data['evidence'])
            self.assertEqual(len({s['document_id'] for s in data['evidence']}),1)
            return dict(answer='Facts',citation_claims=[dict(text=s['text'],evidence_ids=[s['evidence_id']])
                for s in data['evidence']])
        result=request_answer_with_context_retry(model,dict(query='Facts?',evidence=sources))
        self.assertEqual(len(delivered),2)
        self.assertEqual([c['evidence_ids'] for c in result['citation_claims']],[['E0'],['E2'],['E1'],['E3']])
        self.assertEqual(result['generation_input']['unread_source_ids'],[])

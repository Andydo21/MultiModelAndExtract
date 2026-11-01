from django.test import SimpleTestCase
import json
from django.test import override_settings
from unittest.mock import Mock, patch
from ai_extract.services.contract_pipeline_service import (
    validate_answer, structural_retrieve, render_cited_claims,
    _request_openai_compatible, QwenGatewayError,
    _chat, QwenContextLimitError, cited_claim_schema,
)
from ai_extract.services.document_structure_service import analyze_document_structure


class LiveFlowRegressionTests(SimpleTestCase):
    def test_identical_answers_merge_without_losing_parts_or_quotes(self):
        sources=[dict(evidence_id='E1',text='Load is 8 kg. Rated capacity is 8 kg.')]
        data=dict(claims=[dict(text='Load is 8 kg.',evidence_ids=['E1'],part_ids=['P1'],
            quotes=[dict(evidence_id='E1',quote='Load is 8 kg.')]),
            dict(text='Load is 8 kg.',evidence_ids=['E1'],part_ids=['P2'],
            quotes=[dict(evidence_id='E1',quote='Rated capacity is 8 kg.')])])
        answer=render_cited_claims(data,sources)
        self.assertEqual(answer.count('Load is 8 kg'),1)
        self.assertEqual(data['claims'][0]['part_ids'],['P1','P2'])
        self.assertEqual(len(data['claims'][0]['quotes']),2)
        self.assertEqual(len(data['claims']),1)

    def test_dedup_preserves_different_sources_values_conditions_and_line_structure(self):
        sources=[dict(evidence_id='E1',text='Source one.'),dict(evidence_id='E2',text='Source two.')]
        claims=[dict(text=text,evidence_ids=ids) for text,ids in [
            ('Load is 8 kg.',['E1']),('Load is 8 kg.',['E2']),
            ('Load is 5 kg.',['E1']),('Load is not 8 kg.',['E1']),
            ('Load is 8 kg when stationary.',['E1']),
            ('A B\n1 2',['E1']),('A\nB 1\n2',['E1'])]]
        data=dict(claims=claims)
        render_cited_claims(data,sources)
        self.assertEqual(data['claims'],claims)

    def test_source_limit_precedes_answer_and_remains_advisory(self):
        limit='The supplied passage does not specify the implementation.'
        output=dict(insufficient_evidence=False,claims=[dict(
            evidence_ids=['E1'],quotes=[dict(evidence_id='E1',quote='The function prevents damage.')],
            source_limit=limit,answer='The function prevents damage; its implementation is not specified.')],
            answer_notes=[dict(text=limit,evidence_ids=['E1'])])
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(output)) as chat:
            result=_request_openai_compatible('http://unused','contract_qa',dict(
                query='How does this function work?',inline_review=True,check_completeness=True,
                answer_format='cited_claims',evidence=[dict(evidence_id='E1',text='The function prevents damage.')]))
        self.assertEqual(chat.call_count,1)
        schema=chat.call_args.kwargs['json_schema']['properties']['claims']['items']
        fields=list(schema['properties'])
        self.assertLess(fields.index('quotes'),fields.index('answer'))
        self.assertLess(fields.index('source_limit'),fields.index('answer'))
        self.assertIn('source_limit',schema['required'])
        self.assertEqual(schema['properties']['quotes']['minItems'],1)
        self.assertIn('The function prevents damage',result['answer'])
        self.assertEqual(result['answer_notes'],[dict(text=limit,evidence_ids=['E1'])])
        self.assertFalse(result['insufficient_evidence'])

    def test_inline_notes_and_compact_packets_share_one_generation(self):
        notes=[dict(text='The phase is not established.',evidence_ids=['E1'])]
        output=dict(claims=[dict(answer='Supported fact.',evidence_ids=['E1'],part_ids=['P1'],quotes=[])],
                    unanswered_part_ids=[],answer_notes=notes)
        packets=dict(parts=[dict(part_id='P1',query_scope=dict(phase='setup'),direct_source_ids=['E1'])],
                     source_scopes=[dict(id='S1',scope=dict(version_id=5),source_ids=['E1'])])
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(output)) as chat:
            result=_request_openai_compatible('http://unused','contract_qa',dict(query='Fact?',inline_review=True,
                answer_format='cited_claims',check_completeness=True,source_packets=packets,
                evidence=[dict(evidence_id='E1',text='Supported fact.',version_id=5,
                    expansion_reason='x'*400,retrieval_hints=[dict(part=1,rank=1)])],
                answer_parts=[dict(id='P1',question='Requested detail?',source_ids=['E1'])]))
        self.assertEqual(chat.call_count,1)
        self.assertEqual(result['answer_notes'],notes)
        schema=chat.call_args.kwargs['json_schema']
        self.assertIn('answer_notes',schema['required'])
        self.assertIn('answer',schema['properties']['claims']['items']['required'])
        self.assertNotIn('explanation',schema['properties']['claims']['items']['properties'])
        prompt=chat.call_args.args[1][1]['content']
        self.assertNotIn('x'*400,prompt)
        self.assertNotIn('retrieval_hints',prompt)
        self.assertEqual(prompt.count('Requested detail?'),1)
        self.assertEqual(prompt.count('Supported fact.'),1)
        self.assertIn('SOURCE PACKETS',prompt)

    def test_repeated_question_does_not_replace_or_prefix_the_actual_explanation(self):
        output=dict(insufficient_evidence=False,claims=[dict(text='Where does the app run?',
            explanation='The application runs on a PC.',evidence_ids=['E1'],quotes=[])])
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(output)):
            result=_request_openai_compatible('http://unused','contract_qa',dict(query='Where does the app run?',
                evidence=[dict(evidence_id='E1',text='The application runs on a PC.')],
                answer_format='cited_claims',check_completeness=True,answer_detail='explanatory'))
        self.assertNotIn('Where does the app run?',result['answer'])
        self.assertEqual(result['citation_claims'][0]['text'],'The application runs on a PC.')

    def test_advisory_review_reads_uncited_counterevidence_without_changing_claim_binding(self):
        sources=[dict(evidence_id='E1',text='No PC required.'),
                 dict(evidence_id='E2',text='The setup application runs on a PC.')]
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(dict(
                supported=[False],missing_parts=[],review_notes=[]))) as chat:
            result=_request_openai_compatible('http://unused','verify_answer',dict(
                query='PC?',evidence=sources,review_retrieved_context=True,
                citation_claims=[dict(text='Setup requires no PC.',evidence_ids=['E1'])]))
        sent=chat.call_args.args[1][1]['content']
        self.assertIn('The setup application runs on a PC.',sent)
        self.assertEqual(result['verification_input']['evidence_ids'],['E1','E2'])
        self.assertIn('"source_ids":["E1"]',sent)
        self.assertEqual(chat.call_count,1)

    def test_advisory_review_preserves_specific_model_concerns(self):
        note='Nguồn E1 nói về giá, không chứng minh mức phạt.'
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(dict(
                supported=[False],missing_parts=[],review_notes=[dict(check_id='0',reason=note)]))) as chat:
            result=_request_openai_compatible('http://unused','verify_answer',dict(
                query='Mức phạt?',evidence=[dict(evidence_id='E1',text='Price is 120 VND.')],
                citation_claims=[dict(text='Penalty is 120 VND.',evidence_ids=['E1'])]))
        self.assertEqual(chat.call_count,1)
        self.assertFalse(result['supported'])
        self.assertEqual(result['claim_checks'][0]['reason'],note)
        self.assertIn('review_notes',chat.call_args.kwargs['json_schema']['required'])

    def test_part_schema_and_source_first_prompt_preserve_graph(self):
        parts=[dict(id='P1',question='First?'),dict(id='P2',question='Second?')]
        output=dict(insufficient_evidence=False,claims=[dict(text='Supported fact.',evidence_ids=['E1'],
            quotes=[],part_ids=['P1'])],unanswered_part_ids=['P2'])
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(output)) as chat:
            result=_request_openai_compatible('http://unused','contract_qa',dict(query='Both?',
                evidence=[dict(evidence_id='E1',text='Supported fact.')],check_completeness=True,
                answer_format='cited_claims',answer_parts=parts,
                graph_context=dict(nodes=[],edges=[],partial=False)))
        schema=chat.call_args.kwargs['json_schema']
        self.assertNotIn('insufficient_evidence',schema['properties'])
        self.assertEqual(schema['properties']['claims']['items']['properties']['part_ids']['items']['enum'],['P1','P2'])
        content=chat.call_args.args[1][-1]['content']
        self.assertLess(content.index('SOURCE PASSAGES'),content.index('NEO4J STRUCTURE'))
        self.assertLess(content.index('Supported fact.'),content.index('SOURCE METADATA'))
        self.assertEqual(result['answer_part_coverage'][1]['status'],'unavailable')

    def test_repeated_question_does_not_discard_other_supported_statements(self):
        sources=[dict(evidence_id='E1',text='A source fact.')]
        good=dict(text='A source fact.',evidence_ids=['E1'])
        data=dict(claims=[good,dict(text='What is the condition?',evidence_ids=['E1'])])
        self.assertIn('A source fact',render_cited_claims(data,sources))
        self.assertEqual(data['claims'],[good])
        with self.assertRaises(QwenGatewayError):
            render_cited_claims(dict(claims=[dict(text='What is the condition?',evidence_ids=['E1'])]),sources)

    def test_optional_quote_errors_do_not_discard_supported_prose(self):
        source=dict(evidence_id='E1',text='Compatible robots\nHSR Series/HS-A1 Series')
        claim=dict(text='Supported paraphrase.',evidence_ids=['E1'],quotes=[
            dict(evidence_id='E1',quote='Compatible robots HSR Series/HS-A1 Series'),
            dict(evidence_id='E1',quote='Invented quotation'),
            dict(evidence_id='E2',quote='Wrong source')])
        self.assertTrue(render_cited_claims(dict(claims=[claim]),[source]))
        self.assertEqual(claim['quotes'],[dict(evidence_id='E1',quote=source['text'])])
        self.assertEqual(claim['text'],'Supported paraphrase.')

    def test_insufficient_evidence_is_reconsidered_once_with_same_sources(self):
        from NewIdea.services.knowledge_answer_service import generate_checked_answer
        evidence=[dict(evidence_id='E1',text='The source states a supported fact.')]
        seen=[]
        def model(task,payload):
            seen.append((task,payload))
            if len(seen)==1:
                return dict(answer='No evidence.',insufficient_evidence=True)
            if task=='contract_qa':
                return dict(answer='A supported fact [E1].',citation_claims=[
                    dict(text='A supported fact.',evidence_ids=['E1'])])
            return dict(supported=True,completeness=dict(complete=True,missing_parts=[]))
        result=generate_checked_answer(dict(query='What does the source state?',evidence=evidence),request_model=model)
        self.assertEqual(result['generation_status'],'generated')
        self.assertEqual([t for t,_ in seen],['contract_qa','contract_qa'])
        self.assertEqual(seen[1][1]['evidence'],evidence)
        self.assertIn('Re-examine',seen[1][1]['repair_feedback'])

    def test_natural_answer_schema_keeps_direct_quotes_and_internal_source_bindings_separate(self):
        schema=cited_claim_schema([dict(evidence_id='E1',text='Source')],natural_prose=True)
        claim=schema['properties']['claims']['items']
        self.assertIn('quotes',claim['required'])
        self.assertEqual(claim['properties']['quotes']['items']['properties']['evidence_id']['enum'],['E1'])
    def test_verifier_cannot_borrow_sources_outside_selected_claim_support(self):
        for selected in ([],['unknown']):
            with self.subTest(selected=selected), patch(
                    'ai_extract.services.contract_pipeline_service._chat') as chat, self.assertRaises(QwenGatewayError):
                _request_openai_compatible('http://unused','verify_answer',dict(
                    query='Property A?',evidence=[dict(evidence_id='E1',text='Property A has value X.'),
                                                 dict(evidence_id='E2',text='Property B has value Y.')],
                    answer='Property A has value X [E1].',
                    citation_claims=[dict(text='Property A has value X.',evidence_ids=selected)]))
            chat.assert_not_called()

    def test_translated_claim_uses_original_source_binding_without_retranscription(self):
        with patch('ai_extract.services.contract_pipeline_service._chat',
                   return_value=json.dumps(dict(supported=[True],missing_parts=[]))) as chat:
            result=_request_openai_compatible('http://unused','verify_answer',dict(
                query='Thu?c t?nh A?',evidence=[dict(evidence_id='E1',text='Property A has value X.')],
                answer='Thu?c t?nh A c? gi? tr? X [E1].',
                citation_claims=[dict(text='Thu?c t?nh A c? gi? tr? X.',evidence_ids=['E1'])]))
        self.assertTrue(result['supported'])
        self.assertEqual(chat.call_count,1)
        self.assertTrue(result['claim_checks'][0]['supported'])
        self.assertIn('Property A has value X.',chat.call_args.args[1][1]['content'])

    @override_settings(A3_HYBRID_ENABLED=True)
    def test_subquestion_keeps_source_that_full_question_ranks_low(self):
        from ai_extract.services.contract_pipeline_service import retrieve_hybrid
        sources=[dict(chunk_id=k,text=k,expansion_reason='query_match') for k in ('main','detail')]
        backend=Mock()
        backend.rerank.side_effect=lambda query,items:[dict(item,neural_logit=(
            2 if item['chunk_id']==('detail' if query=='Specific detail?' else 'main') else -5)) for item in items]
        with patch('ai_extract.services.contract_pipeline_service.retrieve_hybrid_candidates',
                   return_value=(sources,{})), patch(
                'ai_extract.services.contract_pipeline_service.expand_retrieval_context',
                side_effect=lambda query,tree,version,seeds,coverage,**kwargs:(seeds,coverage)):
            selected,coverage=retrieve_hybrid('Full question',{},None,backend,
                                             queries=['Specific detail?','Specific detail?'])
        self.assertEqual({s['chunk_id'] for s in selected},{'main','detail'})
        self.assertEqual(backend.rerank.call_count,2)
        self.assertEqual(len(coverage['retrieval_questions']),2)

    def test_normal_a2_delivers_sources_and_graph_without_self_generated_explanations(self):
        source = dict(evidence_id='E1', text='Guidance runs on a PC.')
        graph = dict(nodes=[dict(id='a'),dict(id='b')],edges=[dict(from_id='a',to_id='b',type='HAS_EVIDENCE')])
        draft = dict(insufficient_evidence=False, claims=[dict(text='Guidance chạy trên PC.', evidence_ids=['E1'])])
        with patch('ai_extract.services.contract_pipeline_service._chat', return_value=json.dumps(draft)) as chat:
            result = _request_openai_compatible('http://unused', 'contract_qa', dict(
                query='Guidance chạy ở đâu?', evidence=[source], graph_context=graph,
                check_completeness=True, answer_format='cited_claims'))
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(result['generation_input']['graph_context_snapshot'], graph)
        self.assertNotIn('query_normalization', result['generation_input'])
        self.assertEqual(result['citation_claims'], draft['claims'])
        self.assertEqual(result['citation_analysis'], [])
        self.assertNotIn('Trích dẫn:', result['answer'])
        self.assertIn(source['text'], chat.call_args.args[1][1]['content'])
        self.assertIn('HAS_EVIDENCE', chat.call_args.args[1][1]['content'])
        self.assertNotIn('Every factual sentence MUST', chat.call_args.args[1][0]['content'])

    @override_settings(QWEN_ANSWER_MAX_TOKENS=8192)
    def test_concise_answer_limits_output_without_removing_sources_or_graph(self):
        graph=dict(nodes=[dict(id='n1')],edges=[],relation_legend={})
        draft=dict(insufficient_evidence=False,claims=[dict(text='It runs on a PC.',evidence_ids=['E1'])])
        source=dict(evidence_id='E1',text='Guidance runs on a PC.')
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(draft)) as chat:
            result=_request_openai_compatible('http://unused','contract_qa',dict(query='Where?',
                evidence=[source],graph_context=graph,answer_detail='concise',
                answer_format='cited_claims',check_completeness=True))
        self.assertEqual(chat.call_args.kwargs['max_tokens'],2048)
        self.assertEqual(result['generation_input']['requested_output_tokens'],2048)
        self.assertEqual(result['generation_input']['graph_context_snapshot'],graph)
        self.assertIn(source['text'],chat.call_args.args[1][1]['content'])
        self.assertEqual(result['generation_input']['answer_detail'],'concise')

    @override_settings(QWEN_ANSWER_MAX_TOKENS=8192)
    def test_normal_explanatory_answer_keeps_mechanism_instruction_after_prompt_rebuild(self):
        draft=dict(insufficient_evidence=False,claims=[dict(text='It uses one controller.',
            explanation='The cited passage does not explain its internal mechanism.',evidence_ids=['E1'])])
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(draft)) as chat:
            result = _request_openai_compatible('http://unused','contract_qa',dict(query='Explain why.',
                evidence=[dict(evidence_id='E1',text='It uses one controller.')],answer_detail='explanatory',
                answer_format='cited_claims',check_completeness=True))
        self.assertEqual(chat.call_args.kwargs['max_tokens'],8192)
        self.assertIn('developed answer',chat.call_args.args[1][0]['content'])
        self.assertIn('actual causal connection',chat.call_args.args[1][0]['content'])
        self.assertIn('explanation', chat.call_args.kwargs['json_schema']['properties']['claims']['items']['required'])
        self.assertIn('does not explain its internal mechanism', result['answer'])
        self.assertIn('does not explain its internal mechanism', result['citation_claims'][0]['text'])
        self.assertEqual(result['citation_claims'][0]['evidence_ids'], ['E1'])

    def test_semantic_repair_receives_the_rejected_answer_not_only_its_claim_index(self):
        draft=dict(insufficient_evidence=False,claims=[dict(text='Corrected claim.',evidence_ids=['E1'])])
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(draft)) as chat:
            _request_openai_compatible('http://unused','contract_qa',dict(query='Explain.',
                evidence=[dict(evidence_id='E1',text='Original evidence.')],answer_detail='explanatory',
                answer_format='cited_claims',check_completeness=True,
                repair_feedback='Claim 0 is unsupported.',previous_answer='Rejected causal claim [E1].'))
        self.assertIn('Rejected causal claim [E1].',chat.call_args.args[1][1]['content'])

    def test_normal_generation_does_not_repeat_question_or_schema_instructions(self):
        query='Where does Guidance run?'
        draft=dict(insufficient_evidence=False,claims=[dict(text='It runs on a PC.',evidence_ids=['E1'])])
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(draft)) as chat:
            _request_openai_compatible('http://unused','contract_qa',dict(query=query,
                evidence=[dict(evidence_id='E1',text='Guidance runs on a PC.')],
                answer_format='cited_claims',check_completeness=True))
        messages=chat.call_args.args[1]
        self.assertEqual(len(messages),2)
        self.assertEqual(sum(message['content'].count(query) for message in messages),1)
        self.assertEqual(sum(message['content'].count('Return ONLY JSON') for message in messages),1)

    def test_requested_parts_and_duplicate_source_text_are_sent_once(self):
        from ai_extract.services.contract_pipeline_service import _prompt_sources
        text='A complete passage with its original qualification.'
        wire=_prompt_sources([dict(evidence_id='E1',text=text,applicability='Version A'),
                              dict(evidence_id='E2',text=text,applicability='Version B')])
        self.assertEqual(wire.count(text),1)
        self.assertIn('Same source text as [E1]',wire)
        self.assertIn('Version A',wire)
        self.assertIn('Version B',wire)
        draft=dict(insufficient_evidence=False,claims=[dict(text='An answer.',evidence_ids=['E1'])])
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(draft)) as chat:
            _request_openai_compatible('http://unused','contract_qa',dict(query='A question?',
                evidence=[dict(evidence_id='E1',text=text)],coverage=dict(requested_parts=['A requested detail?']),
                answer_format='cited_claims',check_completeness=True))
        prompt=chat.call_args.args[1][1]['content']
        self.assertEqual(prompt.count('A requested detail?'),1)
        self.assertEqual(prompt.count(text),1)

    def test_combined_verifier_rejects_incomplete_or_non_boolean_verdicts(self):
        for flags in ([],[True,False],['true'],[1]):
            with self.subTest(flags=flags), patch('ai_extract.services.contract_pipeline_service._chat',
                    return_value=json.dumps(dict(supported=flags,missing_parts=[]))), self.assertRaises(QwenGatewayError):
                _request_openai_compatible('http://unused','verify_answer',dict(query='Question?',
                    evidence=[dict(evidence_id='E1',text='Source text.')],
                    citation_claims=[dict(text='A claim.',evidence_ids=['E1'])]))

    def test_claim_verification_keeps_selected_support_without_resending_uncited_passages(self):
        sources = [dict(evidence_id='E1', text='Shape search', heading_path=['Search functions']),
                   dict(evidence_id='E2', text='Detect whether a part is within the area.', heading_path=['Presence detection'])]
        graph = dict(nodes=[dict(id='section')],edges=[],relation_legend={})
        outputs = [dict(supported=[False],missing_parts=[])]
        with patch('ai_extract.services.contract_pipeline_service._chat', side_effect=[json.dumps(x) for x in outputs]) as chat:
            result = _request_openai_compatible('http://unused', 'verify_answer', dict(evidence=sources,
                query='What does presence detection do?',graph_context=graph,
                answer='Kiểm tra hiện diện tìm hình dạng [E1].', citation_claims=[dict(
                    text='Kiểm tra hiện diện tìm hình dạng.', evidence_ids=['E1'])]))
        self.assertFalse(result['supported'])
        content=chat.call_args.args[1][1]['content']
        sent=json.loads(content.split('CONTEXT:\n',1)[1].split('\nSOURCE PASSAGES:\n',1)[0])
        context=sent
        self.assertEqual(sent['question'],'What does presence detection do?')
        from NewIdea.services.knowledge_answer_service import model_graph_context
        self.assertEqual(context['graph_context'],model_graph_context(graph))
        self.assertEqual(sent['checks'][0]['source_ids'],['E1'])
        self.assertIn('"heading_path":["Search functions"]',content)
        self.assertIn(sources[0]['text'],content)
        self.assertNotIn(sources[1]['text'],content)
        self.assertEqual(result['claim_checks'][0]['supported'],False)
        self.assertEqual(result['verification_input']['graph_context_snapshot'],graph)

    def test_support_and_completeness_share_one_call_without_repeated_answer(self):
        sources=[dict(evidence_id='E1',text='The result applies only within the area.')]
        graph=dict(nodes=[dict(id='a')],edges=[],relation_legend={})
        responses=[dict(supported=[True],missing_parts=['The area qualification is missing.'])]
        with patch('ai_extract.services.contract_pipeline_service._chat',side_effect=[json.dumps(x) for x in responses]) as chat:
            result=_request_openai_compatible('http://unused','verify_answer',dict(
                query='What does it do?',evidence=sources,graph_context=graph,
                answer='It detects a result.',check_completeness=True,
                citation_claims=[dict(text='It detects a result.',evidence_ids=['E1'])]))
        support=chat.call_args_list[0].args[1][1]['content']
        context=json.loads(support.split('CONTEXT:\n',1)[1].split('\nSOURCE PASSAGES:\n',1)[0])
        from NewIdea.services.knowledge_answer_service import model_graph_context
        self.assertEqual(context['graph_context'],model_graph_context(graph))
        self.assertIn(sources[0]['text'],support)
        self.assertEqual(chat.call_count,1)
        self.assertEqual(support.count('It detects a result.'),1)
        self.assertFalse(result['supported'])
        self.assertFalse(result['completeness']['complete'])

    def test_advisory_completeness_does_not_reject_supported_answer(self):
        responses = [dict(supported=[True],missing_parts=['A requested detail.'])]
        with patch('ai_extract.services.contract_pipeline_service._chat',side_effect=[json.dumps(x) for x in responses]):
            result = _request_openai_compatible('http://unused','verify_answer',dict(
                query='Explain the function.',evidence=[dict(evidence_id='E1',text='Source text.')],
                answer='The function [E1].',check_completeness=True,completeness_policy='advisory',
                citation_claims=[dict(text='The function.',evidence_ids=['E1'])]))
        self.assertTrue(result['supported'])
        self.assertFalse(result['completeness']['complete'])

    def test_multi_part_retrieval_does_not_discard_relevant_seed_after_eighth(self):
        from ai_extract.services.contract_pipeline_service import expand_retrieval_context
        leaves=[dict(chunk_id=str(i),text='Source '+str(i),heading_path=[],ancestor_ids=[],node_id='root') for i in range(12)]
        with patch('ai_extract.services.contract_pipeline_service.layout_repository.latest_run',return_value=None):
            selected,_=expand_retrieval_context('Multiple questions',dict(leaves=leaves,nodes=[]),None,leaves,{})
        self.assertEqual({s['chunk_id'] for s in selected},{str(i) for i in range(12)})

    def test_readable_sources_preserve_text_ids_and_shared_provenance(self):
        from ai_extract.services.contract_pipeline_service import _prompt_sources
        sources=[dict(evidence_id='E1',text='First source.',heading_path=['A'],source_location='Page 1'),
                 dict(evidence_id='E2',text='Second source with its condition.',heading_path=['A'],source_location='Page 1')]
        wire=_prompt_sources(sources)
        self.assertIn('Metadata for [E1], [E2]:',wire)
        self.assertIn('"source_location":"Page 1"',wire)
        for source in sources:
            self.assertIn('['+source['evidence_id']+']\n'+source['text'],wire)
            self.assertEqual(wire.count(source['text']),1)

    def test_shared_metadata_keeps_each_sources_distinct_applicability_and_query_rank(self):
        import re
        from ai_extract.services.contract_pipeline_service import _prompt_sources
        sources=[dict(evidence_id=f'E{i}',text=f'Passage {i}.',
            heading_path=['A long multilingual heading repeated throughout this section '*4],
            applicability='Only after inspection' if i%2 else 'Only before inspection',
            retrieval_hints=[dict(part=1,rank=i)]) for i in range(1,13)]
        wire=_prompt_sources(sources)
        metadata={s['evidence_id']:{} for s in sources}
        for identifiers,fields in re.findall(r'Metadata for ([^\n]+):\n([^\n]+)',wire):
            for identity in re.findall(r'\[([^\]]+)\]',identifiers):metadata[identity].update(json.loads(fields))
        for source in sources:
            self.assertEqual(metadata[source['evidence_id']],{k:v for k,v in source.items() if k not in {'text','evidence_id'}})
            self.assertIn(source['text'],wire)
        self.assertLess(wire.count(sources[0]['heading_path'][0]),len(sources))

    @override_settings(RAG_INFERENCE_URL='http://local')
    def test_expanded_reranking_batches_without_losing_source_score_binding(self):
        from ai_extract.services.contract_pipeline_service import HybridBackend
        sources=[dict(chunk_id=str(i),text='Source '+str(i)) for i in range(35)]
        responses=[]
        for scores in [list(range(30)),list(range(30,35))]:
            response=Mock();response.json.return_value=dict(model='BAAI/bge-reranker-v2-m3',scores=scores)
            responses.append(response)
        with patch('ai_extract.services.contract_pipeline_service.requests.post',side_effect=responses) as post:
            ranked=HybridBackend().rerank('question',sources)
        self.assertEqual([len(c.kwargs['json']['passages']) for c in post.call_args_list],[30,5])
        self.assertEqual(ranked[0]['chunk_id'],'34')
        self.assertEqual(len(ranked),35)

    def test_advisory_completeness_still_rejects_unsupported_claim(self):
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(
                dict(supported=[False],missing_parts=[]))):
            result = _request_openai_compatible('http://unused','verify_answer',dict(
                query='Explain the function.',evidence=[dict(evidence_id='E1',text='Source text.')],
                answer='A different function [E1].',check_completeness=True,completeness_policy='advisory',
                citation_claims=[dict(text='A different function.',evidence_ids=['E1'])]))
        self.assertFalse(result['supported'])

    def test_compact_verdict_keeps_exact_rejected_claim_identity(self):
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(
                dict(supported=[True,False,True],missing_parts=[]))):
            result=_request_openai_compatible('http://unused','verify_answer',dict(
                query='Question?',evidence=[dict(evidence_id='E1',text='Source text.')],
                citation_claims=[dict(text=text,evidence_ids=['E1']) for text in ['First.','Second.','Third.']]))
        self.assertFalse(result['supported'])
        self.assertEqual([c['supported'] for c in result['claim_checks']],[True,False,True])
        self.assertIn('Claim 1',result['reason'])
        self.assertNotIn('Claim 0',result['reason'])

    def test_claim_verification_receives_the_delivered_source_image(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            Image.new('RGB',(12,12),'white').save(Path(directory)/'page.png')
            source=dict(evidence_id='E1',text='Source caption',image_path='page.png')
            response=dict(supported=[False],missing_parts=[])
            with override_settings(MEDIA_ROOT=directory), patch(
                    'ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(response)) as chat:
                result=_request_openai_compatible('http://unused','verify_answer',dict(
                    query='Describe the figure',evidence=[source],use_images=True,
                    citation_claims=[dict(text='An unsupported description',evidence_ids=['E1'])]))
            content=chat.call_args.args[1][1]['content']
            self.assertIsInstance(content,list)
            self.assertEqual(content[0]['type'],'text')
            checks=json.loads(content[0]['text'].split('CONTEXT:\n',1)[1].split('\nSOURCE PASSAGES:\n',1)[0])
            self.assertEqual(checks['checks'][0]['source_ids'],['E1'])
            self.assertIn('[E1]',content[0]['text'])
            self.assertTrue(any(part['type']=='image_url' for part in content))
            self.assertEqual(result['verification_input']['images'],[dict(evidence_id='E1',image_path='page.png')])

    def test_structural_expansion_preserves_real_neural_relevance(self):
        leaf = dict(chunk_id='a', text='Source text', heading_path=[], ancestor_ids=[], node_id='root')
        selected, _ = structural_retrieve('question', dict(leaves=[leaf], nodes=[]),
            seeds=[dict(leaf, neural_logit=4.2)])
        self.assertEqual(selected[0]['neural_logit'], 4.2)

    def test_factually_supported_first_part_is_rejected_when_other_parts_are_missing(self):
        outputs = [dict(supported=True, reason='Nguồn hỗ trợ câu trả lời.'),
                   dict(complete=False, missing_parts=['Thiếu vai trò và yêu cầu thiết bị của Runtime.'])]
        with patch('ai_extract.services.contract_pipeline_service._chat', side_effect=[json.dumps(x) for x in outputs]):
            result = _request_openai_compatible('http://unused', 'verify_answer', dict(
                query='Guidance và Runtime làm gì?', evidence=[], answer='Guidance chạy trên PC.',
                check_completeness=True))
        self.assertFalse(result['supported'])
        self.assertFalse(result['completeness']['complete'])
        self.assertIn('Runtime', result['reason'])

    def test_citation_integrity_does_not_pretend_to_verify_numeric_meaning(self):
        sources=[dict(evidence_id='E1',text='The product has one camera.')]
        self.assertTrue(validate_answer('There are 2 cameras [E1].',sources)['valid'])
        self.assertFalse(validate_answer('There are 2 cameras [E99].',sources)['valid'])
        # Semantic truth is checked by the model, not by matching digits.

    def test_isolated_verifier_accepts_reason_in_any_language(self):
        analysis=[dict(evidence_id='E1',quote='Only a product label.',explanation='Không cần PC.')]
        outputs=[dict(supported=True,reason='Nguồn hỗ trợ.'),
                 dict(checks=[dict(check_id='0',supported=False,reason='Lý do 否')])]
        with patch('ai_extract.services.contract_pipeline_service._chat',side_effect=[json.dumps(x) for x in outputs]) as chat:
            result=_request_openai_compatible('http://unused','verify_answer',
                dict(evidence=[],analyze_citations=True,citation_analysis=analysis))
        self.assertFalse(result['supported'])
        self.assertEqual(chat.call_count,2)
        self.assertEqual(json.loads(chat.call_args.args[1][1]['content'])[0]['quote'],analysis[0]['quote'])

    def test_isolated_analysis_verification_cannot_borrow_another_sources_facts(self):
        analysis = [dict(evidence_id='E1', quote='EVP2 consists of Guidance and Runtime.', explanation='Không cần PC')]
        outputs = [dict(supported=True, reason='Nguồn khác hỗ trợ nội dung.'),
                   dict(checks=[dict(check_id='0', supported=False, reason='Nguồn này không nêu yêu cầu PC.')])]
        with patch('ai_extract.services.contract_pipeline_service._chat', side_effect=[json.dumps(x) for x in outputs]) as chat:
            result = _request_openai_compatible('http://unused', 'verify_answer',
                dict(evidence=[], analyze_citations=True, citation_analysis=analysis))
        self.assertFalse(result['supported'])
        self.assertIn('Nguồn E1', result['reason'])
        isolated_inputs = json.loads(chat.call_args.args[1][1]['content'])
        self.assertEqual(isolated_inputs[0]['quote'], analysis[0]['quote'])
        self.assertNotIn('evidence', isolated_inputs[0])

    def test_answer_language_is_not_validated(self):
        sources = [dict(evidence_id='E1', text='Guidance runs on a PC.')]
        output = dict(claims=[dict(text='Guidance runs on a PC.', evidence_ids=['E1'])],
                      citation_analysis=[dict(evidence_id='E1', quote=sources[0]['text'], explanation='PC is required.')])
        with patch('ai_extract.services.contract_pipeline_service._chat', return_value=json.dumps(output)) as chat:
            result = _request_openai_compatible('http://unused', 'contract_qa',
                dict(query='Guidance', evidence=sources, answer_format='cited_claims', analyze_citations=True))
        self.assertEqual(chat.call_count, 1)
        self.assertIn('Guidance runs on a PC', result['answer'])

    def test_missing_citation_analysis_is_never_accepted(self):
        output = dict(claims=[dict(text='Guidance runs on a PC.', evidence_ids=['E1'])], citation_analysis=[])
        with patch('ai_extract.services.contract_pipeline_service._chat', return_value=json.dumps(output)) as chat:
            with self.assertRaises(QwenGatewayError):
                _request_openai_compatible('http://unused', 'contract_qa',
                    dict(query='Guidance', evidence=[dict(evidence_id='E1', text='Guidance runs on a PC.')],
                         answer_format='cited_claims', analyze_citations=True))
        self.assertEqual(chat.call_count, 2)

    def test_verification_accepts_english_reason(self):
        output = dict(supported=True, reason='The source supports the claim.')
        with patch('ai_extract.services.contract_pipeline_service._chat', return_value=json.dumps(output)) as chat:
            result = _request_openai_compatible('http://unused', 'verify_answer', dict(evidence=[], analyze_citations=True))
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(result['reason'], output['reason'])

    def test_vietnamese_citation_analysis_requires_verbatim_supported_quotes(self):
        sources = [dict(evidence_id='E1', text='Guidance runs on a PC.')]
        output = dict(claims=[dict(text='Guidance chạy trên PC.', evidence_ids=['E1'])],
            citation_analysis=[dict(evidence_id='E1', quote='Guidance runs on a PC.',
                explanation='Nguồn xác nhận PC được dùng để chạy Guidance')])
        with patch('ai_extract.services.contract_pipeline_service._chat', return_value=json.dumps(output)) as chat:
            result = _request_openai_compatible('http://unused', 'contract_qa',
                dict(query='Guidance', evidence=sources, answer_format='cited_claims', analyze_citations=True))
        self.assertIn('Trích dẫn:\nNguồn xác nhận PC được dùng để chạy Guidance [E1].', result['answer'])
        self.assertNotIn('Guidance runs on a PC.', result['answer'])
        self.assertTrue(validate_answer(result['answer'], sources)['valid'])
        self.assertIn('citation_analysis', chat.call_args.kwargs['json_schema']['required'])
        option = chat.call_args.kwargs['json_schema']['properties']['citation_analysis']['items']['anyOf'][0]
        self.assertEqual(option['properties']['evidence_id']['enum'], ['E1'])
        self.assertEqual(option['properties']['quote']['enum'], [sources[0]['text']])
        self.assertIn('Viết câu trả lời và phân tích bằng tiếng Việt', chat.call_args.args[1][0]['content'])
        for quote in ('Runtime needs a PC.', ''):
            output['citation_analysis'][0]['quote'] = quote
            with self.assertRaises(QwenGatewayError):
                render_cited_claims(output, sources, require_analysis=True)

    def test_analysis_must_cover_every_cited_source_and_not_add_uncited_sources(self):
        sources = [dict(evidence_id='E1', text='Guidance runs on a PC.'),
                   dict(evidence_id='E2', text='Runtime uses a camera.')]
        output = dict(claims=[dict(text='Guidance chạy trên PC.', evidence_ids=['E1'])],
            citation_analysis=[dict(evidence_id='E2', quote='Runtime uses a camera.', explanation='Runtime dùng camera')])
        with self.assertRaises(QwenGatewayError):
            render_cited_claims(output, sources, require_analysis=True)
        output['claims'][0]['evidence_ids'] = ['E1', 'E2']
        with self.assertRaisesMessage(QwenGatewayError, 'Every cited source requires analysis'):
            render_cited_claims(output, sources, require_analysis=True)

    def test_verifier_reason_is_requested_in_vietnamese(self):
        with patch('ai_extract.services.contract_pipeline_service._chat',
                   return_value=json.dumps(dict(supported=True, reason='Nguồn xác nhận nội dung.'))) as chat:
            _request_openai_compatible('http://unused', 'verify_answer', dict(evidence=[], answer='Nội dung'))
        self.assertIn('reason phải viết bằng tiếng Việt', chat.call_args.args[1][0]['content'])

    def test_context_limit_retries_with_smaller_output_and_preserves_sources(self):
        from unittest.mock import Mock
        error=Mock(status_code=400,text='maximum context length is 4096 tokens; prompt contains at least 3097 input tokens')
        success=Mock(status_code=200)
        success.json.return_value={'choices':[dict(finish_reason='stop',message=dict(content='complete'))]}
        requests=[]
        def post(url,**kwargs):
            requests.append(dict(kwargs['json']))
            return error if len(requests)==1 else success
        messages=[dict(role='user',content='Source text and source-bound Neo4j graph')]
        with patch('ai_extract.services.contract_pipeline_service.requests.post',side_effect=post):
            self.assertEqual(_chat('http://unused',messages,max_tokens=8192),'complete')
        self.assertEqual([r['max_tokens'] for r in requests],[8192,871])
        self.assertEqual(requests[0]['messages'],messages)
        self.assertEqual(requests[1]['messages'][:-1],messages)
        self.assertIn('fewer than 871',requests[1]['messages'][-1]['content'])

    def test_context_retry_handles_incremental_prompt_lower_bound(self):
        calls=[]
        messages=[dict(role='user',content='Unchanged source text and graph')]
        def post(url, **kwargs):
            body=kwargs['json'];calls.append(dict(body))
            if body['max_tokens']>2048:
                lower_bound=16384-body['max_tokens']+1
                response=Mock(status_code=400,text=(
                    f'maximum context length is 16384 tokens; prompt contains at least {lower_bound} input tokens'))
                response.json.return_value={'error':{'message':response.text}}
                return response
            response=Mock(status_code=200)
            response.json.return_value={'choices':[dict(finish_reason='stop',message=dict(content='complete'))]}
            return response
        with patch('ai_extract.services.contract_pipeline_service.requests.post',side_effect=post):
            self.assertEqual(_chat('http://unused',messages,max_tokens=8192),'complete')
        self.assertEqual([c['max_tokens'] for c in calls],[8192,4096,2048])
        self.assertEqual(calls[0]['messages'],messages)
        self.assertTrue(all(c['messages'][:-1]==messages for c in calls[1:]))
        self.assertIn('fewer than 2048',calls[-1]['messages'][-1]['content'])

    def test_truncated_answer_is_regenerated_concisely_without_partial_text(self):
        partial=Mock(status_code=200)
        partial.json.return_value={'choices':[dict(finish_reason='length',message=dict(content='unfinished answer'))]}
        complete=Mock(status_code=200)
        complete.json.return_value={'choices':[dict(finish_reason='stop',message=dict(content='full concise answer'))]}
        messages=[dict(role='user',content='sources and graph')]
        with patch('ai_extract.services.contract_pipeline_service.requests.post',side_effect=[partial,complete]) as post:
            self.assertEqual(_chat('http://unused',messages),'full concise answer')
        retried=post.call_args.kwargs['json']
        self.assertLess(retried['max_tokens'],2048)
        self.assertEqual(retried['messages'][:-1],messages)
        self.assertIn('Cover every requested part',retried['messages'][-1]['content'])
        self.assertNotIn('unfinished answer',str(retried))

    def test_gateway_rejects_incomplete_answer_instead_of_showing_it_as_complete(self):
        from unittest.mock import Mock
        response=Mock(status_code=200)
        response.json.return_value={'choices':[dict(finish_reason='length',message=dict(content='partial'))]}
        with patch('ai_extract.services.contract_pipeline_service.requests.post',return_value=response) as post:
            with self.assertRaisesMessage(QwenGatewayError,'before completing'):
                _chat('http://unused',[dict(role='user',content='question')])
        self.assertEqual(post.call_count,4)

    def test_output_budget_larger_than_server_is_a_context_diagnostic(self):
        from unittest.mock import Mock
        error=Mock(status_code=400,text='max_tokens=8192 cannot be greater than max_model_len=4096.')
        with patch('ai_extract.services.contract_pipeline_service.requests.post',return_value=error):
            with self.assertRaises(QwenContextLimitError) as raised:
                _chat('http://unused',[dict(role='user',content='question')],max_tokens=8192)
        self.assertEqual(raised.exception.capacity,4096)
        self.assertIsNone(raised.exception.prompt_tokens)

    @override_settings(QWEN_ANSWER_MAX_TOKENS=8192,QWEN_ANSWER_MAX_CLAIMS=96)
    def test_long_cited_answer_keeps_every_claim_and_uses_larger_budget(self):
        facts=[f'Fact {i} is documented.' for i in range(12)]
        sources=[dict(evidence_id='E1',text=' '.join(facts))]
        output=dict(claims=[dict(text=fact,evidence_ids=['E1']) for fact in facts])
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(output)) as chat:
            result=_request_openai_compatible('http://unused','contract_qa',
                dict(query='Guidance',evidence=sources,answer_format='cited_claims'))
        self.assertEqual(len(result['answer'].splitlines()),12)
        self.assertEqual(chat.call_args.kwargs['max_tokens'],8192)
        self.assertEqual(cited_claim_schema(sources)['properties']['claims']['maxItems'],96)

    def test_qwen_selects_sources_for_each_sentence(self):
        sources = [dict(evidence_id='E1', text='Guidance runs on a PC.'),
                   dict(evidence_id='E2', text='Runtime uses the controller and camera.')]
        output = dict(claims=[dict(text='Guidance chạy trên PC.', evidence_ids=['E1']),
                              dict(text='Runtime dùng bộ điều khiển và camera.', evidence_ids=['E2'])])
        with patch('ai_extract.services.contract_pipeline_service._chat',
                   return_value=json.dumps(output)) as chat:
            result = _request_openai_compatible('http://unused', 'contract_qa',
                dict(query='Guidance và Runtime khác nhau thế nào?', evidence=sources,
                     answer_format='cited_claims'))
        self.assertEqual(result['answer'],
            'Guidance chạy trên PC [E1].\nRuntime dùng bộ điều khiển và camera [E2].')
        self.assertTrue(validate_answer(result['answer'], sources)['valid'])
        enum = chat.call_args.kwargs['json_schema']['properties']['claims']['items']['properties']['evidence_ids']['items']['enum']
        self.assertEqual(enum, ['E1', 'E2'])
        self.assertEqual(result['generation_input']['claim_evidence_ids'], [['E1'], ['E2']])

    def test_structured_answer_cannot_invent_source_bindings(self):
        sources = [dict(evidence_id='E1', text='Guidance runs on a PC.')]
        for identifiers in ([], ['E999'], ['E1', 1]):
            with self.subTest(identifiers=identifiers), self.assertRaises(QwenGatewayError):
                render_cited_claims(dict(claims=[dict(text='Guidance chạy trên PC.',
                                                    evidence_ids=identifiers)]), sources)

    def test_gateway_delivers_graph_nodes_and_edges_with_sources(self):
        sources=[dict(evidence_id='E1',text='Guidance runs on a PC.')]
        context=dict(nodes=[dict(id='n1',kind='section',label='Setup',evidence_ids=['E1']),
                            dict(id='n2',kind='evidence',label='Guidance',evidence_ids=['E1'])],
                     edges=[dict(from_id='n1',to_id='n2',type='HAS_EVIDENCE',evidence_ids=['E1'])],partial=False)
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value='Guidance runs on a PC [E1].') as chat:
            result=_request_openai_compatible('http://unused','contract_qa',
                dict(query='Guidance',evidence=sources,graph_context=context,answer_detail='explanatory'))
        messages=chat.call_args.args[1]
        self.assertIn('HAS_EVIDENCE',messages[1]['content'])
        delivered=json.loads(messages[1]['content'].rsplit('\n',1)[1])
        group=delivered['relations']['HAS_EVIDENCE'][0]
        edge=(dict(group['constants'],from_id=group['links'][0][0],to_id=group['links'][0][1][0])
              if 'links' in group else dict(group['constants'],**dict(zip(group['columns'],group['rows'][0]))))
        self.assertEqual(edge['from_id'],'n1')
        self.assertEqual(edge['to_id'],'n2')
        if 'evidence_ids' not in edge:
            nodes=[dict(group['constants'],**dict(zip(group['columns'],row))) for group in delivered['nodes']['groups'] for row in group['rows']]
            endpoints=[set(n['evidence_ids']) for n in nodes if n['id'] in {edge['from_id'],edge['to_id']}]
            edge['evidence_ids']=sorted(set.intersection(*endpoints) or set.union(*endpoints))
            self.assertIn('intersection',delivered['edge_source_default'])
        self.assertEqual(edge['evidence_ids'],['E1'])
        self.assertIn('unreviewed semantic edge',messages[0]['content'])
        self.assertNotIn('Trả lời trực tiếp và ngắn gọn',messages[0]['content'])
        self.assertEqual(result['generation_input']['graph_context']['nodes'],2)
        self.assertEqual(result['generation_input']['graph_context']['edges'],1)

    def test_structured_claim_keeps_bindings_without_claiming_semantic_validation(self):
        sources = [dict(evidence_id='E1', text='Thanh toán trong 30 ngày.')]
        answer = render_cited_claims(dict(claims=[dict(text='Thanh toán trong 90 ngày.', evidence_ids=['E1'])]), sources)
        self.assertTrue(validate_answer(answer, sources)['valid'])
        answer = render_cited_claims(dict(claims=[dict(text='Thanh toán trong 30 ngày [E1]. Đây là thời hạn thanh toán.', evidence_ids=['E1'])]), sources)
        self.assertEqual(answer.count('[E1]'), 2)
        self.assertTrue(validate_answer(answer, sources)['valid'])
        with self.assertRaisesMessage(QwenGatewayError, 'conflicts'):
            render_cited_claims(dict(claims=[dict(text='Thanh toán [E2].', evidence_ids=['E1'])]), sources)

    def test_insufficient_evidence_does_not_create_absence_claim_or_fake_citation(self):
        sources = [dict(evidence_id='E1', text='An image processing product.')]
        answer = render_cited_claims(dict(insufficient_evidence=True, claims=[], citation_analysis=[]), sources, True)
        self.assertNotIn('[E1]', answer)
        self.assertIn('chưa thể kết luận', answer)
        with self.assertRaises(QwenGatewayError):
            render_cited_claims(dict(insufficient_evidence=True,
                claims=[dict(text='Không có khoản phạt.', evidence_ids=['E1'])]), sources)

    def test_citation_after_sentence_period_supports_that_claim(self):
        sources = [{'evidence_id': 'E1', 'text': 'Thanh toán trong 30 ngày.'}]
        self.assertTrue(validate_answer('Thanh toán trong 30 ngày. [E1]', sources)['valid'])
        self.assertTrue(validate_answer('Thanh toán trong 90 ngày. [E1]', sources)['valid'])
        self.assertFalse(validate_answer('Thanh toán trong 30 ngày. Không cần hóa đơn.', sources)['valid'])

    def test_native_office_article_heading_resolves_reference(self):
        records = [dict(chunk_id='heading', text='Điều 1. Thanh toán', page_number=1,
                        layout_type='title', bbox=[], structure_metadata={'heading_level': 1, 'heading_source': 'native_style'}),
                   dict(chunk_id='body', text='Thanh toán trong 30 ngày.', page_number=1,
                        layout_type='paragraph', bbox=[], structure_metadata={})]
        tree = analyze_document_structure(records, 'Test')
        _, coverage = structural_retrieve('Thanh toán', tree)
        self.assertEqual(coverage['unresolved_article_references'], [])


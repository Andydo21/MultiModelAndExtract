import json
from unittest.mock import patch
from django.test import TestCase, SimpleTestCase, override_settings
from contracts.tests.test_intelligence import ContractIntelligenceTests
from ai_extract.models import LayoutExtractionRun
from NewIdea.services.knowledge_answer_service import answer_from_sources, generation_coverage, generate_checked_answer
from NewIdea.services.search_job_service import process_next_job


class KnowledgeAnswerChecks(SimpleTestCase):
    def test_visual_source_context_keeps_child_and_unverified_nearby_caption(self):
        from NewIdea.services.search_evidence_service import visual_source_context, select_answer_sources
        figure=dict(chunk_id='f',version_id=1,document_id='d',page_number=1,layout_type='figure',
                    colpali_point_id='point',bbox=[100,100,300,300],text='',heading_path=[],
                    structure_metadata={'surya_region_id':'fig'})
        child=dict(chunk_id='child',version_id=1,document_id='d',page_number=1,layout_type='title',
                   bbox=[120,120,250,140],text='Rich asset library',heading_path=[],
                   structure_metadata={'surya_parent_region_id':'fig'})
        caption=dict(chunk_id='caption',version_id=1,document_id='d',page_number=1,layout_type='figure_caption',
                     bbox=[120,305,280,325],text='Caption',heading_path=[],structure_metadata={})
        foreign=dict(child,chunk_id='foreign',page_number=2,text='Other page')
        context=visual_source_context([figure],[figure,child,caption,foreign])
        self.assertEqual({x['chunk_id'] for x in context},{'child','caption'})
        self.assertEqual(next(x for x in context if x['chunk_id']=='child')['visual_context_links'][0]['kind'],'surya_child')
        self.assertEqual(next(x for x in context if x['chunk_id']=='caption')['visual_context_links'][0]['kind'],'nearby_caption_candidate')
        direct=dict(caption,expansion_reason='query_match')
        selected=select_answer_sources([figure,direct,*context],limit=10)
        self.assertIn('visual_context_links',next(x for x in selected if x['chunk_id']=='caption'))

    def test_source_markers_inside_answer_are_not_removed(self):
        from ai_extract.services.contract_pipeline_service import render_cited_claims
        evidence=[dict(evidence_id='E1',text='First.'),dict(evidence_id='E2',text='Second.')]
        structured=dict(claims=[dict(text='Hình [E1] và [E2].',evidence_ids=['E1','E2'])])
        answer=render_cited_claims(structured,evidence)
        self.assertIn('Hình [E1] và [E2].',answer)
        self.assertIn('[E1]',structured['claims'][0]['text'])

    def test_unverified_caption_candidate_triggers_source_image(self):
        from NewIdea.services.knowledge_answer_service import needs_source_images
        figure=dict(evidence_id='E1',layout_type='figure',image_path='crop.png',text='')
        caption=dict(evidence_id='E2',layout_type='figure_caption',text='A caption',
                     visual_context=[dict(kind='nearby_caption_candidate',figure_evidence_id='E1')])
        self.assertTrue(needs_source_images({},[figure,caption]))
        self.assertFalse(needs_source_images({},[caption]))

    def test_unresolved_table_warns_llm_and_retains_raw_source_without_candidate_rows(self):
        from NewIdea.services.knowledge_answer_service import answer_source_pods,render_source_pods
        source=dict(evidence_id='E1',text='Name Value',structure_metadata=dict(
            table_structure_status='unresolved',table_structure_review=dict(reason='Column disagreement'),
            table_rows=[['Name','Value']]))
        pods=answer_source_pods([source],{})
        record=pods['pods'][0]['sources'][0]
        self.assertEqual(record['text'],'Name Value')
        self.assertNotIn('table_rows',record)
        rendered=render_source_pods(pods)
        self.assertIn('row/column associations are unverified',rendered)
        self.assertIn('Column disagreement',rendered)

    def test_compact_pods_hide_internal_metadata_but_keep_version_boundaries_and_conditions(self):
        from NewIdea.services.knowledge_answer_service import answer_source_pods, render_source_pods
        sources=[dict(evidence_id=f'E{i}',document_id='internal-document',version_id=f'private-version-{i}',
            original_name='Same filename',reading_stream_id='internal-stream',source_location='bbox-secret',
            text=f'Original {i}',conditions='Only with constant temperature',polarity='negative',
            review_status='unreviewed') for i in (1,2)]
        rendered=render_source_pods(answer_source_pods(sources,{}))
        for value in ('internal-document','private-version','internal-stream','bbox-secret'):
            self.assertNotIn(value,rendered)
        for value in ('Document/version 1','Document/version 2','Only with constant temperature','negative','unreviewed'):
            self.assertIn(value,rendered)
        for source in sources:self.assertEqual(rendered.count(source['text']),1)

    def test_source_links_follow_their_source_without_inventing_a_table_note(self):
        from NewIdea.services.knowledge_answer_service import answer_source_pods, render_source_pods
        sources=[dict(evidence_id='E1',text='Repeatability *3',heading_path=['Table']),
            dict(evidence_id='E2',text='*3 Constant ambient temperature',heading_path=['Notes']),
            dict(evidence_id='E3',text='Unrelated same-page note',heading_path=['Notes'])]
        graph=dict(nodes=[dict(id=i,evidence_ids=[f'E{i}']) for i in (1,2,3)],
            edges=[dict(from_id=1,to_id=2,type='REFERENCES',review_status='candidate')])
        rendered=render_source_pods(answer_source_pods(sources,graph))
        table=rendered.split('--- End source scope ---')[0]
        self.assertIn('Outgoing source links',table)
        self.assertIn('[E1] --REFERENCES--> [E2]',table)
        self.assertIn('candidate',table)
        self.assertNotIn('E3',table)
        self.assertEqual(rendered.count('*3 Constant ambient temperature'),1)

    def test_semantic_relation_in_one_passage_keeps_endpoints_negation_and_conditions(self):
        from copy import deepcopy
        from NewIdea.services.knowledge_answer_service import answer_source_pods, render_source_pods
        text='A must not pay B until acceptance.'
        sources=[dict(evidence_id='E1',text=text)]
        graph=dict(nodes=[dict(id=i,kind='entity',label=name,evidence_ids=['E1'])
            for i,name in ((1,'A'),(2,'B'))],edges=[dict(from_id=1,to_id=2,type='PAYS_TO',
                quote=text,scope_quote=text,review_status='unreviewed',polarity='negative',
                modality='must',condition_quotes=['until acceptance'])])
        original=deepcopy(graph)
        pods=answer_source_pods(sources,graph)
        rendered=render_source_pods(pods)
        self.assertIn('"A" --PAYS_TO--> "B"',rendered)
        self.assertIn('negative',rendered)
        self.assertIn('until acceptance',rendered)
        self.assertIn('unreviewed',rendered)
        self.assertEqual(rendered.count(text),1)
        self.assertEqual(graph,original)

    def test_spatial_relations_are_opt_in_and_structural_edges_do_not_fill_prompt(self):
        from NewIdea.services.knowledge_answer_service import answer_source_pods, render_source_pods
        sources=[dict(evidence_id=f'E{i}',text=f'Original {i}') for i in (1,2)]
        graph=dict(nodes=[dict(id=i,evidence_ids=[f'E{i}']) for i in (1,2)],edges=[
            dict(from_id=1,to_id=2,type=kind) for kind in
            ('LEFT_OF','ABOVE','NEAR','NEXT','CONTAINS','HAS_CELL','HAS_EMBEDDING')])
        self.assertFalse(answer_source_pods(sources,graph)['source_links'])
        rendered=render_source_pods(answer_source_pods(sources,graph,include_spatial=True))
        for kind in ('LEFT_OF','ABOVE','NEAR'):self.assertIn(kind,rendered)
        for kind in ('NEXT','CONTAINS','HAS_CELL','HAS_EMBEDDING'):self.assertNotIn(kind,rendered)

    def test_source_pods_keep_document_boundaries_without_inventing_subject_ownership(self):
        from NewIdea.services.knowledge_answer_service import answer_source_pods
        sources=[dict(evidence_id='E1',document_id='one',heading_path=['Alpha'],text='Load: 8 kg.'),
            dict(evidence_id='E2',document_id='two',heading_path=['Other'],text='Temperature: 85 C.')]
        pods=answer_source_pods(sources,dict(nodes=[],edges=[]))['pods']
        self.assertEqual(len(pods),2)
        self.assertEqual(pods[1]['scope']['document_id'],'two')
        self.assertNotIn('subject',pods[1]['scope'])
        self.assertEqual([p['sources'][0]['text'] for p in pods],[s['text'] for s in sources])

    def test_source_pods_preserve_table_cells_spans_and_unmatched_source_text(self):
        from NewIdea.services.knowledge_answer_service import answer_source_pods
        source=dict(evidence_id='E1',text='Name Value\nAlpha 8 kg',structure_metadata=dict(
            table_rows=[['Name','Value'],['Alpha','8 kg']],
            table_cells=[dict(row=1,column=1,rowspan=1,colspan=2)]))
        record=answer_source_pods([source],{})['pods'][0]['sources'][0]
        self.assertEqual(record['table_rows'],source['structure_metadata']['table_rows'])
        self.assertEqual(record['merged_cells'][0]['colspan'],2)
        self.assertNotIn('text',record)
        source['text']+=' Additional source condition.'
        record=answer_source_pods([source],{})['pods'][0]['sources'][0]
        self.assertEqual(record['text'],source['text'])

    def test_markdown_pods_keep_values_with_their_source_headings(self):
        from NewIdea.services.knowledge_answer_service import answer_source_pods, render_source_pods
        sources=[dict(evidence_id=f'E{i}',document_id='same',heading_path=[name],
            text=f'Property Value\nCapacity {value}',structure_metadata=dict(
                table_rows=[['Property','Value'],['Capacity',value]],
                table_cells=[dict(row=1,column=1,header=True)]))
            for i,name,value in [(1,'Alpha','8 kg'),(2,'Beta','5 kg')]]
        rendered=render_source_pods(answer_source_pods(sources,{}))
        scopes=rendered.split('--- End source scope ---')
        self.assertIn('Alpha',scopes[0]);self.assertIn('| Capacity | 8 kg |',scopes[0])
        self.assertNotIn('5 kg',scopes[0])
        self.assertIn('Beta',scopes[1]);self.assertIn('| Capacity | 5 kg |',scopes[1])
        self.assertNotIn('8 kg',scopes[1])

    def test_source_pods_prompt_does_not_send_graph_nodes_or_spatial_edges(self):
        from ai_extract.services.contract_pipeline_service import _request_openai_compatible
        graph=dict(nodes=[dict(id='noise',label='UNRELATED_GRAPH_LABEL',kind='section',evidence_ids=['E1'])],edges=[])
        output=dict(claims=[dict(answer='Thông tin được hỗ trợ.',evidence_ids=['E1'],quotes=[],source_limit='')],
            answer_notes=[],insufficient_evidence=False)
        with patch('ai_extract.services.contract_pipeline_service._chat',return_value=json.dumps(output)) as model:
            result=_request_openai_compatible('http://unused','contract_qa',dict(query='Question?',
                evidence=[dict(evidence_id='E1',text='Original source.',document_id='one')],
                graph_context=graph,source_pods=True,check_completeness=True,inline_review=True,answer_format='cited_claims'))
        prompt=model.call_args.args[1][1]['content']
        self.assertIn('ORIGINAL SOURCE PODS',prompt)
        self.assertEqual(prompt.count('Original source.'),1)
        self.assertNotIn('UNRELATED_GRAPH_LABEL',prompt)
        self.assertNotIn('NEO4J STRUCTURE',prompt)
        self.assertFalse(result['generation_input']['graph_representation']['raw_graph_sent'])

    def test_visual_hit_sends_images_even_if_query_planner_misses_intent(self):
        from NewIdea.services.knowledge_answer_service import needs_source_images
        source=dict(evidence_id='E1', layout_type='figure', image_path='figure.png',
                    colpali_point_id='point', expansion_reason='query_match')
        self.assertTrue(needs_source_images({}, [source]))

    def test_context_split_keeps_figure_and_linked_text_together(self):
        from NewIdea.services.knowledge_answer_service import split_linked_visual_sources
        sources=[dict(evidence_id='E1',layout_type='figure',text=''),
                 dict(evidence_id='E2',text='A separate passage.'),
                 dict(evidence_id='E3',text='Caption',visual_context=[dict(figure_evidence_id='E1',kind='nearby_caption_candidate')]),
                 dict(evidence_id='E4',text='OCR',visual_context=[dict(figure_evidence_id='E1',kind='surya_child')])]
        batches=split_linked_visual_sources(sources)
        ids=[{source['evidence_id'] for source in batch} for batch in batches]
        self.assertTrue(any({'E1','E3','E4'} <= batch for batch in ids))
        self.assertEqual(set.union(*ids),{'E1','E2','E3','E4'})

    def test_citation_inside_sentence_does_not_split_claim(self):
        from ai_extract.services.contract_pipeline_service import validate_answer
        result=validate_answer('Hình [E1] cho thấy robot hoạt động [E1].',
                               [dict(evidence_id='E1', text='robot')])
        self.assertTrue(result['valid'])

    def test_answer_prose_rejects_chinese_but_not_source_quotes(self):
        from ai_extract.services.contract_pipeline_service import render_cited_claims, QwenGatewayError
        evidence=[dict(evidence_id='E1', text='机器人')]
        answer=dict(claims=[dict(text='Robot hoạt động 自动生成路径', evidence_ids=['E1'])])
        with self.assertRaises(QwenGatewayError):
            render_cited_claims(answer, evidence, require_vietnamese=True)

    def test_figure_citation_requires_pixels_in_same_model_call(self):
        from ai_extract.services.contract_pipeline_service import validate_delivered_figure_citations, QwenGatewayError
        evidence=[dict(evidence_id='E1',layout_type='figure'),dict(evidence_id='E2',layout_type='figure'),
                  dict(evidence_id='E3',layout_type='paragraph')]
        sent=[dict(evidence_id='E1',image_path='image.png')]
        validate_delivered_figure_citations(dict(claims=[dict(evidence_ids=['E1','E3'])]),evidence,sent)
        with self.assertRaises(QwenGatewayError):
            validate_delivered_figure_citations(dict(claims=[dict(evidence_ids=['E2'])]),evidence,sent)
        with self.assertRaises(QwenGatewayError):
            validate_delivered_figure_citations(dict(claims=[dict(evidence_ids=['E1'])]),evidence,[])

    def test_model_retries_citation_to_unsent_figure(self):
        from ai_extract.services.contract_pipeline_service import _request_openai_compatible
        wrong=dict(insufficient_evidence=False,claims=[dict(answer='Đây là hình robot.',evidence_ids=['E1'],quotes=[],source_limit='')],answer_notes=[])
        fixed=dict(insufficient_evidence=False,claims=[dict(answer='Nguồn mô tả robot.',evidence_ids=['E2'],quotes=[],source_limit='')],answer_notes=[])
        evidence=[dict(evidence_id='E1',layout_type='figure',image_path='missing.png',text=''),
                  dict(evidence_id='E2',layout_type='paragraph',text='A robot.',document_id='one')]
        with patch('ai_extract.services.contract_pipeline_service._chat',side_effect=[json.dumps(wrong),json.dumps(fixed)]) as model:
            result=_request_openai_compatible('http://unused','contract_qa',dict(query='Robot?',evidence=evidence,
                use_images=False,source_pods=True,check_completeness=True,inline_review=True,answer_format='cited_claims'))
        self.assertEqual(model.call_count,2)
        self.assertEqual(result['citation_claims'][0]['evidence_ids'],['E2'])

    def test_demo_hits_cannot_consume_the_seed_budget_for_real_document_evidence(self):
        from NewIdea.services.knowledge_answer_service import eligible_answer_sources
        from NewIdea.services.search_evidence_service import select_retrieval_seeds
        demo=dict(chunk_id='demo',text='Sample combined answer.',is_demo=True,
            retrieval_hints=[dict(part=1,rank=1),dict(part=2,rank=1)],retrieval_ranks=[dict(rank=1)])
        setup=dict(chunk_id='setup',text='Setup requires a workstation.',
            retrieval_hints=[dict(part=1,rank=2)],retrieval_ranks=[dict(rank=2)])
        operation=dict(chunk_id='operation',text='Operation needs only the controller.',
            retrieval_hints=[dict(part=2,rank=3)],retrieval_ranks=[dict(rank=3)])
        selected=select_retrieval_seeds(eligible_answer_sources([demo,setup,operation]),limit=2)
        self.assertEqual({s['chunk_id'] for s in selected},{'setup','operation'})
        self.assertEqual(eligible_answer_sources([demo]),[])
        self.assertEqual(eligible_answer_sources([setup,operation]),[setup,operation])

    def test_verification_does_not_expand_unrelated_sources_through_shared_parent(self):
        from copy import deepcopy
        from NewIdea.services.knowledge_answer_service import claim_verification_context
        graph=dict(nodes=[dict(id=1,kind='evidence',evidence_ids=['E1']),
            dict(id=2,kind='section',evidence_ids=['E1','E2']),
            dict(id=3,kind='evidence',evidence_ids=['E2']),
            dict(id=4,kind='table_cell',evidence_ids=['E1'],row=1,column=2,rowspan=2)],edges=[
            dict(from_id=2,to_id=1,type='HAS_EVIDENCE',evidence_ids=['E1']),
            dict(from_id=2,to_id=3,type='HAS_EVIDENCE',evidence_ids=['E2']),
            dict(from_id=2,to_id=4,type='HAS_CELL',evidence_ids=['E1'],review_status='unreviewed')])
        original=deepcopy(graph)
        sources=[dict(evidence_id='E1',text='Selected fact.'),dict(evidence_id='E2',text='Other fact.')]
        focused,passages=claim_verification_context(graph,sources,[dict(evidence_ids=['E1'])])
        self.assertEqual(passages,sources[:1])
        self.assertEqual({n['id'] for n in focused['nodes']},{1,2,4})
        self.assertEqual(focused['edges'],[graph['edges'][0],graph['edges'][2]])
        self.assertEqual(next(n for n in focused['nodes'] if n['id']==4)['rowspan'],2)
        self.assertEqual(graph,original)

    def test_shared_parent_bindings_do_not_route_unselected_sibling_branches(self):
        from NewIdea.services.knowledge_answer_service import claim_verification_context
        graph=dict(nodes=[dict(id='parent',kind='section',evidence_ids=['E1','E2']),
            dict(id='a',kind='evidence',evidence_ids=['E1']),
            dict(id='b',kind='evidence',evidence_ids=['E2'])],edges=[
                dict(from_id='parent',to_id='a',type='HAS_EVIDENCE'),
                dict(from_id='parent',to_id='b',type='HAS_EVIDENCE',evidence_ids=['E1','E2'])])
        sources=[dict(evidence_id='E1',text='First branch'),dict(evidence_id='E2',text='Other branch')]
        routed,selected=claim_verification_context(graph,sources,[dict(evidence_ids=['E1'])])
        self.assertEqual({node['id'] for node in routed['nodes']},{'parent','a'})
        self.assertEqual(routed['edges'],graph['edges'][:1])
        self.assertEqual(selected,sources[:1])

    def test_each_part_receives_ranked_source_hints_without_cutting_other_context(self):
        sources=[dict(evidence_id='E1',text='Context.',retrieval_hints=[dict(part=1,rank=2)]),
                 dict(evidence_id='E2',text='Direct explanation.',retrieval_hints=[dict(part=1,rank=1)])]
        calls=[]
        def model(task,sent):
            calls.append(sent)
            return dict(supported=True) if task=='verify_answer' else dict(answer='Explanation [E2].',
                citation_claims=[dict(text='Explanation.',evidence_ids=['E2'],part_ids=['P1'])])
        generate_checked_answer(dict(query='Why?',evidence=sources,coverage=dict(requested_parts=['Why?'])),request_model=model)
        self.assertEqual(calls[0]['answer_parts'][0]['source_ids'],['E2','E1'])
        self.assertEqual(calls[0]['evidence'],sources)

    def test_missing_part_is_reported_inline_without_an_extra_model_call(self):
        calls=[]
        def model(task,sent):
            calls.append(task)
            return dict(answer='First fact [E1].',citation_claims=[dict(text='First fact.',evidence_ids=['E1'],part_ids=['P1'])],
                answer_part_coverage=[dict(id='P1',question='First?',status='answered'),
                    dict(id='P2',question='Second?',status='unavailable')],
                answer_notes=[dict(text='Second detail is not established.',evidence_ids=[])])
        result=generate_checked_answer(dict(query='Both?',evidence=[dict(evidence_id='E1',text='First fact.')],
            coverage=dict(requested_parts=['First?','Second?'])),request_model=model)
        self.assertEqual(calls,['contract_qa'])
        self.assertEqual(result['answer_text'],'First fact.')
        self.assertEqual(result['generation_diagnostics']['completeness_warning']['missing_parts'],['Second?'])
        self.assertIn('Second detail',result['generation_diagnostics']['review_warning']['reason'])

    def test_reconsidered_abstention_still_allows_one_focused_missing_part_repair(self):
        requests=[]
        def model(task,sent):
            if task=='verify_answer':return dict(supported=True)
            requests.append(sent)
            if len(requests)==1:return dict(answer='No evidence.',insufficient_evidence=True)
            part='P1' if len(requests)==2 else 'P2'
            return dict(answer='Supported fact [E1].',citation_claims=[dict(
                text='First fact.' if part=='P1' else 'Second fact.',evidence_ids=['E1'],part_ids=[part])],
                answer_part_coverage=[dict(id='P1',question='First?',status='answered'),
                    dict(id='P2',question='Second?',status='unavailable')] if part=='P1' else [
                    dict(id='P2',question='Second?',status='answered')])
        result=generate_checked_answer(dict(query='Both?',evidence=[dict(evidence_id='E1',text='Both facts.')],
            coverage=dict(requested_parts=['First?','Second?'])),request_model=model)
        self.assertEqual(len(requests),2)
        self.assertEqual(result['answer_text'],'First fact.')
        self.assertEqual(result['generation_diagnostics']['completeness_warning']['missing_parts'],['Second?'])

    def test_atomic_question_facets_survive_shared_retrieval_and_one_generation(self):
        sources=[dict(evidence_id='E1',version_id=1,text='Setup requires a workstation.',retrieval_hints=[dict(part=1,rank=1)]),
                 dict(evidence_id='E2',version_id=2,text='Operation needs a controller.',retrieval_hints=[dict(part=2,rank=1)])]
        planned=[dict(question='Setup?',query_scope=dict(subject='Application',attribute='requirements',phase='setup')),
                 dict(question='Operation?',query_scope=dict(subject='Application',attribute='requirements',phase='operation'))]
        sent=[]
        def model(task,payload):
            self.assertEqual(task,'contract_qa');sent.append(payload)
            return dict(answer='Setup requires a workstation [E1]. Operation needs a controller [E2].')
        result=generate_checked_answer(dict(query='Both?',evidence=sources,coverage=dict(answer_parts=planned)),request_model=model)
        self.assertEqual(len(sent),1)
        packets=sent[0]['source_packets']
        self.assertEqual([p['query_scope']['phase'] for p in packets['parts']],['setup','operation'])
        self.assertEqual([p['direct_source_ids'] for p in packets['parts']],[['E1'],['E2']])
        self.assertEqual(len(packets['source_scopes']),2)
        self.assertEqual(result['generation_status'],'generated')

    def test_contract_answer_compacts_diagnostics_without_removing_source_or_graph(self):
        source=dict(evidence_id='E1',text='Payment within 30 days.')
        graph=dict(nodes=[dict(id='source')],edges=[])
        coverage=dict(mode='structural',graph_context=graph,knowledge_analysis={'trace':'x'*10000})
        calls=[]
        def model(task,sent):
            calls.append(sent)
            return dict(supported=True) if task=='verify_answer' else dict(answer='Payment within 30 days [E1].')
        result=generate_checked_answer(dict(query='payment',evidence=[source],coverage=coverage,graph_context=graph),request_model=model)
        self.assertEqual(result['generation_status'],'generated')
        for sent in calls:
            self.assertEqual(sent['evidence'],[source])
            self.assertEqual(sent['graph_context'],graph)
            self.assertNotIn('knowledge_analysis',sent['coverage'])
            self.assertNotIn('graph_context',sent['coverage'])
        self.assertIn('knowledge_analysis',coverage)

    def test_partial_answer_stays_visible_with_its_inline_limitation(self):
        calls=[]
        def model(task,sent):
            calls.append(task)
            return dict(answer='Supported answer [E1].',answer_notes=[dict(text='More detail is unavailable.',evidence_ids=['E1'])])
        result=generate_checked_answer(dict(query='Explain.',evidence=[dict(evidence_id='E1',text='Source.')]),request_model=model)
        self.assertEqual(result['generation_status'],'generated')
        self.assertEqual(result['answer'],'Supported answer [E1].')
        self.assertEqual(calls,['contract_qa'])
        self.assertIn('More detail',result['generation_diagnostics']['review_warning']['reason'])

    def test_display_prose_is_separate_from_model_selected_citations(self):
        claims=[dict(text='First explanation.',evidence_ids=['E2']),
                dict(text='A qualification.',evidence_ids=['E2','E1'])]
        def model(task,sent):
            return dict(supported=True) if task=='verify_answer' else dict(
                answer='First explanation [E2].\nA qualification [E2] [E1].',citation_claims=claims)
        result=generate_checked_answer(dict(query='Explain.',evidence=[
            dict(evidence_id='E1',text='Source one.'),dict(evidence_id='E2',text='Source two.')]),request_model=model)
        self.assertEqual(result['answer_text'],'First explanation.\n\nA qualification.')
        self.assertEqual(result['answer_source_ids'],['E2','E1'])
        self.assertIn('[E2]',result['answer'])

    def test_validated_source_quotes_are_exposed_once_with_their_source(self):
        from ai_extract.services.contract_pipeline_service import render_cited_claims
        evidence=[dict(evidence_id='E1',text='First complete sentence. Second sentence.')]
        structured=dict(claims=[dict(text='Explanation.',evidence_ids=['E1'],quotes=[
            dict(evidence_id='E1',quote='First complete sentence.'),
            dict(evidence_id='E1',quote='Fabricated sentence.'),
            dict(evidence_id='E1',quote='First complete sentence.')])])
        answer=render_cited_claims(structured,evidence)
        def model(task,sent):
            return dict(answer=answer,citation_claims=structured['claims'])
        result=generate_checked_answer(dict(query='Explain.',evidence=evidence),request_model=model)
        self.assertEqual(result['answer_quotes'],[dict(evidence_id='E1',quote='First complete sentence.')])

    def test_table_quote_uses_original_source_not_generated_markdown(self):
        from ai_extract.services.contract_pipeline_service import render_cited_claims
        evidence=[dict(evidence_id='E1',layout_type='table',text='Maximum payload\n3kg\nCycle time\n0.45sec')]
        structured=dict(claims=[dict(text='Maximum is 3 kg.',evidence_ids=['E1'],quotes=[
            dict(evidence_id='E1',quote='| Maximum payload | 3kg |'),
            dict(evidence_id='E1',quote='| Maximum payload | 4kg |')])])
        render_cited_claims(structured,evidence)
        self.assertEqual(structured['claims'][0]['quotes'],[dict(evidence_id='E1',quote='Maximum payload\n3kg')])

    def test_contained_document_sentence_is_not_a_second_independent_citation(self):
        from NewIdea.services.knowledge_answer_service import distinct_document_passages
        short = dict(evidence_id='E1', version_id=1, page_number=1, heading_path=['Features'], text='Two components.')
        full = dict(short, evidence_id='E2', text='Two components. Guidance runs on a PC.')
        other = dict(short, evidence_id='E3', version_id=2)
        self.assertEqual(distinct_document_passages([short, full, other]), [full, other])
        self.assertEqual(distinct_document_passages([dict(short, heading_path=['Other section']), full]),
                         [dict(short, heading_path=['Other section']), full])

    def test_inline_uncertainty_does_not_suppress_or_regenerate_answer(self):
        calls=[]
        def model(task,sent):
            calls.append(task)
            return dict(answer='Guidance runs on a PC [E1].',answer_notes=[dict(
                text='The setup scope needs review.',evidence_ids=['E1'])])
        result=generate_checked_answer(dict(query='Guidance',evidence=[dict(evidence_id='E1',text='Guidance runs on a PC.')]),request_model=model)
        self.assertEqual(calls,['contract_qa'])
        self.assertIn('Guidance runs',result['answer_text'])
        self.assertIn('setup scope',result['generation_diagnostics']['review_warning']['reason'])
        self.assertNotIn('semantic_check',result['citation_check'])

    def test_raw_outputs_survive_answer_repair_and_verification(self):
        payload=dict(query='payment',evidence=[dict(evidence_id='E1',text='Payment within 30 days.')])
        answers=iter(('Payment within 90 days [E99].','Payment within 30 days [E1].'))
        count=0
        def model(task,sent):
            nonlocal count
            count+=1
            raw=dict(task=task,response=dict(id=f'call-{count}',choices=[dict(message=dict(content='raw'))]))
            if task=='verify_answer':
                return dict(supported=True,qwen_outputs=[raw])
            return dict(answer=next(answers),qwen_outputs=[raw])
        result=generate_checked_answer(payload,request_model=model)
        self.assertEqual(result['generation_status'],'generated')
        self.assertEqual([item['stage'] for item in result['qwen_outputs']],['answer','citation_repair'])
        self.assertEqual([item['response']['id'] for item in result['qwen_outputs']],['call-1','call-2'])
        self.assertNotIn('semantic_check',result['citation_check'])

    def test_citation_repair_keeps_same_sources_and_graph_before_verification(self):
        source = dict(evidence_id='E1', text='Payment within 30 days.')
        payload = dict(query='payment', evidence=[source], instruction='Use source citations.',
                       graph_context=dict(nodes=[dict(id='n1')], edges=[]))
        calls = []

        def model(task, sent):
            calls.append((task, sent))
            if task == 'verify_answer':
                return dict(supported=True)
            return dict(answer='Payment within 90 days [E99].' if len(calls) == 1
                        else 'Payment within 30 days [E1].')

        result = generate_checked_answer(payload, request_model=model)
        self.assertEqual(result['generation_status'], 'generated')
        self.assertEqual([task for task, _ in calls], ['contract_qa', 'contract_qa'])
        for _, sent in calls:
            self.assertEqual(sent['evidence'], payload['evidence'])
            self.assertEqual(sent['graph_context'], payload['graph_context'])
        self.assertEqual(result['generation_calls'][1]['stage'], 'citation_repair')
        self.assertEqual(payload['instruction'], 'Use source citations.')

    def test_repeated_invalid_claim_stops_after_one_repair_without_verification(self):
        payload = dict(query='payment', evidence=[dict(evidence_id='E1', text='Payment within 30 days.')])
        with patch('NewIdea.services.knowledge_answer_service.request_qwen',
                   return_value=dict(answer='Payment within 90 days [E99].')) as model:
            result = generate_checked_answer(payload)
        self.assertEqual(model.call_count, 2)
        self.assertTrue(all(call.args[0] == 'contract_qa' for call in model.call_args_list))
        self.assertEqual(result['generation_status'], 'source_excerpt')
        self.assertNotIn('90 days', result['answer'])
        self.assertFalse(result['citation_check']['valid'])

    def test_demo_advice_does_not_pad_ordinary_document_answers(self):
        sources=[dict(evidence_id='E1',text='Guidance runs on a PC.'),
                 dict(evidence_id='E2',text='Guidance advice sample.',is_demo=True,evidence_kind='expert_opinion')]
        seen=[]
        def model(task,payload):
            seen.append([e['evidence_id'] for e in payload['evidence']])
            return dict(supported=True) if task=='verify_answer' else dict(answer='Guidance runs on a PC [E1].')
        with patch('NewIdea.services.knowledge_answer_service.request_qwen',side_effect=model):
            answer_from_sources('Guidance PC',sources,{})
            self.assertTrue(all(ids==['E1'] for ids in seen))
            seen.clear()
            answer_from_sources('Guidance advice',sources,{})
            self.assertTrue(all(ids==['E1'] for ids in seen))

    def test_generation_and_verification_receive_source_bound_graph(self):
        source=dict(evidence_id='E1',version_id=1,chunk_id='leaf1',text='Guidance runs on a PC.')
        graph=dict(state='ready',version_id=1,nodes=[
            dict(id='v1:n1',kind='section',label='Setup',source_leaf_ids=['leaf1']),
            dict(id='v1:n2',kind='evidence',label='Guidance',source_leaf_ids=['leaf1'])],
            edges=[dict(from_id='v1:n1',to_id='v1:n2',type='HAS_EVIDENCE',source_leaf_ids=[])])
        calls=[]
        def model(task,payload):
            calls.append(payload)
            return dict(supported=True) if task=='verify_answer' else dict(answer='Guidance runs on a PC [E1].')
        with patch('NewIdea.services.knowledge_answer_service.request_qwen',side_effect=model):
            result=answer_from_sources('Guidance PC',[source],dict(version_details=[dict(coverage=dict(graph_context=graph))]))
        self.assertEqual(result['generation_status'],'generated')
        self.assertEqual(len(calls),1)
        self.assertTrue(all(c['graph_context']['edges'][0]['evidence_ids']==['E1'] for c in calls))
        self.assertNotIn('graph_context',calls[0]['coverage'])
        self.assertEqual(calls[0]['answer_detail'],'explanatory')

    def test_verbose_retrieval_diagnostics_do_not_fill_qwen_context(self):
        coverage = dict(mode='knowledge_discovery', partial=True, versions_scanned=4,
            version_details=[dict(coverage=dict(structure_analysis={'large': 'x' * 10000},
                unresolved_document_references=['Appendix X']))])
        compact = generation_coverage(coverage)
        self.assertNotIn('version_details', compact)
        self.assertTrue(compact['partial'])
        self.assertEqual(compact['unresolved_document_references'], ['Appendix X'])
        self.assertLess(len(json.dumps(compact)), 500)
        self.assertIn('version_details', coverage)

    def test_source_sufficiency_is_decided_by_model_not_keyword_overlap(self):
        source = dict(evidence_id='E1', text='Replace the E17 sensor.', source_location='Page 1')
        with patch('NewIdea.services.knowledge_answer_service.request_qwen', return_value=dict(
                answer='The supplied source does not describe Guidance or Runtime.', insufficient_evidence=True)) as model:
            result = answer_from_sources('EVP2 Guidance Runtime PC', [source], {})
        self.assertEqual(result['generation_status'], 'abstained')
        self.assertNotIn('Replace', result['answer'])
        self.assertTrue(model.called)
        self.assertTrue(all(c.args[0]=='contract_qa' for c in model.call_args_list))

    def test_unsupported_claim_remains_visible_with_advisory_warning(self):
        source = dict(evidence_id='E1', text='The contract price is 120 VND.', source_location='Page 1')
        def model(task, payload):
            self.assertEqual(payload['answer_format'], 'cited_claims')
            self.assertEqual(task,'contract_qa')
            return dict(answer='The penalty is 120 VND [E1].',answer_notes=[dict(
                text='The source states a price, not a penalty.',evidence_ids=['E1'])])
        with patch('NewIdea.services.knowledge_answer_service.request_qwen', side_effect=model):
            result = answer_from_sources('contract price', [source], {})
        self.assertEqual(result['generation_status'], 'generated')
        self.assertIn('The penalty is', result['answer'])
        self.assertIn('review_warning', result['generation_diagnostics'])

    def test_review_timeout_does_not_discard_generated_answer(self):
        from ai_extract.services.contract_pipeline_service import QwenGatewayError
        calls=[]
        def model(task,payload):
            calls.append(task)
            if task=='verify_answer':raise QwenGatewayError('review timeout')
            return dict(answer='Payment within 30 days [E1].')
        result=generate_checked_answer(dict(query='payment',evidence=[
            dict(evidence_id='E1',text='Payment within 30 days.')]),request_model=model)
        self.assertEqual(calls,['contract_qa'])
        self.assertEqual(result['generation_status'],'generated')
        self.assertIn('Payment within 30 days',result['answer_text'])
        self.assertNotIn('review_warning',result['generation_diagnostics'])
        self.assertNotIn('semantic_check',result['citation_check'])

    def test_figure_pods_render_structured_markdown_and_preserve_evidence_for_text_llm(self):
        from NewIdea.services.knowledge_answer_service import answer_source_pods, render_source_pods
        fig_source = dict(
            evidence_id='E1',
            layout_type='figure',
            text='',
            structure_metadata=dict(
                caption_text='Hình 3.1: Sơ đồ khối driver động cơ',
                in_image_text=['24V DC', 'Port EtherCAT', 'E-Stop'],
                summary_description='Sơ đồ kết nối nguồn 24V và cổng tín hiệu an toàn E-Stop tới driver.',
            )
        )
        pods = answer_source_pods([fig_source], {})
        rendered = render_source_pods(pods)
        self.assertIn('**[HÌNH ẢNH / SƠ ĐỒ KỸ THUẬT: Hình 3.1: Sơ đồ khối driver động cơ]**', rendered)
        self.assertIn('- Chú thích: Hình 3.1: Sơ đồ khối driver động cơ', rendered)
        self.assertIn('Port EtherCAT', rendered)
        self.assertIn('E-Stop', rendered)
        self.assertIn('- Mô tả nội dung sơ đồ: Sơ đồ kết nối nguồn 24V', rendered)

        calls = []
        def model(task, sent):
            calls.append((task, sent))
            return dict(supported=True) if task == 'verify_answer' else dict(answer='Driver kết nối qua Port EtherCAT và có nút E-Stop [E1].')
        result = generate_checked_answer(
            dict(query='Kết nối driver thế nào?', evidence=[fig_source], use_images=False),
            request_model=model
        )
        self.assertTrue(any(task == 'contract_qa' for task, _ in calls))
        qa_sent = next(sent for task, sent in calls if task == 'contract_qa')
        self.assertEqual(qa_sent['evidence'][0]['evidence_id'], 'E1')
        self.assertEqual(result['generation_status'], 'generated')



@override_settings(A3_HYBRID_ENABLED=False)
class CorpusKnowledgeAnswers(TestCase):
    setUp = ContractIntelligenceTests.setUp

    def test_raw_qwen_output_is_persisted_and_job_access_remains_scoped(self):
        from NewIdea.services.search_job_service import reserve_search, finish_search
        raw=[dict(task='contract_qa',stage='answer',response=dict(id='raw-output',choices=[dict(message=dict(content='{"claims":[]}'))]))]
        receipt=reserve_search(self.user,'Question',generated=True,scope='knowledge')
        finish_search(receipt,dict(answer='Answer',qwen_outputs=raw))
        self.client.force_login(self.user)
        response=self.client.get('/api/newidea/jobs/',dict(id=receipt.pk))
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['result']['qwen_outputs'],raw)
        self.client.force_login(self.outsider)
        response=self.client.get('/api/newidea/jobs/',dict(id=receipt.pk))
        self.assertIn(response.status_code,(403,404))

    def post(self, path='/api/newidea/search/', **body):
        return self.client.post(path, json.dumps({'query': 'payment', 'generate': True, **body}), content_type='application/json')

    def model(self, task, payload):
        return {'supported': True} if task == 'verify_answer' else {'answer': 'Payment within 60 days [E1].'}

    def index(self):
        return LayoutExtractionRun.objects.create(version=self.new_version, status='COMPLETED', dense_index_status='COMPLETED')

    def foreign_index(self):
        from ai_extract.models import LayoutExtractionPage, LayoutExtractionBlock
        run = LayoutExtractionRun.objects.create(version=self.foreign_version, status='COMPLETED', dense_index_status='COMPLETED')
        page = LayoutExtractionPage.objects.create(run=run, page_number=1)
        LayoutExtractionBlock.objects.create(page=page, block_order=1, text='EVP2 Guidance configures on a PC. EVP2 Runtime uses the controller and camera.')

    def test_admin_discovers_indexed_knowledge_in_another_company(self):
        self.foreign_index()
        self.user.is_superuser = True
        self.user.save(update_fields=['is_superuser'])
        result = self.post(query='EVP2', generate=False).json()
        self.assertEqual(result['context_coverage']['access_scope'], 'all_companies')
        self.assertEqual({c['version_id'] for c in result['citations']}, {self.foreign_version.pk})

    def test_regular_user_cannot_discover_foreign_knowledge(self):
        self.foreign_index()
        result = self.post(query='EVP2', generate=False).json()
        self.assertEqual(result['citations'], [])

    def test_named_product_excludes_unrelated_runtime_document(self):
        from ai_extract.models import Clause
        self.foreign_index()
        self.index()
        Clause.objects.create(version=self.new_version, clause_title='Runtime', clause_content='Runtime for machine AX-203 uses a controller and camera.')
        self.user.is_superuser = True
        self.user.save(update_fields=['is_superuser'])
        result = self.post(query='EVP2 Runtime controller camera', generate=False).json()
        self.assertEqual({c['version_id'] for c in result['citations']}, {self.foreign_version.pk})
        self.assertIn(self.foreign_version.pk, result['context_coverage']['identifier_matched_versions'])

    def test_comparison_keeps_documents_for_both_requested_identifiers(self):
        from ai_extract.models import Clause
        self.foreign_index()
        self.index()
        Clause.objects.create(version=self.new_version, clause_title='Runtime', clause_content='AX-203 runtime uses a controller.')
        self.user.is_superuser = True
        self.user.save(update_fields=['is_superuser'])
        result = self.post(query='EVP2 versus AX-203 runtime', generate=False).json()
        self.assertEqual({c['version_id'] for c in result['results']}, {self.foreign_version.pk, self.new_version.pk})

    def test_admin_without_company_can_queue_and_read_corpus_job(self):
        self.foreign_index()
        self.user.company = None
        self.user.is_superuser = True
        self.user.save(update_fields=['company', 'is_superuser'])
        queued = self.post('/api/newidea/jobs/', scope='knowledge', query='EVP2')
        self.assertEqual(queued.status_code, 202)
        pk = queued.json()['search_id']
        self.assertEqual(self.client.get('/api/newidea/jobs/', {'id': pk}).status_code, 200)
        with patch('NewIdea.services.knowledge_answer_service.request_qwen', side_effect=lambda task, payload:
                   {'supported': True} if task == 'verify_answer' else {'answer': 'EVP2 Runtime uses the controller and camera [E1].'}):
            process_next_job(lambda *args, **kwargs: self.fail('Unexpected version job'))
        receipt = self.client.get('/api/newidea/jobs/', {'id': pk}).json()
        self.assertEqual(receipt['status'], 'COMPLETED')
        self.assertEqual({c['version_id'] for c in receipt['result']['sources']}, {self.foreign_version.pk})

    def test_revoking_admin_access_hides_corpus_job_and_prevents_execution(self):
        self.user.is_superuser = True
        self.user.save(update_fields=['is_superuser'])
        pk = self.post('/api/newidea/jobs/', scope='knowledge').json()['search_id']
        self.user.is_superuser = False
        self.user.save(update_fields=['is_superuser'])
        with patch('NewIdea.services.knowledge_search_service.search_knowledge') as search:
            process_next_job(lambda *args, **kwargs: self.fail('Unexpected version job'))
        search.assert_not_called()
        self.assertEqual(self.client.get('/api/newidea/jobs/', {'id': pk}).status_code, 404)

    def test_answers_without_contract_selection_and_only_from_indexed_knowledge(self):
        self.index()
        with patch('ai_extract.services.contract_pipeline_service.request_qwen', side_effect=RuntimeError('Structure offline')), patch('NewIdea.services.knowledge_answer_service.request_qwen', side_effect=self.model):
            response = self.post()
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result['generation_status'], 'generated')
        self.assertIn('60 days', result['answer'])
        self.assertTrue(all(c['version_id'] == self.new_version.pk for c in result['citations']))

    def test_available_text_is_searchable_even_without_vector_index(self):
        with patch('NewIdea.services.knowledge_answer_service.request_qwen',side_effect=self.model) as model:
            result = self.post().json()
        self.assertEqual(result['generation_status'], 'generated')
        self.assertIn('60 days',result['answer'])
        self.assertFalse(result['context_coverage']['indexed_only'])
        self.assertEqual(result['context_coverage']['versions_scanned'],result['context_coverage']['versions_total'])
        self.assertTrue(model.called)

    def test_company_job_is_queued_and_readable_without_version(self):
        self.index()
        response = self.post('/api/newidea/jobs/', scope='company')
        self.assertEqual(response.status_code, 202)
        pk = response.json()['search_id']
        self.assertEqual(self.client.get('/api/newidea/jobs/', {'id':pk}).status_code, 200)
        with patch('ai_extract.services.contract_pipeline_service.request_qwen', side_effect=RuntimeError('Structure offline')), patch('NewIdea.services.knowledge_answer_service.request_qwen', side_effect=self.model):
            process_next_job(lambda *args, **kwargs: self.fail('Version worker must not run for company job'))
        receipt = self.client.get('/api/newidea/jobs/', {'id':pk}).json()
        self.assertEqual(receipt['status'], 'COMPLETED')
        self.assertEqual(receipt['result']['response']['generation_status'], 'generated')
        self.user.company = self.other
        self.user.save(update_fields=['company'])
        self.assertEqual(self.client.get('/api/newidea/jobs/', {'id':pk}).status_code, 404)

    def test_company_change_before_worker_prevents_execution(self):
        pk = self.post('/api/newidea/jobs/', scope='company').json()['search_id']
        self.user.company = self.other
        self.user.save(update_fields=['company'])
        with patch('NewIdea.services.knowledge_search_service.search_knowledge') as search:
            process_next_job(lambda *args, **kwargs: self.fail('Unexpected version job'))
        search.assert_not_called()
        from contracts.models import KnowledgeSearchLog
        self.assertEqual(KnowledgeSearchLog.objects.get(pk=pk).status, 'FAILED')

    def test_approved_expert_case_reaches_generation_and_stale_case_is_excluded(self):
        pk = ContractIntelligenceTests.write_case(self).json()['case']['id']
        self.client.force_login(self.manager)
        self.assertEqual(ContractIntelligenceTests.action(self, 'approve', pk).status_code, 200)
        self.client.force_login(self.user)
        self.index()
        payloads = []
        def model(task, payload):
            payloads.append(payload)
            return self.model(task, payload)
        with patch('NewIdea.services.knowledge_answer_service.request_qwen', side_effect=model):
            result = self.post().json()
        opinions = [c for c in result['citations'] if c.get('evidence_kind') == 'expert_opinion']
        self.assertEqual(len(opinions), 1)
        self.assertEqual(opinions[0]['applicability'], self.fields['applicability'])
        self.assertTrue(any(c.get('evidence_kind') == 'expert_opinion' for c in payloads[0]['evidence']))
        from ai_extract.models import Clause
        Clause.objects.filter(version=self.version).update(clause_content='Replaced source')
        with patch('NewIdea.services.knowledge_answer_service.request_qwen', side_effect=model):
            changed = self.post().json()
        self.assertFalse(any(c.get('evidence_kind') == 'expert_opinion' for c in changed['citations']))
class StructureReviewOrderTests(SimpleTestCase):
    def test_disputed_next_cannot_restore_native_order_as_confirmed(self):
        from NewIdea.services.knowledge_answer_service import graph_source_order
        sources=[dict(version_id=1,chunk_id='a',reading_document_position=1,reading_stream_status='model_predicted'),
                 dict(version_id=1,chunk_id='b',reading_document_position=0,reading_stream_status='model_predicted')]
        context=dict(state='ready',version_id=1,nodes=[dict(id='a',source_leaf_ids=['a']),dict(id='b',source_leaf_ids=['b'])],
                     edges=[dict(from_id='b',to_id='a',type='NEXT',review_status='model_disputed')])
        self.assertEqual(graph_source_order([context],sources),sources)



class FocusedRepairIntegrationTests(SimpleTestCase):
    def test_packets_keep_direct_dependencies_without_shared_scope_fanout(self):
        sources=[dict(evidence_id='E1',text='First source.',retrieval_hints=[dict(part=1,rank=1)]),
                 dict(evidence_id='E2',text='Second source.',retrieval_hints=[dict(part=2,rank=1)]),
                 dict(evidence_id='E3',text='Applicability condition.')]
        graph=dict(nodes=[dict(id=i,evidence_ids=['E'+str(i)]) for i in (1,2,3)],
                   edges=[dict(type='DEPENDS_ON',from_id=2,to_id=3)])
        calls=[]
        def model(task,sent):
            calls.append((task,sent))
            return dict(answer='First answer [E1]. Second answer under its condition [E2] [E3].')
        result=generate_checked_answer(dict(query='Both?',evidence=sources,graph_context=graph,
            coverage=dict(requested_parts=['First?','Second?'])),request_model=model)
        self.assertEqual(len(calls),1)
        packets=calls[0][1]['source_packets']['parts']
        self.assertEqual(packets[0]['direct_source_ids'],['E1'])
        self.assertEqual(packets[0]['related_source_ids'],[])
        self.assertEqual(packets[1]['direct_source_ids'],['E2'])
        self.assertEqual(packets[1]['related_source_ids'],['E3'])
        self.assertEqual(calls[0][1]['graph_context'],graph)
        self.assertEqual(result['generation_status'],'generated')

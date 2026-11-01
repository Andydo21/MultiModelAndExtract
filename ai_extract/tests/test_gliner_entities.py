from types import SimpleNamespace
from unittest.mock import Mock, patch
from django.test import SimpleTestCase
from ai_extract.services.gliner_entity_service import infer_entities,extract_graph_entities,validate_mentions,text_windows,mention_key
from ai_extract.services.document_graph_service import projection


class GLiNEREntityTests(SimpleTestCase):
    def test_flat_source_decoder_resolves_window_overlap_but_preserves_other_occurrences(self):
        from ai_extract.services.gliner_entity_service import flat_source_entities
        def e(source,start,end,name,score):
            return dict(source_id=source,start=start,end=end,name=name,confidence=score,entity_type='ENTITY')
        items=[e('a',0,14,'Unity Industry',.9),e('a',6,14,'Industry',.6),
               e('a',15,20,'Unity',.8),e('b',0,14,'Unity Industry',.7),e('a',20,23,'ABC',.8)]
        result=flat_source_entities(items)
        self.assertEqual([(e['source_id'],e['start'],e['end']) for e in result],
                         [('a',0,14),('a',15,20),('a',20,23),('b',0,14)])

    def test_flat_ner_admission_rejects_duplicate_and_nested_runtime_spans(self):
        text='Unity Industry'
        full=dict(source_id='u',start=0,end=14,name=text,entity_type='ENTITY',confidence=.9)
        nested=dict(source_id='u',start=6,end=14,name='Industry',entity_type='ENTITY',confidence=.8)
        for proposals in ([full,dict(full)],[full,nested]):
            with self.subTest(proposals=proposals):
                result=extract_graph_entities(dict(leaves=[dict(chunk_id='u',text=text)]),
                    lambda units:dict(entities=proposals,relations=[]))
                self.assertEqual(result['status'],'unavailable')
                self.assertEqual(result['entities'],[])
                self.assertIn('overlapping',result['errors'][0]['reason'])

    def relation_review_fixture(self,polarity='affirmed'):
        from ai_extract.services.gliner_identity_classifier import RELATION_REVIEW_SCHEMA,RELATION_SUPPORT_LABELS
        from ai_extract.tests.test_gliner_identity_classifier import entailment_proof,kind_proof
        text='Motor A is not part of Robot B.' if polarity=='negated' else 'Motor A is part of Robot B.'
        entities=[dict(source_id='u',start=text.index(n),end=text.index(n)+len(n),name=n,
                       entity_type='ENTITY',confidence=.95) for n in ('Motor A','Robot B')]
        proposal=dict(source_id='u',from_key=mention_key(entities[0]),to_key=mention_key(entities[1]),
                      predicate='part_of',quote=text,confidence=.92)
        result=dict(entities=entities,relations=[proposal])
        scores={k:.97 if k==polarity else .01 for k in RELATION_SUPPORT_LABELS}
        status=polarity if polarity!='uncertain' else 'unresolved'
        claim=dict(id='0',head='Motor A',tail='Robot B',predicate='part of',source=text,target_source=None)
        reviewer=Mock(return_value=dict(schema=RELATION_REVIEW_SCHEMA,decisions=[
            dict(id='0',status=status,scores=[scores,dict(scores)],source_entailment=entailment_proof(claim,polarity),**kind_proof(claim))]))
        return result,[dict(id='u',text=text)],reviewer

    def test_active_relation_schema_excludes_previous_unsupported_predicates(self):
        from ai_extract.services.gliner_entity_service import RELATION_LABELS
        self.assertEqual(set(RELATION_LABELS.values()),
            {'part_of','located_in','designed_by','supports','integrates_with'})
        self.assertFalse(set(RELATION_LABELS)&{'references','requires','has amount','pays to','applies to'})

    def test_model_review_preserves_negation_and_original_extractor_confidence(self):
        from ai_extract.services.gliner_entity_service import review_entity_relations
        result,units,reviewer=self.relation_review_fixture('negated')
        reviewed=review_entity_relations(result,units,reviewer)
        self.assertEqual(reviewed['relations'][0]['polarity'],'negated')
        self.assertEqual(reviewed['relations'][0]['confidence'],.92)
        self.assertEqual(reviewed['entities'],result['entities'])
        self.assertEqual(reviewed['relation_proposals'],result['relations'])
        tree=dict(nodes=[dict(id='root',kind='document',parent_id=None)],source_fingerprint='x',
                  leaves=[dict(chunk_id='u',text=units[0]['text'],node_id='root',ancestor_ids=['root'])])
        tree['knowledge_graph']=dict(reviewed,entities=[dict(e,key=mention_key(e),source_ids=[e['source_id']]) for e in reviewed['entities']],
            relations=[dict(r,origin='gliner_relex_source_span') for r in reviewed['relations']])
        run=SimpleNamespace(pk=1,version_id=2,version=SimpleNamespace(contract=SimpleNamespace(pk=3,company_id=4)))
        edge=next(e for e in projection(run,tree)[2] if e['type']=='PART_OF')
        self.assertEqual(edge['polarity'],'negated')
        self.assertIn('negated',edge['relation_review_json'])

    def test_uncertain_or_conditional_model_review_withholds_edges_but_keeps_ner(self):
        from ai_extract.services.gliner_entity_service import review_entity_relations
        for status in ('uncertain','conditional'):
            result,units,reviewer=self.relation_review_fixture(status)
            reviewed=review_entity_relations(result,units,reviewer)
            self.assertEqual(reviewed['relations'],[])
            self.assertEqual(reviewed['entities'],result['entities'])
            self.assertEqual(reviewed['relation_review']['withheld'],1)

    def test_failed_or_contradictory_relation_review_cannot_publish_edges(self):
        from ai_extract.services.gliner_entity_service import review_entity_relations
        for invalid in (False,True):
            result,units,reviewer=self.relation_review_fixture('negated')
            if invalid:reviewer.return_value['decisions'][0]['status']='affirmed'
            else:reviewer.side_effect=RuntimeError('Model unavailable')
            reviewed=review_entity_relations(result,units,reviewer)
            self.assertEqual(reviewed['relation_review']['status'],'unavailable')
            self.assertEqual(reviewed['relations'],[])
            self.assertEqual(reviewed['entities'],result['entities'])

    def test_missing_entailment_evidence_cannot_publish_relation(self):
        from ai_extract.services.gliner_entity_service import review_entity_relations
        result,units,reviewer=self.relation_review_fixture()
        reviewer.return_value['decisions'][0].pop('source_entailment')
        reviewed=review_entity_relations(result,units,reviewer)
        self.assertEqual(reviewed['relations'],[])
        self.assertEqual(reviewed['entities'],result['entities'])
        self.assertEqual(reviewed['relation_review']['status'],'unavailable')

    def grouped_pairs(self):
        from ai_extract.services.gliner_entity_service import reference_context_pairs
        tree=dict(leaves=[dict(chunk_id='table',text='Torque values: 10 Nm.',layout_type='table',
                              reading_document_position=0,page_number=1,heading_path=['Installation']),
                          dict(chunk_id='menu',text='Site Map',layout_type='paragraph',reading_document_position=1),
                          dict(chunk_id='source',text='Use the values above.',layout_type='paragraph',
                               reading_document_position=2,heading_path=['Installation'])])
        search=dict(selected_pairs=[dict(from_id='source',from_offset=0,to_id='table',to_offset=0),
                                    dict(from_id='source',from_offset=0,to_id='menu',to_offset=0)])
        return reference_context_pairs(tree,search)[0]

    def group_predictor(self,mode='table'):
        import re
        model=Mock()
        def classify(text,tasks,**kwargs):
            labels=tasks['reference_target']['labels']
            winner=('no_reference' if mode=='none' else 'candidate_0' if mode=='first' else
                    re.search(r'(candidate_\d+) \{"id": "table"',text).group(1))
            return dict(reference_target=[dict(label=k,confidence=.99 if k==winner else .001) for k in labels])
        def extract(text,*args,**kwargs):
            left='Use the values above.';right='Torque values' if mode!='wrong_span' else 'Site Map'
            a=text.index(left);b=text.index(right)
            return dict(relation_extraction={'references_table':[dict(
                head=dict(text=left,start=a,end=a+len(left),confidence=.98),
                tail=dict(text=right,start=b,end=b+len(right),confidence=.97))]})
        model.classify_text.side_effect=classify;model.extract.side_effect=extract
        return model

    def test_group_keeps_layout_and_competing_targets_with_original_spans(self):
        from ai_extract.services.gliner_entity_service import infer_reference_pairs
        pairs=self.grouped_pairs();model=self.group_predictor()
        decisions=infer_reference_pairs(pairs,model)['decisions']
        self.assertEqual([d['status'] for d in decisions],['reference','unresolved'])
        self.assertEqual(decisions[0]['references'][0]['evidence']['target_span'],
                         dict(text='Torque values',start=0,end=13))
        text=model.extract.call_args.args[0]
        self.assertIn('Installation',text);self.assertIn('"layout_type": "table"',text)
        self.assertLess(text.index('Torque values: 10 Nm.'),text.index('Use the values above.'))
        self.assertIn('Site Map',text)
        self.assertEqual(model.classify_text.call_count,2)

    def test_group_none_and_candidate_id_bias_never_create_reference(self):
        from ai_extract.services.gliner_entity_service import infer_reference_pairs
        for mode,reason in (('none','model_no_reference'),('first','target_assignment_disagreement')):
            model=self.group_predictor(mode)
            decisions=infer_reference_pairs(self.grouped_pairs(),model)['decisions']
            self.assertTrue(all(d['status']=='unresolved' and d['reason']==reason for d in decisions))
            model.extract.assert_not_called()

    def test_selected_target_cannot_use_a_span_from_another_candidate(self):
        from ai_extract.services.gliner_entity_service import infer_reference_pairs
        decisions=infer_reference_pairs(self.grouped_pairs(),self.group_predictor('wrong_span'))['decisions']
        self.assertTrue(all(d['status']=='unresolved' for d in decisions))
        self.assertEqual(len(decisions[0]['rejected_instances']),1)

    def test_incomplete_group_is_rejected_instead_of_comparing_pairs_independently(self):
        from ai_extract.services.gliner_entity_service import infer_reference_pairs
        model=self.group_predictor()
        with self.assertRaisesRegex(ValueError,'Incomplete reference target group'):
            infer_reference_pairs(self.grouped_pairs()[:1],model)
        model.classify_text.assert_not_called()

    def test_layout_adds_a_previous_table_missing_from_semantic_retrieval(self):
        from ai_extract.services.gliner_entity_service import reference_context_pairs
        leaves=[dict(chunk_id='table',text='Dimensions table',layout_type='table',reading_document_position=0)]
        leaves.extend(dict(chunk_id=str(i),text=f'Paragraph {i}',reading_document_position=i) for i in range(1,6))
        search=dict(selected_pairs=[dict(from_id='5',from_offset=0,to_id='4',to_offset=0)])
        pairs,_=reference_context_pairs(dict(leaves=leaves),search)
        self.assertIn('table',{p['to_id'] for p in pairs})
        source={l['chunk_id']:l['text'] for l in leaves}
        for pair in pairs:
            for prefix in ('from','to'):
                offset=pair[prefix+'_offset'];text=pair[prefix+'_text']
                self.assertEqual(source[pair[prefix+'_id']][offset:offset+len(text)],text)

    def test_target_choice_rejects_missing_and_nonfinite_vote_scores(self):
        from ai_extract.services.gliner_entity_service import reference_target_choice
        for votes in ([],[{'0':.99},{'0':.99}],
                      [{'0':float('nan'),'no_reference':.01}]*2):
            with self.assertRaises(ValueError):reference_target_choice(votes,['0'])

    def test_described_reference_http_contract_checks_model_revision_and_policy(self):
        from ai_extract.services.gliner_entity_service import request_reference_pairs,reference_policy,REFERENCE_SCHEMA
        policy=reference_policy();pair=dict(id='pair',left='See the table.',right='Table heading.')
        result=dict(model=policy['model'],revision=policy['revision'],schema=REFERENCE_SCHEMA,
            inference_policy=policy,decisions=[dict(id='pair',status='unresolved',references=[])])
        with patch('requests.post') as post:
            post.return_value.json.return_value=result
            self.assertEqual(request_reference_pairs([pair]),result)
            self.assertTrue(post.call_args.args[0].endswith('/rag/gliner2_references'))
            for key in ('model','revision','schema','inference_policy'):
                post.return_value.json.return_value=dict(result,**{key:'wrong'})
                with self.assertRaises(ValueError):request_reference_pairs([pair])

    def test_described_relations_normalize_legacy_keys_without_losing_instances(self):
        from ai_extract.services.gliner_entity_service import normalize_described_relations
        row=dict(head=dict(text='John',start=0,end=4,confidence=.9),tail=dict(text='Company',start=5,end=12,confidence=.8))
        description='Employment relationship'
        raw={'relation_extraction':{'works_for: '+description:[row],'works_for':[]}}
        result=normalize_described_relations(raw,{'works_for':description})
        self.assertEqual(result,{'works_for':[row]})
        self.assertEqual(raw['relation_extraction']['works_for'],[])
        raw['relation_extraction']['works_for']=[row]
        self.assertEqual(normalize_described_relations(raw,{'works_for':description}),result)

    def described_predictor(self,left,right,score=.9,target_first=False):
        from ai_extract.services.gliner_entity_service import REFERENCE_DESCRIPTIONS
        source_start=len(right)+2 if target_first else 0
        target_start=0 if target_first else len(left)+2
        row=dict(head=dict(start=source_start,end=source_start+len(left),text=left,confidence=score),
            tail=dict(start=target_start,end=target_start+len(right),text=right,confidence=.85))
        model=Mock()
        alias='references_table: '+REFERENCE_DESCRIPTIONS['references_table']
        model.extract.return_value={'relation_extraction':{alias:[row],'references_table':[]}}
        return model,row

    def test_described_extraction_keeps_span_scores_separate_from_edge_confidence(self):
        from ai_extract.services.gliner_entity_service import infer_reference_pairs,REFERENCE_DESCRIPTIONS
        left='See the table.';right='A table heading.'
        model,row=self.described_predictor(left,right)
        r=infer_reference_pairs([dict(id='pair',left=left,right=right)],model)
        d=r['decisions'][0]
        self.assertEqual(d['status'],'reference')
        self.assertIsNone(d['relation_confidence'])
        self.assertEqual(d['references'][0]['endpoint_confidences'],dict(head=.9,tail=.85))
        self.assertIsNone(d['references'][0]['relation_confidence'])
        self.assertEqual(d['references'][0]['evidence']['source_span']['text'],left)
        definitions=model.create_schema.return_value.relations.call_args.args[0]
        self.assertEqual({k:v['description'] for k,v in definitions.items()},REFERENCE_DESCRIPTIONS)
        self.assertTrue(model.extract.call_args.kwargs['include_spans'])
        self.assertTrue(model.extract.call_args.kwargs['include_confidence'])

    def test_described_reference_preserves_source_order_and_direction(self):
        from ai_extract.services.gliner_entity_service import infer_reference_pairs
        left='See above.';right='A table heading.'
        model,row=self.described_predictor(left,right,target_first=True)
        d=infer_reference_pairs([dict(id='pair',left=left,right=right,target_first=True)],model)['decisions'][0]
        self.assertEqual(model.extract.call_args.args[0],right+'\n\n'+left)
        self.assertEqual(d['references'][0]['evidence']['source_span'],dict(start=0,end=len(left),text=left))
        self.assertEqual(d['references'][0]['evidence']['target_span'],dict(start=0,end=len(right),text=right))

    def test_described_reference_rejects_wrong_region_or_low_span_confidence(self):
        from ai_extract.services.gliner_entity_service import infer_reference_pairs
        left='See the table.';right='A table heading.'
        for wrong_side in (False,True):
            model,row=self.described_predictor(left,right,score=.4 if not wrong_side else .9)
            if wrong_side:row['tail']=dict(row['head'])
            d=infer_reference_pairs([dict(id='pair',left=left,right=right)],model)['decisions'][0]
            self.assertEqual(d['status'],'unresolved')
            self.assertEqual(d['references'],[])
            self.assertEqual(len(d['rejected_instances']),1)

    def test_described_reference_empty_result_stays_unknown(self):
        from ai_extract.services.gliner_entity_service import infer_reference_pairs
        model=Mock();model.extract.return_value={'relation_extraction':{}}
        d=infer_reference_pairs([dict(id='pair',left='A heading',right='Site Map')],model)['decisions'][0]
        self.assertEqual(d['status'],'unresolved')
        self.assertEqual(d['references'],[])
        self.assertIsNone(d['relation_confidence'])

    def test_previous_instruction_cannot_target_a_later_or_unknown_source(self):
        from ai_extract.services.gliner_entity_service import infer_reference_pairs,REFERENCE_DESCRIPTIONS
        for target_first in (False,True):
            left='Use the method above.';right='Bend the strap up.'
            model,row=self.described_predictor(left,right,target_first=target_first)
            alias='references_previous_instruction: '+REFERENCE_DESCRIPTIONS['references_previous_instruction']
            model.extract.return_value={'relation_extraction':{alias:[row]}}
            result=infer_reference_pairs([dict(id='pair',left=left,right=right,target_first=target_first)],model)
            decision=result['decisions'][0]
            self.assertEqual(decision['status'],'reference' if target_first else 'unresolved')
            if not target_first:
                self.assertEqual(decision['rejected_instances'][0]['reason'],
                                 'previous_instruction_requires_known_earlier_target')

    def test_display_and_skipped_sources_never_reach_ner_but_child_ocr_does(self):
        request=Mock(return_value=dict(entities=[],relations=[]))
        leaves=[dict(chunk_id='ui',text='[Technical image page 1]',
                    structure_metadata=dict(display_only=True,text_source='visual_display_label')),
                dict(chunk_id='old-ui',text='[Technical image page 1]',
                    structure_metadata=dict(ocr_status='skipped')),
                dict(chunk_id='real-child',text='RC8 Controller',
                    structure_metadata=dict(ocr_status='completed',surya_parent_region_id='figure'))]
        result=extract_graph_entities(dict(leaves=leaves),request)
        self.assertEqual(request.call_args.args[0],[dict(id='real-child',text='RC8 Controller')])
        self.assertEqual(set(result['source_coverage']['excluded_ocr_ids']),{'ui','old-ui'})

    def test_ner_schema_does_not_request_contract_or_numeric_attribute_labels(self):
        predictor=Mock(device='cuda:0')
        predictor.inference.return_value=[[]]
        infer_entities([dict(id='source',text='RC8 Controller specifications')],predictor)
        from ai_extract.services.gliner_entity_service import ENTITY_LABELS
        self.assertEqual(set(ENTITY_LABELS.values()),{'ENTITY'})
        self.assertNotIn('contract number',predictor.inference.call_args.kwargs['labels'])
        self.assertNotIn('tax identification number',predictor.inference.call_args.kwargs['labels'])

    def structure_classifier(self,**scores):
        predictor=Mock()
        predictor.classify_text.return_value={'source_relation':[
            dict(label=k,confidence=v) for k,v in scores.items()]}
        opposite=dict(scores,supported=scores.get('disputed',0),disputed=scores.get('supported',0))
        reverse={'source_relation':[dict(label=k,confidence=v) for k,v in opposite.items()]}
        predictor.classify_text.side_effect=[predictor.classify_text.return_value]*2+[reverse]*2
        return predictor

    def test_structure_review_uses_whole_sources_without_ner_and_keeps_endpoint_roles(self):
        from ai_extract.services.gliner_entity_service import infer_structure_pairs
        predictor=self.structure_classifier(supported=.98,disputed=.01,uncertain=.01)
        pair=dict(id='next',kind='NEXT',left='First source',right='Second source')
        result=infer_structure_pairs([pair],predictor)['decisions'][0]
        self.assertEqual(result['status'],'supported')
        predictor.inference.assert_not_called()
        inputs=[c.args[0] for c in predictor.classify_text.call_args_list]
        for text in inputs[:2]:
            self.assertIn('NEXT(A, B)',text)
            self.assertIn('Source A quotation: First source',text)
            self.assertIn('Source B quotation: Second source',text)
            self.assertNotIn('[Source boundary]',text)
        self.assertLess(inputs[0].index('Source A quotation:'),inputs[0].index('Source B quotation:'))
        self.assertGreater(inputs[1].index('Source A quotation:'),inputs[1].index('Source B quotation:'))
        self.assertIn('Source A quotation: Second source',inputs[2])
        self.assertIn('Source B quotation: First source',inputs[2])

    def test_structure_review_retains_low_scores_without_confirming_relation(self):
        from ai_extract.services.gliner_entity_service import infer_structure_pairs
        predictor=self.structure_classifier(supported=.85,disputed=.05,uncertain=.1)
        decision=infer_structure_pairs([dict(id='pair',kind='NEXT',left='First',right='Second')],predictor)['decisions'][0]
        self.assertEqual(decision['status'],'unresolved')
        self.assertEqual(decision['supported'],.85)
        self.assertEqual(decision['reason'],'below_acceptance_threshold')
        self.assertEqual(decision['raw_outputs'][0],predictor.classify_text.return_value)

    def test_structure_review_can_dispute_and_distinguishes_uncertainty(self):
        from ai_extract.services.gliner_entity_service import infer_structure_pairs
        pair=dict(id='pair',kind='CONTAINS',left='Heading',right='Paragraph')
        predictor=self.structure_classifier(supported=.01,disputed=.98,uncertain=.01)
        self.assertEqual(infer_structure_pairs([pair],predictor)['decisions'][0]['status'],'disputed')
        predictor=self.structure_classifier(supported=.01,disputed=.01,uncertain=.98)
        decision=infer_structure_pairs([pair],predictor)['decisions'][0]
        self.assertEqual(decision['status'],'unresolved')
        self.assertEqual(decision['reason'],'model_uncertain')

    def test_structure_review_presentation_disagreement_abstains(self):
        from ai_extract.services.gliner_entity_service import infer_structure_pairs
        predictor=self.structure_classifier(supported=.98,disputed=.01,uncertain=.01)
        predictor.classify_text.side_effect=[predictor.classify_text.return_value,
            self.structure_classifier(supported=.01,disputed=.98,uncertain=.01).classify_text.return_value]*2
        decision=infer_structure_pairs([dict(id='pair',kind='NEXT',left='First',right='Second')],predictor)['decisions'][0]
        self.assertEqual(decision['status'],'unresolved')
        self.assertEqual(decision['reason'],'presentation_disagreement')

    def test_structure_review_rejects_blanket_support_in_both_directions(self):
        from ai_extract.services.gliner_entity_service import infer_structure_pairs
        predictor=self.structure_classifier(supported=.999,disputed=.0005,uncertain=.0005)
        predictor.classify_text.side_effect=None
        decision=infer_structure_pairs([dict(id='pair',kind='NEXT',left='First',right='Second')],predictor)['decisions'][0]
        self.assertEqual(decision['status'],'unresolved')
        self.assertEqual(decision['reason'],'relation_direction_not_distinguished')

    def test_structure_http_adapter_accepts_classifier_and_rejects_wrong_checkpoint(self):
        from ai_extract.services.gliner_entity_service import request_structure_pairs,structure_review_policy
        from ai_extract import STRUCTURE_REVIEW_SCHEMA
        policy=structure_review_policy()
        pair=dict(id='pair',kind='NEXT',left='First',right='Second')
        payload=dict(model=policy['model'],revision=policy['revision'],schema=STRUCTURE_REVIEW_SCHEMA,
            inference_policy=policy,decisions=[dict(id='pair',status='unresolved')])
        with patch('requests.post') as post:
            post.return_value.json.return_value=payload
            self.assertEqual(request_structure_pairs([pair]),payload)
            self.assertTrue(post.call_args.args[0].endswith('/rag/gliner_structure'))
            for field in ('model','revision','schema','inference_policy'):
                post.return_value.json.return_value=dict(payload,**{field:'wrong'})
                with self.assertRaises(ValueError):request_structure_pairs([pair])

    def test_structure_review_rejects_missing_scores_and_invalid_sources(self):
        from ai_extract.services.gliner_entity_service import infer_structure_pairs
        predictor=self.structure_classifier(supported=.98)
        with self.assertRaises(ValueError):
            infer_structure_pairs([dict(id='pair',kind='NEXT',left='First',right='Second')],predictor)
        for changes in ({'left':''},{'right':'a'*1801},{'kind':'invented'}):
            with self.assertRaises(ValueError):
                infer_structure_pairs([dict(dict(id='pair',kind='NEXT',left='First',right='Second'),**changes)],predictor)

    def test_identity_pairs_require_exact_names_in_both_sources(self):
        from ai_extract.services.gliner_entity_service import infer_identity_pairs,CONTEXT_SEPARATOR
        predictor=Mock()
        left='ABC supplies components.';right='ABC Ltd supplies components.'
        offset=len(left)+len(CONTEXT_SEPARATOR)
        predictor.inference.return_value=([[]],[[dict(relation='same entity as',score=.96,
            head=dict(start=0,end=3,text='ABC'),tail=dict(start=offset,end=offset+7,text='ABC Ltd'))]])
        result=infer_identity_pairs([dict(id='1',left=dict(name='ABC',context=left),right=dict(name='ABC Ltd',context=right))],predictor)
        self.assertEqual(result['decisions'][0]['status'],'same')
        predictor.inference.return_value[1][0][0]['tail']['start']=0
        result=infer_identity_pairs([dict(id='1',left=dict(name='ABC',context=left),right=dict(name='ABC Ltd',context=right))],predictor)
        self.assertEqual(result['decisions'][0]['status'],'unresolved')

    def test_identity_name_absent_from_source_never_reaches_model(self):
        from ai_extract.services.gliner_entity_service import infer_identity_pairs
        predictor=Mock()
        with self.assertRaises(ValueError):
            infer_identity_pairs([dict(id='1',left=dict(name='ABC',context='Different company.'),
                                      right=dict(name='ABC',context='ABC supplies components.'))],predictor)
        predictor.inference.assert_not_called()

    def test_identity_retains_low_scores_without_accepting_identity(self):
        from ai_extract.services.gliner_entity_service import infer_identity_pairs, CONTEXT_SEPARATOR
        predictor=Mock();left='ABC supplies parts.';right='ABC makes parts.'
        offset=len(left)+len(CONTEXT_SEPARATOR)
        head=dict(start=0,end=3,text='ABC');tail=dict(start=offset,end=offset+3,text='ABC')
        predictor.inference.return_value=([[head,tail]],[[dict(relation='same entity as',score=.12,head=head,tail=tail)]])
        result=infer_identity_pairs([dict(id='low',left=dict(name='ABC',context=left),right=dict(name='ABC',context=right))],predictor)['decisions'][0]
        self.assertEqual(result['same'],.12)
        self.assertEqual(result['status'],'unresolved')
        self.assertEqual(result['reason'],'low_confidence')
        self.assertEqual(result['grounded_relations'],1)

    def test_identity_missing_endpoint_is_distinct_from_model_difference(self):
        from ai_extract.services.gliner_entity_service import infer_identity_pairs
        predictor=Mock();predictor.inference.return_value=([[dict(start=0,end=3,text='ABC')]],[[]])
        result=infer_identity_pairs([dict(id='missing',left=dict(name='ABC',context='ABC'),right=dict(name='ABC',context='ABC'))],predictor)['decisions'][0]
        self.assertEqual(result['status'],'unresolved')
        self.assertEqual(result['reason'],'endpoint_not_detected')
        self.assertEqual(result['detected_endpoints'],dict(left=True,right=False))

    def test_identity_incomplete_entity_batch_is_rejected(self):
        from ai_extract.services.gliner_entity_service import infer_identity_pairs
        predictor=Mock();predictor.inference.return_value=([],[[]])
        with self.assertRaisesRegex(ValueError,'Incomplete identity'):
            infer_identity_pairs([dict(id='missing',left=dict(name='ABC',context='ABC'),right=dict(name='ABC',context='ABC'))],predictor)

    def test_ner_keeps_section_titles_when_only_retrieval_flag_is_set(self):
        request=Mock(return_value=dict(entities=[],relations=[]))
        tree=dict(leaves=[dict(chunk_id='header',text='Robot Vision',retrieval_excluded=True,
                              structure_metadata={'ocr_status':'completed'}),
                          dict(chunk_id='bad',text='Unreliable OCR',structure_metadata={'ocr_status':'rejected'})])
        result=extract_graph_entities(tree,request)
        self.assertEqual(request.call_args.args[0],[dict(id='header',text='Robot Vision')])
        self.assertEqual(result['source_coverage']['completed_ids'],['header'])
        self.assertEqual(result['source_coverage']['without_entity_ids'],['header'])
        self.assertEqual(result['source_coverage']['excluded_ocr_ids'],['bad'])

    def test_surya_headers_footers_remain_ner_sources_and_only_exclude_mentions(self):
        from ai_extract.services.gliner_entity_service import source_text_eligible
        request=Mock(return_value=dict(entities=[],relations=[]))
        leaves=[dict(chunk_id='header',text='DENSO',layout_type='PageHeader'),
                dict(chunk_id='footer',text='Contact',layout_type='page_footer'),
                dict(chunk_id='section',text='Robot Viewer',layout_type='SectionHeader'),
                dict(chunk_id='body',text='The header is part of this product.',layout_type='Text')]
        result=extract_graph_entities(dict(leaves=leaves),request)
        self.assertEqual([u['id'] for u in request.call_args.args[0]],['header','footer','section','body'])
        self.assertEqual(result['source_coverage']['excluded_layout_ids'],[])
        self.assertEqual(result['source_coverage']['mention_excluded_ids'],['footer','header'])
        self.assertEqual(result['source_coverage']['excluded_ocr_ids'],[])
        self.assertTrue(source_text_eligible(leaves[-1]))

    def test_only_mentions_from_page_headers_and_footers_are_withheld(self):
        leaves=[dict(chunk_id=kind,text='ABC',layout_type=kind,node_id='root',ancestor_ids=['root'])
                for kind in ('page_header','page_footer','title','paragraph')]
        tree=dict(nodes=[dict(id='root',kind='document',parent_id=None)],leaves=leaves,source_fingerprint='test')
        entities=[dict(key=kind,source_id=kind,source_ids=[kind],start=0,end=3,name='ABC',
                       entity_type='ENTITY',confidence=.9) for kind in ('page_header','page_footer','title','paragraph')]
        tree['knowledge_graph']=dict(entities=entities,relations=[])
        run=SimpleNamespace(pk=1,version_id=2,version=SimpleNamespace(contract=SimpleNamespace(pk=3,company_id=4)))
        _,nodes,edges=projection(run,tree)
        self.assertEqual(sum(n.get('category')=='entity' for n in nodes),4)
        self.assertEqual({e['source_leaf_id'] for e in edges if e['type']=='MENTIONS'},{'title','paragraph'})
    def test_batch_keeps_surya_regions_as_independent_texts(self):
        predictor=Mock(device='cuda:0')
        mention=dict(start=0,end=3,text='ABC',label='entity',score=.9)
        predictor.inference.return_value=[[mention],[mention]]
        result=infer_entities([dict(id='region-a',text='ABC supplier'),
                               dict(id='region-b',text='ABC customer')],predictor)
        self.assertEqual(predictor.inference.call_args.kwargs['texts'],['ABC supplier','ABC customer'])
        self.assertEqual({e['source_id'] for e in result['entities']},{'region-a','region-b'})
        self.assertEqual(len(result['entities']),2)

    def test_enrichment_preserves_sources_and_indexes(self):
        from ai_extract.services.layout_service import enrich_entity_graph
        tree=dict(source_fingerprint='x', nodes=[{'id':'root'}], leaves=[{'chunk_id':'u1','text':'ABC'}],
                  knowledge_graph={'relations':[{'predicate':'references'}],'source_relations':[]})
        run=SimpleNamespace(id=29,status='COMPLETED',total_pages=1,structure=tree,dense_points_count=129)
        extracted=dict(provider='gliner',status='complete',entities=[{'key':'mention'}],errors=[])
        def prepare(run,progress):
            run.structure['knowledge_graph']=dict(tree['knowledge_graph'],**extracted)
        with patch('ai_extract.services.layout_service.prepare_document',side_effect=prepare), \
             patch('ai_extract.services.document_graph_service.sync_graph',return_value={'status':'ready'}) as sync, \
             patch('ai_extract.services.layout_service.layout_repository.save_run') as save:
            enrich_entity_graph(run)
        for key,value in tree['leaves'][0].items():self.assertEqual(run.structure['leaves'][0][key],value)
        self.assertEqual(run.structure['nodes'],tree['nodes'])
        self.assertEqual(run.structure['knowledge_graph']['relations'],tree['knowledge_graph']['relations'])
        self.assertEqual(run.dense_points_count,129)
        self.assertEqual(run.status,'COMPLETED')
        sync.assert_called_once_with(run)
        save.assert_called_once_with(run,['structure'])

    def test_partial_enrichment_does_not_replace_saved_graph(self):
        from ai_extract.services.layout_service import enrich_entity_graph
        tree=dict(source_fingerprint='x')
        run=SimpleNamespace(id=29,status='COMPLETED',total_pages=1,structure=tree)
        with patch('ai_extract.services.layout_service.prepare_document',side_effect=RuntimeError('Model unavailable')), \
             patch('ai_extract.services.document_graph_service.sync_graph') as sync, \
             patch('ai_extract.services.layout_service.layout_repository.save_run') as save:
            with self.assertRaises(RuntimeError):enrich_entity_graph(run)
        self.assertIs(run.structure,tree)
        sync.assert_not_called()
        save.assert_not_called()

    def test_source_offset_is_preserved_without_generating_relations(self):
        text='Công ty ABC không phải thanh toán 100 triệu đồng.'
        start=text.index('ABC')
        predictor=Mock(device='cuda:0')
        predictor.inference.return_value=[[dict(start=start,end=start+3,text='ABC',label='entity',score=.9)]]
        result=infer_entities([dict(id='u1',text=text)],predictor)
        self.assertEqual(result['entities'][0]['name'],'ABC')
        self.assertEqual(result['entities'][0]['start'],start)
        self.assertEqual(result['relations'],[])

    def test_production_ner_adapter_rejects_semantic_relations(self):
        from ai_extract.services.gliner_entity_service import request_entities,ENTITY_SCHEMA,MODEL_ID
        response=dict(model=MODEL_ID,provider='gliner',entity_schema=ENTITY_SCHEMA,
                      extraction_mode='entities_only',entities=[],relations=[])
        with patch('ai_extract.services.gliner_entity_service._request_runtime',return_value=response):
            self.assertEqual(request_entities([])['relations'],[])
            response['relations']=[dict(predicate='supports')]
            with self.assertRaises(ValueError):request_entities([])

    def test_default_ner_does_not_request_or_accept_relation_decoding(self):
        predictor=Mock(device='cuda:0')
        predictor.inference.return_value=[[]]
        result=infer_entities([dict(id='u',text='RC8 supports encoder HUB.')],predictor)
        self.assertEqual(predictor.inference.call_args.kwargs['relations'],[])
        self.assertFalse(predictor.inference.call_args.kwargs['return_relations'])
        self.assertTrue(predictor.inference.call_args.kwargs['flat_ner'])
        self.assertFalse(predictor.inference.call_args.kwargs['multi_label'])
        self.assertEqual(result['extraction_mode'],'entities_only')
        self.assertEqual(result['relations'],[])

    def test_hallucinated_label_span_and_low_confidence_are_rejected(self):
        rows=[dict(start=0,end=3,text='XYZ',label='entity',score=.9),
              dict(start=0,end=3,text='ABC',label='pays_to',score=.9),
              dict(start=0,end=3,text='ABC',label='entity',score=.4)]
        self.assertEqual(validate_mentions('ABC',rows),[])

    def test_same_spelling_at_different_positions_does_not_merge(self):
        leaf=dict(chunk_id='u1',text='ABC và ABC',node_id='root',ancestor_ids=['root'])
        items=[dict(source_id='u1',start=a,end=a+3,name='ABC',entity_type='ENTITY',confidence=.9) for a in (0,7)]
        tree=dict(nodes=[dict(id='root',kind='document',parent_id=None)],leaves=[leaf],source_fingerprint='x')
        tree['knowledge_graph']=extract_graph_entities(tree,lambda batch:dict(entities=items,relations=[]))
        self.assertEqual(len({e['key'] for e in tree['knowledge_graph']['entities']}),2)
        run=SimpleNamespace(pk=1,version_id=2,version=SimpleNamespace(contract=SimpleNamespace(pk=3,company_id=4)))
        _,nodes,edges=projection(run,tree)
        mentions=[n for n in nodes if n.get('category')=='entity']
        self.assertTrue(all('entity_type' not in n for n in mentions))
        self.assertEqual(sum(e['type']=='MENTIONS' for e in edges),2)
        self.assertTrue(all(e['source_leaf_id']=='u1' for e in edges if e.get('origin')=='gliner_source_span'))
        self.assertFalse(any(e['type']=='RELATED' for e in edges))

    def test_ner_cannot_smuggle_generated_semantic_relations(self):
        tree=dict(leaves=[dict(chunk_id='u1',text='ABC pays DEF.')])
        result=extract_graph_entities(tree,lambda batch:dict(entities=[],relations=[dict(predicate='pays_to')]))
        self.assertEqual(result['status'],'unavailable')
        self.assertEqual(result['entities'],[])

    def test_ner_only_rejects_even_allowlisted_and_source_grounded_relations(self):
        source='ABC supports XYZ.'
        items=[dict(source_id='u1',start=a,end=a+3,name=name,entity_type='ENTITY',confidence=.9)
               for a,name in ((0,'ABC'),(13,'XYZ'))]
        link=dict(source_id='u1',from_key=mention_key(items[0]),to_key=mention_key(items[1]),
                  predicate='supports',quote=source,confidence=.9)
        result=extract_graph_entities(dict(leaves=[dict(chunk_id='u1',text=source)]),
            lambda batch:dict(entities=items,relations=[link]))
        self.assertEqual(result['status'],'unavailable')
        self.assertEqual(result['relations'],[])
        self.assertIn('NER-only',result['errors'][0]['reason'])

    def test_relex_cannot_link_entities_from_different_surya_regions(self):
        items=[dict(source_id=source,start=0,end=3,name=name,entity_type='ENTITY',confidence=.9)
               for source,name in (('a','ABC'),('b','XYZ'))]
        link=dict(source_id='a',from_key=mention_key(items[0]),to_key=mention_key(items[1]),
                  predicate='supports',quote='ABC',confidence=.9)
        result=extract_graph_entities(dict(leaves=[dict(chunk_id='a',text='ABC'),dict(chunk_id='b',text='XYZ')]),
            lambda batch:dict(entities=items,relations=[link]))
        self.assertEqual(result['status'],'unavailable')
        self.assertEqual(result['relations'],[])

    def test_relation_survives_projection_with_predicate_confidence_and_source(self):
        source='ABC supports XYZ.'
        items=[dict(source_id='u1',start=a,end=a+3,name=name,entity_type='ENTITY',confidence=.9)
               for a,name in ((0,'ABC'),(13,'XYZ'))]
        link=dict(source_id='u1',from_key=mention_key(items[0]),to_key=mention_key(items[1]),
                  predicate='supports',quote=source,confidence=.9)
        tree=dict(nodes=[dict(id='root',kind='document',parent_id=None)],source_fingerprint='x',
                  leaves=[dict(chunk_id='u1',text=source,node_id='root',ancestor_ids=['root'])])
        tree['knowledge_graph']=dict(entities=[dict(e,key=mention_key(e),source_ids=[e['source_id']]) for e in items],
                                    relations=[dict(link,origin='gliner_relex_source_span')])
        run=SimpleNamespace(pk=1,version_id=2,version=SimpleNamespace(contract=SimpleNamespace(pk=3,company_id=4)))
        _,_,edges=projection(run,tree)
        relation=next(e for e in edges if e['type']=='SUPPORTS')
        self.assertEqual((relation['predicate'],relation['confidence'],relation['source_leaf_id']),('supports',.9,'u1'))
        self.assertEqual(relation['quote'],source)
        self.assertEqual(relation['review_status'],'unreviewed')

    def test_boolean_score_and_self_relations_are_rejected(self):
        source='ABC supports XYZ.'
        items=[dict(source_id='u1',start=a,end=a+3,name=name,entity_type='ENTITY',confidence=.9)
               for a,name in ((0,'ABC'),(13,'XYZ'))]
        link=dict(source_id='u1',from_key=mention_key(items[0]),to_key=mention_key(items[1]),
                  predicate='supports',quote=source,confidence=.9)
        for bad in (dict(link,confidence=True),dict(link,to_key=link['from_key'])):
            with self.subTest(relation=bad):
                result=extract_graph_entities(dict(leaves=[dict(chunk_id='u1',text=source)]),
                    lambda batch:dict(entities=items,relations=[bad]))
                self.assertEqual(result['status'],'unavailable')
                self.assertEqual(result['relations'],[])
        self.assertEqual(validate_mentions('ABC',[dict(start=0,end=3,text='ABC',label='entity',score=True)]),[])

    def test_long_source_is_fully_windowed_without_changing_text(self):
        text=' '.join('word'+str(i) for i in range(500))
        windows=list(text_windows(text))
        self.assertGreater(len(windows),1)
        self.assertEqual(windows[0][0],0)
        self.assertEqual(windows[-1][0]+len(windows[-1][1]),len(text))
        self.assertTrue(all(text[offset:offset+len(part)]==part for offset,part in windows))
        self.assertTrue(all(windows[i+1][0]<windows[i][0]+len(windows[i][1]) for i in range(len(windows)-1)))

    def test_unavailable_runtime_does_not_retry_every_source_batch(self):
        tree=dict(leaves=[dict(chunk_id=str(i),text='ABC') for i in range(40)])
        request=Mock(side_effect=RuntimeError('Model not installed'))
        result=extract_graph_entities(tree,request)
        request.assert_called_once()
        self.assertEqual(result['status'],'unavailable')

    def test_duplicate_sources_are_rejected_before_any_entity_model_call(self):
        request=Mock()
        with self.assertRaisesRegex(ValueError,'Duplicate entity source IDs'):
            extract_graph_entities(dict(leaves=[dict(chunk_id='same',text=text) for text in ('A','B')]),request)
        request.assert_not_called()

    def test_larger_transport_batches_preserve_independent_sources_and_coverage(self):
        tree=dict(leaves=[dict(chunk_id=str(i),text='Source '+str(i)) for i in range(130)])
        request=Mock(side_effect=lambda units:dict(entities=[],relations=[]))
        result=extract_graph_entities(tree,request)
        self.assertEqual([len(call.args[0]) for call in request.call_args_list],[64,64,2])
        self.assertEqual(result['source_coverage']['completed_ids'],[str(i) for i in range(130)])
        self.assertEqual(result['status'],'complete')

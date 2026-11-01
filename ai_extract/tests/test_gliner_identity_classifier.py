from unittest import TestCase
from unittest.mock import Mock
from ai_extract.services.gliner_identity_classifier import classify_pair,identity_text,read_scores,infer_pairs,MODEL_ID,SCHEMA


def output(same,different,uncertain):
    return {'entity_identity':[{'label':label,'confidence':score} for label,score in
        [('same_entity',same),('different_entity',different),('uncertain',uncertain)]]}


def entailment_proof(claim,polarity='affirmed',supported=True):
    from ai_extract.services.gliner_identity_classifier import NLI_SCHEMA,NLI_MODEL_ID,NLI_REVISION,source_claim_digest
    row=dict(contradiction=.01,entailment=.98 if supported else .01,neutral=.01 if supported else .98)
    return dict(id=claim['id'],schema=NLI_SCHEMA,model=NLI_MODEL_ID,revision=NLI_REVISION,
                claim_digest=source_claim_digest(claim),**{k:[dict(row),dict(row)] if k==polarity else [] for k in ('affirmed','negated')})


def kind_proof(claim,direct=True):
    from ai_extract.services.gliner_identity_classifier import RELATION_KIND_LABELS,source_claim_digest
    winner='direct_relation' if direct else 'information_about'
    row={k:.97 if k==winner else .01 for k in RELATION_KIND_LABELS}
    return dict(relation_kind_digest=source_claim_digest(claim),relation_kind_scores=[dict(row),dict(row)])


class GLiNERIdentityClassifierTests(TestCase):
    def test_relation_claim_review_uses_model_scores_in_both_presentations(self):
        from ai_extract.services.gliner_identity_classifier import infer_relation_claims,RELATION_SUPPORT_LABELS
        predictor=Mock()
        def raw(winner):
            return dict(relation_support=[dict(label=k,confidence=.97 if k==winner else .01) for k in RELATION_SUPPORT_LABELS])
        claim=dict(id='r',head='Motor A',tail='Robot B',predicate='part of',source='Motor A is not part of Robot B.')
        role=dict(source_relation_kind=[dict(label=k,confidence=v) for k,v in kind_proof(claim)['relation_kind_scores'][0].items()])
        predictor.classify_text.side_effect=[raw('negated'),raw('negated'),role,role]
        scorer=lambda claims:[entailment_proof(c,'negated') for c in claims]
        result=infer_relation_claims([claim],predictor,scorer)
        self.assertEqual(result['decisions'][0]['status'],'negated')
        self.assertEqual(predictor.classify_text.call_count,4)
        predictor.classify_text.side_effect=[raw('affirmed'),raw('negated'),role,role]
        self.assertEqual(infer_relation_claims([claim],predictor,scorer)['decisions'][0]['status'],'unresolved')

    def test_neutral_source_entailment_blocks_even_confident_classifier(self):
        from ai_extract.services.gliner_identity_classifier import infer_relation_claims,RELATION_SUPPORT_LABELS
        predictor=Mock()
        claim=dict(id='r',head='A',tail='B',predicate='designed by',source='A B')
        support=dict(relation_support=[dict(label=k,confidence=.999 if k=='affirmed' else .0001) for k in RELATION_SUPPORT_LABELS])
        role=dict(source_relation_kind=[dict(label=k,confidence=v) for k,v in kind_proof(claim)['relation_kind_scores'][0].items()])
        predictor.classify_text.side_effect=[support,support,role,role]
        result=infer_relation_claims([claim],predictor,lambda claims:[entailment_proof(c,supported=False) for c in claims])
        self.assertEqual(result['decisions'][0]['status'],'unresolved')
        self.assertEqual(result['decisions'][0]['reason'],'quoted_source_does_not_entail_directed_claim')

    def test_information_about_is_not_a_direct_support_relation(self):
        from ai_extract.services.gliner_identity_classifier import infer_relation_claims,RELATION_SUPPORT_LABELS
        claim=dict(id='r',head='A',tail='B',predicate='supports',source='A provides information about B.')
        support=dict(relation_support=[dict(label=k,confidence=.97 if k=='affirmed' else .01) for k in RELATION_SUPPORT_LABELS])
        role=dict(source_relation_kind=[dict(label=k,confidence=v) for k,v in kind_proof(claim,False)['relation_kind_scores'][0].items()])
        predictor=Mock();predictor.classify_text.side_effect=[support,support,role,role]
        result=infer_relation_claims([claim],predictor,lambda claims:[entailment_proof(c) for c in claims])
        self.assertEqual(result['decisions'][0]['status'],'unresolved')
        self.assertEqual(result['decisions'][0]['reason'],'information_about_or_no_direct_relation')

    def test_source_entailment_rejects_changed_source_incomplete_and_nonfinite_scores(self):
        from ai_extract.services.gliner_identity_classifier import source_entailment_supported
        claim=dict(id='c',head='A',tail='B',predicate='part of',source='A is part of B.')
        proof=entailment_proof(claim)
        self.assertTrue(source_entailment_supported(proof,'affirmed',claim))
        self.assertFalse(source_entailment_supported(proof,'affirmed',dict(claim,source='A and B.')))
        proof['affirmed'].pop()
        self.assertFalse(source_entailment_supported(proof,'affirmed',claim))
        proof=entailment_proof(claim);proof['affirmed'][0]['entailment']=float('nan')
        self.assertFalse(source_entailment_supported(proof,'affirmed',claim))

    def test_passage_link_requires_specific_subject_in_both_orders(self):
        from ai_extract.services.gliner_identity_classifier import infer_passage_links,PASSAGE_LINK_LABELS
        predictor=Mock()
        def raw(score):
            return dict(passage_link=[dict(label=k,confidence=score if k=='specific_subject_link' else (1-score)/3) for k in PASSAGE_LINK_LABELS])
        predictor.classify_text.side_effect=[raw(.99),raw(.99),raw(.78),raw(.99)]
        pairs=[dict(id='positive',source='A named interface',target='This device supports a named interface.'),
               dict(id='generic',source='Generic fragment',target='Another generic fragment')]
        result=infer_passage_links(pairs,predictor)
        self.assertEqual([d['accepted'] for d in result['decisions']],[True,False])
        self.assertEqual(predictor.classify_text.call_count,4)

    def test_relation_claim_review_rejects_absent_source_endpoint(self):
        from ai_extract.services.gliner_identity_classifier import infer_relation_claims
        predictor=Mock()
        with self.assertRaises(ValueError):
            infer_relation_claims([dict(id='r',head='Absent',tail='Robot B',predicate='part of',source='Robot B exists.')],predictor)
        predictor.classify_text.assert_not_called()

    def setUp(self):
        # Isolated parser fixtures; never used as source corpus or benchmark gold.
        self.pair=dict(id='parser-fixture',entity_type='ENTITY',left=dict(name='Model A',context='Model A specifications supplied by the manufacturer.'),
                       right=dict(name='Model A',context='Model A specifications supplied in another source.'))

    def test_input_preserves_both_quotations(self):
        text=identity_text(self.pair)
        self.assertIn(self.pair['left']['context'],text)
        self.assertIn(self.pair['right']['context'],text)
        self.assertTrue(identity_text(self.pair,reverse=True).startswith('Mention 1: Model A'))

    def test_missing_source_name_is_rejected_before_inference(self):
        self.pair['left']['name']='Absent'
        predictor=Mock()
        with self.assertRaises(ValueError):classify_pair(self.pair,predictor)
        predictor.classify_text.assert_not_called()

    def test_same_requires_both_directions(self):
        predictor=Mock();predictor.classify_text.side_effect=[output(.96,.02,.02),output(.02,.96,.02)]
        self.assertEqual(classify_pair(self.pair,predictor)['reason'],'orientation_disagreement')

    def test_low_confidence_stays_unresolved(self):
        predictor=Mock();predictor.classify_text.return_value=output(.7,.2,.1)
        self.assertEqual(classify_pair(self.pair,predictor)['status'],'unresolved')

    def test_uncertain_is_not_converted_to_identity(self):
        predictor=Mock();predictor.classify_text.return_value=output(.02,.01,.97)
        self.assertEqual(classify_pair(self.pair,predictor)['reason'],'model_uncertain')

    def test_preserves_actual_scores_for_accepted_proposal(self):
        self.pair['right']=dict(self.pair['left'])
        reference={'identity_reference':[dict(label=k,confidence=v) for k,v in
                   [('specific_named_entity',.96),('generic_reference',.02),('uncertain',.02)]]}
        predictor=Mock();predictor.classify_text.side_effect=[output(.96,.02,.02),output(.96,.02,.02),reference,reference]
        result=classify_pair(self.pair,predictor)
        self.assertEqual(result['status'],'same')
        self.assertEqual(result['scores_forward']['same_entity'],.96)
        self.assertEqual(result['policy'],'unvalidated_model_proposal_no_graph_write')

    def test_different_product_model_names_follow_the_model_difference(self):
        predictor=Mock();predictor.classify_text.return_value=output(.005,.99,.005)
        result=classify_pair(self.pair,predictor)
        self.assertEqual(result['status'],'different')
        self.assertEqual(result['reason'],'bidirectional_model_decision')

    def test_different_names_rejected_even_if_model_would_approve(self):
        self.pair['right']=dict(name='Model B',context='Model B specifications.')
        predictor=Mock()
        with self.assertRaisesRegex(ValueError,'same entity name'):
            classify_pair(self.pair,predictor)
        predictor.classify_text.assert_not_called()

    def test_context_decision_does_not_require_an_entity_subtype(self):
        predictor=Mock();predictor.classify_text.return_value=output(.96,.02,.02)
        self.assertEqual(classify_pair(self.pair,predictor)['status'],'same')
        self.assertEqual(predictor.classify_text.call_count,2)

    def test_isolated_identical_labels_do_not_establish_identity(self):
        self.pair['left']=dict(name='Company',context='Company')
        self.pair['right']=dict(self.pair['left'])
        predictor=Mock();predictor.classify_text.return_value=output(.005,.005,.99)
        result=classify_pair(self.pair,predictor)
        self.assertEqual(result['status'],'unresolved')
        self.assertEqual(result['reason'],'insufficient_source_context')

    def test_isolated_labels_do_not_establish_difference_either(self):
        self.pair['left']=dict(name='Company',context='Company')
        self.pair['right']=dict(self.pair['left'])
        predictor=Mock();predictor.classify_text.return_value=output(.005,.005,.99)
        result=classify_pair(self.pair,predictor)
        self.assertEqual(result['status'],'unresolved')
        self.assertEqual(result['reason'],'insufficient_source_context')

    def test_confident_model_cannot_merge_name_only_source_quotes(self):
        self.pair['left']=dict(name='Company',context='Company')
        self.pair['right']=dict(name='Company',context='Company.')
        predictor=Mock();predictor.classify_text.return_value=output(.999,.0005,.0005)
        result=classify_pair(self.pair,predictor)
        self.assertEqual(result['status'],'unresolved')
        self.assertEqual(result['reason'],'insufficient_source_context')
        self.assertEqual(result['scores_forward']['same_entity'],.999)
        self.assertEqual(len(result['raw_outputs']),2)

    def test_provenance_targets_the_second_equal_name_in_the_quote(self):
        context='Model A and Model A have different owners.'
        start=context.rindex('Model A')
        self.pair['left']=dict(name='Model A',context=context,
            scope=dict(mention_start=100+start,mention_end=100+start+7,quote_start=100))
        self.assertIn(f'Mention character range 1: [{start}, {start+7})',identity_text(self.pair))
        self.pair['left']['scope']['mention_end']+=1
        with self.assertRaisesRegex(ValueError,'occurrence offsets'):
            classify_pair(self.pair,Mock())

    def test_missing_scores_are_not_invented(self):
        with self.assertRaises(ValueError):read_scores({'entity_identity':[{'label':'same_entity','confidence':.99}]})

    def test_production_adapter_retains_actual_bidirectional_scores_and_model(self):
        predictor=Mock();predictor.classify_text.side_effect=[output(.02,.96,.02),output(.03,.95,.02)]
        result=infer_pairs([self.pair],predictor)
        self.assertEqual(result['model'],MODEL_ID)
        self.assertEqual(result['schema'],SCHEMA)
        row=result['decisions'][0]
        self.assertEqual(row['status'],'different')
        self.assertEqual(row['same'],.02)
        self.assertEqual(row['different'],.96)
        self.assertEqual(len(row['raw_outputs']),2)

    def test_production_adapter_rejects_duplicate_pair_ids_before_model_call(self):
        predictor=Mock()
        with self.assertRaises(ValueError):infer_pairs([self.pair,self.pair],predictor)
        predictor.classify_text.assert_not_called()

    def test_invalid_or_duplicate_scores_are_rejected(self):
        for bad in [float('nan'),float('inf'),True,-.1,1.1]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):read_scores(output(bad,.02,.02))
        data=output(.96,.02,.02);data['entity_identity'][2]['label']='same_entity'
        with self.assertRaises(ValueError):read_scores(data)


class WholeDocumentIdentityTests(TestCase):
    def test_same_entity_name_in_two_documents_has_separate_local_identity(self):
        from copy import deepcopy
        from types import SimpleNamespace
        from ai_extract.services.gliner_identity_classifier import resolve_identities
        original=dict(leaves=[dict(chunk_id='u',text='DENSO manufactures robots.')],
            knowledge_graph=dict(entities=[dict(key='mention',name='DENSO',start=0,end=5,source_ids=['u'])]))
        keys=[];model=Mock()
        for version_id in (10,11):
            tree=deepcopy(original)
            run=SimpleNamespace(pk=version_id,version_id=version_id,
                version=SimpleNamespace(contract=SimpleNamespace(company_id=1)))
            resolve_identities(run,tree,model)
            keys.append(tree['knowledge_graph']['entities'][0]['canonical_key'])
        self.assertNotEqual(keys[0],keys[1])
        model.assert_not_called()

    def test_remote_positive_decision_cannot_merge_name_only_occurrences(self):
        from ai_extract.services.gliner_identity_classifier import resolve_identities
        run,tree=self.fixture()
        for leaf in tree['leaves']:leaf['text']='Model A'
        result=resolve_identities(run,tree,self.model)
        self.assertEqual(result['canonical_entities'],2)
        self.assertEqual(result['linked'],0)
        self.assertEqual(result['rejected'][0]['reason'],'insufficient_source_context')

    def test_same_name_in_one_source_preserves_each_occurrence_context_offset(self):
        from ai_extract.services.gliner_identity_classifier import resolve_identities
        run,tree=self.fixture()
        text='Model A is installed here; Model A is installed elsewhere.'
        tree['leaves']=[dict(chunk_id='a',text=text,page_number=1)]
        for i,e in enumerate(tree['knowledge_graph']['entities']):
            e.update(source_ids=['a'],start=0 if i==0 else text.rindex('Model A'))
        result=resolve_identities(run,tree,lambda pairs:dict(decisions=[dict(
            id=p['id'],status='unresolved',same=.5,different=.1) for p in pairs]))
        pair=result['pairs'][0]
        self.assertEqual({pair['left']['scope']['mention_start'],pair['right']['scope']['mention_start']},
            {0,text.rindex('Model A')})

    def test_accepted_merge_reduces_graph_nodes_and_preserves_each_source_span(self):
        from ai_extract.services.gliner_identity_classifier import resolve_identities
        from ai_extract.services.document_graph_service import projection
        run,tree=self.fixture()
        run.version.contract.pk=1
        tree['nodes']=[dict(id='root',parent_id=None,kind='document',title='Source')]
        for leaf in tree['leaves']:leaf.update(node_id='root',ancestor_ids=['root'])
        for entity in tree['knowledge_graph']['entities']:entity.update(start=0,end=7,confidence=.97)
        resolve_identities(run,tree,self.model)
        _,nodes,edges=projection(run,tree)
        graph_entities=[n for n in nodes if n['category']=='entity']
        self.assertEqual(len(graph_entities),1)
        self.assertEqual(graph_entities[0]['occurrence_count'],2)
        self.assertNotIn('entity_type',graph_entities[0])
        mentions=[e for e in edges if e['type']=='MENTIONS']
        self.assertEqual(len(mentions),2)
        self.assertEqual({e['source_leaf_id'] for e in mentions},{'a','b'})
        self.assertEqual({e['occurrence_key'] for e in mentions},{'e0','e1'})
        self.assertTrue(all(e['from_char_start']==0 and e['from_char_end']==7 for e in mentions))

    def embedding(self,texts):
        return [[1.]+[0.]*1023 for text in texts]
    def fixture(self,count=2):
        from types import SimpleNamespace
        run=SimpleNamespace(pk=10,version_id=2,version=SimpleNamespace(contract=SimpleNamespace(company_id=1)))
        leaves=[dict(chunk_id=chr(97+i),text='Model A operates under condition '+str(i),
            page_number=i+1,reading_stream_id='stream-'+str(i),bbox=[0,0,100,100]) for i in range(count)]
        entities=[dict(key='e'+str(i),name='Model A',entity_type='ENTITY',source_ids=[l['chunk_id']])
                  for i,l in enumerate(leaves)]
        return run,dict(source_fingerprint='test-source',leaves=leaves,knowledge_graph=dict(entities=entities))

    def model(self,pairs):
        return dict(decisions=[dict(id=p['id'],status='same',same=.97,different=.01) for p in pairs])

    def test_cross_page_stream_mentions_merge_before_graph_without_merging_source_conditions(self):
        from copy import deepcopy
        from ai_extract.services.gliner_identity_classifier import resolve_identities
        run,tree=self.fixture();original=deepcopy(tree['leaves'])
        result=resolve_identities(run,tree,self.model)
        entities=tree['knowledge_graph']['entities']
        self.assertEqual(entities[0]['canonical_key'],entities[1]['canonical_key'])
        self.assertEqual(tree['leaves'],original)
        self.assertEqual({s['scope']['reading_stream_id'] for s in result['source_mentions']},{'stream-0','stream-1'})
        self.assertEqual(result['mentions'],2)

    def test_equal_names_cannot_merge_without_positive_model_decision(self):
        from ai_extract.services.gliner_identity_classifier import resolve_identities
        run,tree=self.fixture()
        result=resolve_identities(run,tree,lambda pairs:dict(decisions=[dict(
            id=p['id'],status='unresolved',same=.7,different=.1) for p in pairs]))
        self.assertEqual(result['linked'],0)
        self.assertEqual(result['canonical_entities'],2)

    def test_different_names_never_reach_identity_model(self):
        from ai_extract.services.gliner_identity_classifier import resolve_identities
        run,tree=self.fixture()
        tree['leaves'][1]['text']='Alternate model name operates under condition 1'
        tree['knowledge_graph']['entities'][1]['name']='Alternate model name'
        model=Mock()
        result=resolve_identities(run,tree,model)
        model.assert_not_called()
        self.assertEqual(result['canonical_entities'],2)
        self.assertEqual(result['pairs'],[])

    def test_model_difference_blocks_transitive_merge(self):
        from ai_extract.services.gliner_identity_classifier import resolve_identities
        run,tree=self.fixture(3)
        def model(pairs):
            result=[]
            for p in pairs:
                different={p['left']['scope']['source_id'],p['right']['scope']['source_id']}=={'a','c'}
                result.append(dict(id=p['id'],status='different' if different else 'same',
                    same=.01 if different else .97,different=.97 if different else .01))
            return dict(decisions=result)
        resolve_identities(run,tree,model)
        self.assertNotEqual(tree['knowledge_graph']['entities'][0]['canonical_key'],
                            tree['knowledge_graph']['entities'][2]['canonical_key'])

    def test_missing_model_decision_cannot_publish_canonical_assignments(self):
        from ai_extract.services.gliner_identity_classifier import resolve_identities
        run,tree=self.fixture()
        with self.assertRaises(ValueError):resolve_identities(run,tree,lambda pairs:dict(decisions=[]))
        self.assertNotIn('canonical_key',tree['knowledge_graph']['entities'][0])

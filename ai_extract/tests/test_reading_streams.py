from types import SimpleNamespace
from unittest.mock import Mock, patch
from django.test import SimpleTestCase
from ai_extract.services.document_structure_service import build_document_structure
from ai_extract.services.document_graph_service import projection
from ai_extract.services.gliner_entity_service import (
    infer_entities, extract_graph_entities, document_context_units, enrich_document_relations,CONTEXT_SEPARATOR,
)


def region(identity,text,rank,kind='paragraph',page=1,parent=None,level=None):
    metadata=dict(chunk_policy='surya_layout_region',layout_source='surya',surya_region_id=identity,
        reading_order=rank,reading_order_source='surya_full_page_model',reading_order_scope=parent or 'page')
    if parent:metadata['surya_parent_region_id']=parent
    if level:metadata.update(heading_level=level,heading_source='surya_html')
    return dict(chunk_id=identity,text=text,layout_type=kind,page_number=page,image_path=f'page{page}.png',
                bbox=[0,0,500,500],structure_metadata=metadata)


def project(tree):
    run=SimpleNamespace(pk=1,version_id=2,version=SimpleNamespace(contract=SimpleNamespace(pk=3,company_id=4)))
    return projection(run,dict(tree,source_fingerprint='test'))


def run_for(tree):
    return SimpleNamespace(pk=1,version_id=2,structure=tree,
        version=SimpleNamespace(contract=SimpleNamespace(pk=3,company_id=4)))


class ReadingStreamsTests(SimpleTestCase):
    def test_surya_order_is_preserved_without_calling_gliner(self):
        from ai_extract.services.document_structure_service import prepare_reading
        tree=build_document_structure([region('a','Second paragraph.',0),region('b','First paragraph.',1)])
        tree['source_fingerprint']='source'
        original={l['chunk_id']:l['structure_metadata']['reading_order'] for l in tree['leaves']}
        tree['structure_review']=dict(status='complete',relations={'old':dict(status='supported')})
        with patch('ai_extract.services.document_graph_service.review_surya_relations') as review:
            prepare_reading(run_for(tree))
        review.assert_not_called()
        self.assertNotIn('structure_review',tree)
        self.assertEqual(tree['reading_analysis']['provider'],'surya_source')
        self.assertEqual(tree['reading_analysis']['semantic_review'],'disabled')
        first,second=tree['leaves']
        self.assertEqual(first['reading_stream_id'],second['reading_stream_id'])
        self.assertLess(first['reading_stream_position'],second['reading_stream_position'])
        self.assertTrue(tree['reading_analysis']['source_order_preserved'])
        self.assertEqual({l['chunk_id']:l['structure_metadata']['reading_order'] for l in tree['leaves']},original)

    def test_source_preparation_keeps_surya_next(self):
        from ai_extract.services.document_structure_service import prepare_reading
        tree=build_document_structure([region('a','First paragraph.',0),region('b','Second paragraph.',1)])
        tree['source_fingerprint']='test'
        prepare_reading(run_for(tree))
        self.assertEqual(len(tree['reading_streams']),1)
        self.assertTrue(any(e['type']=='NEXT' for e in project(tree)[2]))
        self.assertEqual(tree['reading_analysis']['new_reading_edges'],0)

    def test_model_analysis_preserves_image_text_parent_source(self):
        from ai_extract.services.document_structure_service import prepare_reading
        tree=build_document_structure([region('image','A figure',0,'figure'),region('inside','A label',0,parent='image')])
        tree['source_fingerprint']='test'
        prepare_reading(run_for(tree))
        by_source={l['source_chunk_id']:l for l in tree['leaves']}
        self.assertEqual(by_source['inside']['parent_reading_stream_id'],by_source['image']['reading_stream_id'])

    def test_region_branch_and_supplement_do_not_pollute_main_next(self):
        rows=[region('h','A heading',0,'title',level=1),region('a','Before figure.',1),
              region('picture','',2,'figure'),region('inside','Text in figure.',0,parent='picture'),
              region('after','After figure.',3),region('footer','A footer',4,'page_footer')]
        tree=build_document_structure(rows)
        by_source={l['source_chunk_id']:l for l in tree['leaves']}
        self.assertEqual(by_source['inside']['parent_reading_stream_id'],by_source['picture']['reading_stream_id'])
        self.assertNotEqual(by_source['footer']['reading_stream_id'],by_source['a']['reading_stream_id'])
        self.assertLess(by_source['picture']['reading_document_position'],by_source['inside']['reading_document_position'])
        self.assertLess(by_source['inside']['reading_document_position'],by_source['after']['reading_document_position'])
        _,nodes,edges=project(tree)
        blocks={n['key']:n for n in nodes if n.get('category')=='layout_block'}
        for edge in [e for e in edges if e['type']=='NEXT']:
            a,b=blocks[edge['from_key']],blocks[edge['to_key']]
            self.assertEqual(a['reading_stream_id'],b['reading_stream_id'])
            self.assertNotEqual(a['reading_stream_kind'],'page_footer')

    def test_heading_ownership_survives_page_boundary_without_paragraph_continuation(self):
        tree=build_document_structure([region('title','Payment',0,'title',level=1),
                                       region('a','A condition',1),region('b','Another clause',0,page=2)])
        a,b=[l for l in tree['leaves'] if l['source_chunk_id'] in {'a','b'}]
        self.assertEqual(a['node_id'],b['node_id'])
        self.assertNotEqual(a['reading_stream_id'],b['reading_stream_id'])
        _,_,edges=project(tree)
        self.assertFalse(any(e['type']=='REFERENCES' for e in edges))
        self.assertTrue(all(e.get('origin')!='semantic_continuation' for e in edges))

    def test_unknown_ranks_and_cyclic_parent_scopes_never_authorize_next(self):
        for rows in [[region('a','A',0),region('b','B',0)],
                     [region('a','A',0,parent='b'),region('b','B',0,parent='a')]]:
            tree=build_document_structure(rows)
            self.assertTrue(all(s['status']=='unresolved' for s in tree['reading_streams']))
            self.assertEqual(len(tree['leaves']),2)
            self.assertFalse(any(e['type']=='NEXT' for e in project(tree)[2]))

    def test_table_keeps_separate_owner_and_native_grid(self):
        row=region('table','A table',0,'table')
        row['structure_metadata']['table_rows']=[['Item','Amount'],['A','10']]
        tree=build_document_structure([row])
        leaf=tree['leaves'][0]
        self.assertEqual(next(n for n in tree['nodes'] if n['id']==leaf['node_id'])['kind'],'table')
        edges=project(tree)[2]
        self.assertEqual(sum(e['type']=='HAS_ROW' for e in edges),2)
        self.assertEqual(sum(e['type']=='HAS_CELL' for e in edges),4)
        nodes=project(tree)[1]
        tables=[n for n in nodes if n.get('kind')=='table' and n.get('category')=='structure']
        self.assertEqual(len(tables),1)
        self.assertFalse(any(n.get('category')=='table' for n in nodes))
        owner=tables[0]['key']
        self.assertTrue(any(e['type']=='HAS_ROW' and e['from_key']==owner for e in edges))
        self.assertEqual(sum(e['type']=='HAS_EVIDENCE' and e['from_key']==owner for e in edges),1)


class ContextRelationTests(SimpleTestCase):
    def test_http_batches_never_split_a_competing_target_group(self):
        from ai_extract.services.gliner_entity_service import REFERENCE_SCHEMA
        leaves=[dict(chunk_id='s'+str(i),text='Source '+str(i)) for i in range(19)]
        leaves.extend(dict(chunk_id='t'+str(i),text='Target '+str(i)) for i in range(7))
        selected=[dict(from_id='s'+str(i),from_offset=0,from_text='Source '+str(i),
                       to_id='t'+str(j),to_offset=0,to_text='Target '+str(j))
                  for i in range(19) for j in range(7)]
        def respond(pairs):
            for pair in pairs:
                ids={c['pair_id'] for c in pair['context_group']['candidates']}
                received={p['id'] for p in pairs if p['context_group']['id']==pair['context_group']['id']}
                self.assertEqual(ids,received)
            return dict(schema=REFERENCE_SCHEMA,decisions=[dict(id=p['id'],status='unresolved',references=[]) for p in pairs])
        request=Mock(side_effect=respond)
        with patch('ai_extract.services.gliner_entity_service.document_context_units',
                   return_value=([],dict(selected_pairs=selected,candidate_pairs=len(selected)))), \
             patch.dict('os.environ',{'GLINER_REFERENCE_GROUP_CANDIDATES':'7'}):
            result=enrich_document_relations(dict(leaves=leaves),dict(status='complete',source_relations=[]),request)
        self.assertEqual([len(c.args[0]) for c in request.call_args_list],[126,7])
        self.assertEqual(result['document_relations']['metrics']['completed'],133)

    def test_reference_request_keeps_known_source_order(self):
        tree=dict(leaves=[dict(chunk_id='a',text='See earlier content.',reading_document_position=5),
            dict(chunk_id='b',text='Earlier content.',reading_document_position=2)])
        old=dict(status='complete',entities=[],relations=[],source_relations=[])
        request=Mock(side_effect=self.reference_response)
        enrich_document_relations(tree,old,request,embedding=lambda texts:[[1.,0.] for _ in texts])
        pairs=request.call_args.args[0]
        self.assertEqual(pairs[0]['left'],'See earlier content.')
        self.assertTrue(pairs[0]['target_first'])
        self.assertFalse(pairs[1]['target_first'])

    def test_reference_proposal_rejects_unquoted_endpoint_even_at_high_score(self):
        tree=dict(leaves=[dict(chunk_id='a',text='See earlier content.'),dict(chunk_id='b',text='Earlier content.')])
        old=dict(status='complete',entities=[],relations=[],source_relations=[])
        def invalid(pairs):
            result=self.reference_response(pairs,True)
            result['decisions'][0]['references'][0]['evidence']['target_span']['text']='Invented target'
            return result
        result=enrich_document_relations(tree,old,invalid,embedding=lambda texts:[[1.,0.] for _ in texts])
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['source_relations'],[])
        self.assertEqual(result['document_relations']['reference_proposals'],[])
        self.assertIn('not grounded',result['document_relations']['errors'][0]['reason'])

    def reference_response(self,pairs,accept_first=False):
        from ai_extract.services.gliner_entity_service import REFERENCE_SCHEMA
        def selection(p):
            group=p['context_group']
            scores={c['pair_id']:.001 for c in group['candidates']}
            scores.update(no_reference=.001);scores[p['id']]=.99
            return dict(group_id=group['id'],selected_pair_id=p['id'],votes=[scores,dict(scores)])
        return dict(schema=REFERENCE_SCHEMA,decisions=[dict(id=p['id'],
            status='reference' if accept_first and p['id']=='0' else 'unresolved',
            target_selection=selection(p),
            relation_confidence=None,references=[dict(type='references_table',relation_confidence=None,
                endpoint_confidences=dict(head=.98,tail=.97),evidence=dict(
                source_span=dict(start=0,end=len(p['left']),text=p['left']),
                target_span=dict(start=0,end=len(p['right']),text=p['right'])))] if accept_first and p['id']=='0' else []) for p in pairs])

    def test_reference_proposal_preserves_sources_without_adding_entity_nodes(self):
        a,b,unit,result=self.pair()
        result['relations'].append(dict(result['relations'][0],predicate='depends_on'))
        tree=dict(nodes=[dict(id='root',kind='document',parent_id=None)],leaves=[
            dict(chunk_id='a',text=a,page_number=1,node_id='root',ancestor_ids=['root']),
            dict(chunk_id='b',text=b,page_number=8,node_id='root',ancestor_ids=['root'])])
        old=extract_graph_entities(tree,lambda units:dict(entities=[],relations=[],metrics={}))
        output=enrich_document_relations(tree,old,lambda pairs:self.reference_response(pairs,True),embedding=lambda texts:[[1.,0.] for _ in texts])
        self.assertEqual(output['relations'],[])
        self.assertEqual(output['entities'],[])
        self.assertEqual(output['source_relations'],[])
        self.assertEqual(output['document_relations']['reference_publication'],'withheld_unvalidated_source_pair_model')
        proposals=output['document_relations']['reference_proposals']
        self.assertEqual(len(proposals),1)
        self.assertEqual(proposals[0]['from_id'],'a')
        self.assertEqual(proposals[0]['to_id'],'b')
        tree['knowledge_graph']=output
        references=[e for e in project(tree)[2] if e['type']=='REFERENCES']
        self.assertEqual(references,[])
        # Projection supports grounded source endpoints without entity nodes;
        # publication is withheld separately until model quality is established.
        tree['knowledge_graph']=dict(output,source_relations=proposals)
        references=[e for e in project(tree)[2] if e['type']=='REFERENCES']
        self.assertEqual(len(references),1)
        self.assertEqual(references[0]['quote'],a)
        self.assertEqual(references[0]['target_quote'],b)

    def test_context_does_not_add_entities_without_an_accepted_reference(self):
        a,b,unit,result=self.pair()
        result['relations']=[]
        tree=dict(leaves=[dict(chunk_id='a',text=a),dict(chunk_id='b',text=b)])
        old=dict(status='complete',entities=[],relations=[],source_relations=[])
        output=enrich_document_relations(tree,old,self.reference_response,embedding=lambda texts:[[1.,0.] for _ in texts])
        self.assertEqual(output['entities'],[])
        self.assertEqual(output['source_relations'],[])
        diagnostic=output['document_relations']
        self.assertEqual(diagnostic['quality_status'],'unresolved')
        self.assertEqual(diagnostic['metrics']['proposed_pairs'],0)
        self.assertEqual(diagnostic['metrics']['unresolved_pairs'],diagnostic['metrics']['completed'])
        self.assertEqual(diagnostic['metrics']['missing_pairs'],0)

    def test_remote_previous_instruction_cannot_bypass_source_order_validation(self):
        tree=dict(leaves=[dict(chunk_id='a',text='Use that method.',reading_document_position=1),
                          dict(chunk_id='b',text='Bend the strap.',reading_document_position=2)])
        def response(pairs):
            result=self.reference_response(pairs,True)
            result['decisions'][0]['references'][0]['type']='references_previous_instruction'
            return result
        output=enrich_document_relations(tree,dict(status='complete',source_relations=[]),response,
            embedding=lambda texts:[[1.,0.] for _ in texts])
        self.assertEqual(output['document_relations']['quality_status'],'unavailable')
        self.assertEqual(output['document_relations']['reference_proposals'],[])

    def pair(self):
        a='ABC supports XYZ.';b='XYZ is a specification.'
        unit=dict(id='a',text=a,context_windows=[dict(offset=0,text=a,contexts=[dict(id='b',offset=0,text=b)])])
        predictor=Mock(device='cuda:0')
        tail=len(a)+len(CONTEXT_SEPARATOR)
        mentions=[dict(start=0,end=3,text='ABC',label='entity',score=.95),
                  dict(start=tail,end=tail+3,text='XYZ',label='entity',score=.95)]
        link=dict(head=dict(start=0,end=3,text='ABC',type='entity'),
                  tail=dict(start=tail,end=tail+3,text='XYZ',type='entity'),relation='supports',score=.95)
        predictor.inference.return_value=([mentions],[[link]])
        result=infer_entities([unit],predictor,include_relations=True)
        return a,b,unit,result

    def test_context_mentions_keep_native_offsets_without_cross_page_semantic_relations(self):
        a,b,unit,result=self.pair()
        result['relations']=[]
        self.assertEqual([(e['source_id'],e['start'],e['end']) for e in result['entities']],[('a',0,3),('b',0,3)])
        tree=dict(nodes=[dict(id='root',kind='document',parent_id=None)],leaves=[
            dict(chunk_id='a',text=a,page_number=1,node_id='root',ancestor_ids=['root']),
            dict(chunk_id='b',text=b,page_number=8,node_id='root',ancestor_ids=['root'])])
        tree['knowledge_graph']=extract_graph_entities(tree,lambda units:result,context_units=[unit])
        self.assertEqual(tree['knowledge_graph']['status'],'complete')
        self.assertEqual(len(tree['knowledge_graph']['source_relations']),0)
        edges=project(tree)[2]
        refs=[e for e in edges if e['type']=='SUPPORTS']
        self.assertEqual(refs,[])
        self.assertEqual({e['source_leaf_id'] for e in edges if e['type']=='MENTIONS'},{'a','b'})

    def test_context_never_relabels_other_region_text_as_the_target(self):
        a,b,unit,result=self.pair()
        result['relations']=[]
        tree=dict(leaves=[dict(chunk_id='a',text=a),dict(chunk_id='b',text=b)])
        result['entities'][1]['source_id']='a'
        output=extract_graph_entities(tree,lambda units:result,context_units=[unit])
        self.assertEqual(output['status'],'unavailable')
        self.assertEqual(output['entities'],[])

    def test_candidate_similarity_does_not_create_a_reference(self):
        tree=build_document_structure([region('a','Payment is due later.',0,page=1),
                                       region('b','Payment is due later.',0,page=9)])
        empty=dict(status='complete',entities=[],relations=[],source_relations=[],source_coverage={})
        request=Mock(side_effect=self.reference_response)
        output=enrich_document_relations(tree,empty,request,embedding=lambda texts:[[1.,0.] for _ in texts])
        self.assertTrue(request.called)
        self.assertEqual(output['source_relations'],[])
        self.assertEqual(output['document_relations']['candidate_search']['state'],'dense_model')
        pairs=request.call_args.args[0]
        self.assertTrue(pairs)
        self.assertEqual(set(pairs[0]),{'id','left','right','target_first','context_group'})

    def test_candidate_search_failure_is_explicit_and_no_qwen_is_used(self):
        tree=build_document_structure([region('a','Alpha.',0),region('b','Beta.',1)])
        with patch('ai_extract.services.contract_pipeline_service.request_qwen') as qwen:
            with self.assertRaises(RuntimeError):
                document_context_units(tree,embedding=Mock(side_effect=RuntimeError('Offline')))
        qwen.assert_not_called()

    def test_context_offsets_are_absolute_even_for_a_middle_source_window(self):
        a,b,unit,_=self.pair()
        prefix='Preface. '
        unit['context_windows'][0]['contexts'][0]['offset']=len(prefix)
        predictor=Mock(device='cuda:0')
        start=len(a)+len(CONTEXT_SEPARATOR)
        predictor.inference.return_value=[[dict(start=start,end=start+3,text='XYZ',label='entity',score=.95)]]
        result=infer_entities([unit],predictor)
        entity=result['entities'][0]
        self.assertEqual((entity['source_id'],entity['start'],entity['end']),('b',len(prefix),len(prefix)+3))
        output=extract_graph_entities(dict(leaves=[dict(chunk_id='a',text=a),dict(chunk_id='b',text=prefix+b)]),
                                      lambda units:result,context_units=[unit])
        self.assertEqual(output['status'],'complete')

    def test_a_span_crossing_the_source_boundary_cannot_be_an_entity(self):
        a,b,unit,_=self.pair()
        combined=a+CONTEXT_SEPARATOR+b
        predictor=Mock(device='cuda:0')
        start=len(a)-2;end=len(a)+len(CONTEXT_SEPARATOR)+2
        predictor.inference.return_value=[[dict(start=start,end=end,text=combined[start:end],label='entity',score=.99)]]
        self.assertEqual(infer_entities([unit],predictor)['entities'],[])

    def test_dense_candidates_can_reach_a_distant_different_language_region(self):
        tree=build_document_structure([region('a','Payment obligations',0,page=1),
            region('unrelated','Operating temperature',0,page=2),region('b','Nghĩa vụ thanh toán',0,page=19)])
        embed=lambda texts:[[0.,1.] if text=='Operating temperature' else [1.,0.] for text in texts]
        with patch.dict('os.environ',{'GLINER_DOCUMENT_CANDIDATES':'1'}):
            units,state=document_context_units(tree,embedding=embed)
        a=next(u for u in units if u['id'].startswith('a:'))
        self.assertTrue(any(c['id'].startswith('b:') for w in a['context_windows'] for c in w['contexts']))
        self.assertEqual(state['state'],'dense_model')
        self.assertEqual(state['candidate_pairs'],len(state['selected_pairs']))
        originals={l['chunk_id']:l['text'] for l in tree['leaves']}
        for pair in state['selected_pairs']:
            for prefix in ('from','to'):
                text=pair[prefix+'_text'];offset=pair[prefix+'_offset']
                self.assertEqual(originals[pair[prefix+'_id']][offset:offset+len(text)],text)

    def test_reference_retrieval_retains_lower_ranked_sources_with_bounded_diverse_budget(self):
        rows=[dict(chunk_id='query',text='query')]+[dict(chunk_id=str(i),text=str(i)) for i in range(20)]
        vectors={'query':[1.,0.],**{str(i):[1.,(i+1)/10] for i in range(20)}}
        with patch.dict('os.environ',{'GLINER_DOCUMENT_CANDIDATES':'16'}):
            units,state=document_context_units(dict(leaves=rows),embedding=lambda texts:[vectors[t] for t in texts])
        query=next(u for u in units if u['id']=='query')
        contexts=query['context_windows'][0]['contexts']
        self.assertEqual(len(contexts),16)
        self.assertIn('13',{c['id'] for c in contexts})
        self.assertEqual(len({c['id'] for c in contexts}),16)

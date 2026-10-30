from django.test import SimpleTestCase, override_settings
from ai_extract.services.document_structure_service import table_context
from NewIdea.services.knowledge_answer_service import bind_graph_context, needs_source_images, graph_source_order, model_graph_context
from ai_extract.services.document_graph_service import infer_relations, projection
from types import SimpleNamespace


class RelationContextTests(SimpleTestCase):
    def test_selected_evidence_node_is_sent_even_when_it_has_no_edges(self):
        context=dict(state='ready',version_id=1,nodes=[dict(id='leaf',kind='evidence',source_leaf_ids=['leaf'])],edges=[])
        output=bind_graph_context([context],[dict(evidence_id='E1',version_id=1,chunk_id='leaf')])
        self.assertEqual(len(output['nodes']),1)
        self.assertEqual(output['nodes'][0]['evidence_ids'],['E1'])
        self.assertEqual(output['edges'],[])
        self.assertEqual(output['scope'],'source_bound_subgraph')

    def test_unavailable_context_is_not_reported_as_complete(self):
        output=bind_graph_context([dict(state='unavailable',version_id=1)],[])
        self.assertTrue(output['partial'])
        self.assertEqual(output['unavailable_contexts'],[dict(state='unavailable',version_id=1)])
    def test_compact_quotes_are_exact_source_spans_and_proposals_keep_qualifiers(self):
        text='Only after approval: '+('a multilingual source passage. '*8)
        quote=text[21:]
        graph=dict(nodes=[dict(id=1,kind='condition',label='Approval'),dict(id=2,kind='product',label='Item')],
            edges=[dict(type='DEPENDS_ON',from_id=1,to_id=2,evidence_ids=['E1'],quote=quote,
                scope_quote=text,polarity='negated',condition_quotes=['Only after approval'],review_status='unreviewed')])
        wire=model_graph_context(graph,[dict(evidence_id='E1',text=text)])
        group=wire['relations']['DEPENDS_ON'][0]
        edge=group['constants'] if 'links' in group else dict(group['constants'],**dict(zip(group['columns'],group['rows'][0])))
        self.assertEqual(edge['scope_quote'],dict(source='E1'))
        start,end=edge['quote']['span']
        self.assertEqual(text[start:end],quote)
        self.assertEqual(edge['polarity'],'negated')
        self.assertEqual(edge['condition_quotes'],['Only after approval'])
        self.assertEqual(edge['review_status'],'unreviewed')
        self.assertEqual(graph['edges'][0]['scope_quote'],text)

    def table_graph(self):
        nodes=[dict(id=1,kind='table',evidence_ids=['E1']),
               dict(id=2,kind='table_row',label='table_row',row=1,evidence_ids=['E1']),
               dict(id=3,kind='table_cell',label='Amount',row=1,column=1,rowspan=1,colspan=2,evidence_ids=['E1']),
               dict(id=4,kind='table_cell',label='10 kg',row=1,column=3,rowspan=1,colspan=1,evidence_ids=['E1'])]
        edges=[dict(type='HAS_ROW',from_id=1,to_id=2,review_status='source_extracted'),
               dict(type='HAS_CELL',from_id=2,to_id=3,review_status='unreviewed'),
               dict(type='HAS_CELL',from_id=2,to_id=4,review_status='unreviewed')]
        return dict(nodes=nodes,edges=edges,partial=False)

    def test_grid_preserves_columns_merged_cells_source_and_review_status(self):
        graph=self.table_graph()
        wire=model_graph_context(graph,[])
        grid=wire['table_grids'][0]
        self.assertEqual(grid['rows'],[[1,[[1,'Amount',1,2],[3,'10 kg']]]])
        self.assertEqual(grid['cell_defaults']['evidence_ids'],['E1'])
        self.assertEqual(grid['HAS_CELL']['review_status'],'unreviewed')
        self.assertEqual(len(graph['nodes']),4)
        self.assertEqual(len(graph['edges']),3)

    def test_referenced_cell_or_differing_review_status_is_never_folded(self):
        for extra in (True,False):
            graph=self.table_graph()
            if extra:graph['edges'].append(dict(type='REFERENCES',from_id=1,to_id=3,quote='Amount'))
            else:graph['edges'][-1]['review_status']='source_extracted'
            wire=model_graph_context(graph,[])
            self.assertNotIn('table_grids',wire)
            self.assertEqual(sum(len(g['rows']) for g in wire['nodes']['groups']),4)

    def test_compact_graph_keeps_every_relation_target_and_tabbed_labels(self):
        labels=['Amount\tkg','First line\nSecond line']*8
        graph=dict(nodes=[dict(id=i,kind='term',label=label,evidence_ids=['E1']) for i,label in enumerate(labels)],
            edges=[dict(type='MENTIONS',from_id=0,to_id=i,evidence_ids=['E1']) for i in range(1,len(labels))])
        wire=model_graph_context(graph,[])
        self.assertEqual([r[1] for g in wire['nodes']['groups'] for r in g['rows']],labels)
        self.assertEqual(wire['relations']['MENTIONS'][0]['links'],[[0,list(range(1,len(labels)))]])
        self.assertEqual(len(graph['edges']),15)

    def test_grouped_model_graph_preserves_every_edge_field(self):
        graph=dict(nodes=[dict(id='a'),dict(id='b')],edges=[
            dict(type='DEPENDS_ON',from_id='a',to_id='b',evidence_ids=['E1'],
                 condition_quotes=['Only after inspection'],review_status='unreviewed'),
            dict(type='DEPENDS_ON',from_id='b',to_id='a',evidence_ids=['E2'],
                 polarity='negated',review_status='unreviewed')],partial=True,omitted_edges=2,
            relation_legend=dict(DEPENDS_ON='Dependency'))
        wire=model_graph_context(graph)
        restored=[]
        for kind,groups in wire['relations'].items():
            for group in groups:
                for row in group['rows']:
                    edge=dict(type=kind,**group['constants'],**dict(zip(group['columns'],row)))
                    restored.append({key:value for key,value in edge.items() if value is not None})
        self.assertEqual(restored,graph['edges'])
        self.assertEqual([dict(group['constants'],**dict(zip(group['columns'],row)))
            for group in wire['nodes']['groups'] for row in group['rows']],graph['nodes'])
        self.assertEqual(wire['coverage']['omitted_edges'],2)

    def test_shared_page_does_not_assign_other_paragraphs_sources_to_an_edge(self):
        context=dict(state='ready',version_id=1,nodes=[
            dict(id='a',kind='evidence',source_leaf_ids=['a']),
            dict(id='b',kind='evidence',source_leaf_ids=['b']),
            dict(id='page',kind='source_image',source_leaf_ids=['a','b'])],edges=[
                dict(from_id='a',to_id='page',type='HAS_VISUAL'),
                dict(from_id='b',to_id='page',type='HAS_VISUAL')])
        result=bind_graph_context([context],[dict(evidence_id='E1',version_id=1,chunk_id='a'),
            dict(evidence_id='E2',version_id=1,chunk_id='b')])
        self.assertEqual([edge['evidence_ids'] for edge in result['edges']],[['E1'],['E2']])

    def test_prompt_source_order_uses_stored_next_without_rebinding_citations(self):
        evidence=[dict(evidence_id='E1',version_id=1,chunk_id='second'),
                  dict(evidence_id='E2',version_id=1,chunk_id='first')]
        context=dict(state='ready',version_id=1,nodes=[dict(id='a',source_leaf_ids=['first']),
            dict(id='b',source_leaf_ids=['second'])],edges=[dict(type='NEXT',from_id='a',to_id='b')])
        self.assertEqual(graph_source_order([context],evidence),[evidence[1],evidence[0]])
        context['edges'].append(dict(type='NEXT',from_id='b',to_id='a'))
        self.assertEqual(graph_source_order([context],evidence),evidence)
        self.assertEqual(graph_source_order([dict(context,version_id=2)],evidence),evidence)

    def test_database_ids_do_not_crowd_out_source_bindings_or_reading_order(self):
        nodes = [dict(id=str(i)*64, kind='evidence', source_leaf_ids=[f'l{i}']) for i in range(12)]
        nodes.append(dict(id='heading'*20, kind='section', label='Setup', source_leaf_ids=[f'l{i}' for i in range(12)]))
        edges = [dict(from_id=nodes[-1]['id'], to_id=node['id'], type='HAS_EVIDENCE') for node in nodes[:-1]]
        edges += [dict(from_id=a['id'], to_id=b['id'], type='NEXT') for a,b in zip(nodes[:10], nodes[1:11])]
        evidence = [dict(evidence_id=f'E{i+1}', version_id=1, chunk_id=f'l{i}') for i in range(12)]
        result = bind_graph_context([dict(state='ready',version_id=1,nodes=nodes,edges=edges)], evidence)
        self.assertEqual({edge['type'] for edge in result['edges']}, {'HAS_EVIDENCE','NEXT'})
        self.assertTrue(all(isinstance(node['id'],int) for node in result['nodes']))
        self.assertTrue(all(edge['evidence_ids'] for edge in result['edges']))

    def test_direct_semantic_relation_has_room_before_large_heading_tree(self):
        nodes=[];edges=[]
        for i in range(30):
            nodes.extend([dict(id=f'h{i}',kind='section',label='Long heading '+str(i),source_leaf_ids=['u1']),
                          dict(id=f'b{i}',kind='section',label='Subheading '+str(i),source_leaf_ids=['u1'])])
            edges.append(dict(from_id=f'h{i}',to_id=f'b{i}',type='CONTAINS',source_leaf_ids=['u1']))
        nodes.extend([dict(id='product',kind='entity',label='Runtime',source_leaf_ids=['u1']),
                      dict(id='controller',kind='entity',label='Controller',source_leaf_ids=['u1'])])
        edges.append(dict(from_id='product',to_id='controller',type='DEPENDS_ON',predicate='depends_on',
                          source_leaf_ids=['u1'],quote='Runtime requires a controller',review_status='unreviewed'))
        result=bind_graph_context([dict(state='ready',version_id=1,nodes=nodes,edges=edges)],
                                  [dict(evidence_id='E1',version_id=1,chunk_id='u1')],max_chars=3000)
        self.assertIn('DEPENDS_ON',{e['type'] for e in result['edges']})
        self.assertTrue(result['partial'])
        self.assertLessEqual(len(__import__('json').dumps(result,ensure_ascii=False)),3000)
    def test_typed_semantic_edge_keeps_its_evidence_and_review_status_for_a2(self):
        context=dict(state='ready',version_id=1,nodes=[
            dict(id='a',kind='entity',label='Payment',source_leaf_ids=['u1']),
            dict(id='b',kind='entity',label='30 days',source_leaf_ids=['u1'])],
            edges=[dict(from_id='a',to_id='b',type='HAS_DEADLINE',predicate='has_deadline',
                source_leaf_ids=['u1'],quote='Payment within 30 days',review_status='unreviewed')])
        result=bind_graph_context([context],[dict(version_id=1,chunk_id='u1',evidence_id='E1')])
        self.assertEqual(result['edges'][0]['type'],'HAS_DEADLINE')
        self.assertEqual(result['edges'][0]['review_status'],'unreviewed')
        self.assertEqual(result['edges'][0]['quote'],'Payment within 30 days')
    def test_geometry_is_bounded_supplement_after_primary_context(self):
        evidence=[dict(evidence_id='E1',version_id=1,chunk_id='l1'),dict(evidence_id='E2',version_id=1,chunk_id='l2')]
        nodes=[dict(id='a',kind='paragraph',bbox=[0,0,100,100],page_number=1,image_path='p1.png',source_leaf_ids=['l1'])]
        nodes.extend(dict(id='b'+str(i),kind='paragraph',bbox=[120,0,220,100],page_number=1,image_path='p1.png',source_leaf_ids=['l2']) for i in range(10))
        edges=[dict(from_id='a',to_id='b'+str(i),type='LEFT_OF') for i in range(10)]
        edges.append(dict(from_id='a',to_id='b0',type='HAS_EVIDENCE'))
        context=dict(state='ready',version_id=1,nodes=nodes,edges=edges)
        result=bind_graph_context([context],evidence,max_chars=20000)
        self.assertEqual(result['edges'][0]['type'],'HAS_EVIDENCE')
        spatial=[e for e in result['edges'] if e['type']=='LEFT_OF']
        self.assertEqual(len(spatial),10)
        self.assertIn('LEFT_OF',result['relation_legend'])
        self.assertIn('HAS_EVIDENCE',result['relation_legend'])
        self.assertTrue(all(e['role']=='supplementary' and 'not_reading_order' in e['meaning'] for e in spatial))
        result=bind_graph_context([context],evidence,max_chars=20000,query='Mô tả hình')
        self.assertEqual(sum(e['type']=='LEFT_OF' for e in result['edges']),10)

    def test_a2_preserves_stored_spatial_edges_instead_of_recomputing_them(self):
        evidence=[dict(evidence_id='E1',version_id=1,chunk_id='l1'),dict(evidence_id='E2',version_id=1,chunk_id='l2')]
        for bbox,page in [([120,200,220,300],1),([500,0,600,100],1),([120,0,220,100],2)]:
            context=dict(state='ready',version_id=1,nodes=[
                dict(id='a',kind='paragraph',bbox=[0,0,100,100],page_number=1,image_path='p1.png',source_leaf_ids=['l1']),
                dict(id='b',kind='paragraph',bbox=bbox,page_number=page,image_path='p1.png',source_leaf_ids=['l2'])],
                edges=[dict(from_id='a',to_id='b',type='LEFT_OF')])
            result=bind_graph_context([context],evidence)
            self.assertEqual(len(result['edges']),1)
            self.assertEqual(result['edges'][0]['type'],'LEFT_OF')
            self.assertEqual(result['edges'][0]['role'],'supplementary')

    def test_spatial_edges_do_not_consume_answer_context_or_imply_continuation(self):
        context=dict(state='ready',version_id=1,nodes=[
            dict(id='a',kind='paragraph',source_leaf_ids=['l1']),
            dict(id='b',kind='paragraph',source_leaf_ids=['l2'])],edges=[
                dict(from_id='a',to_id='b',type=kind,origin='source_bbox_geometry_not_semantics')
                for kind in ['LEFT_OF','ABOVE','NEAR','NEXT','HAS_EVIDENCE']])
        evidence=[dict(evidence_id='E1',version_id=1,chunk_id='l1'),
                  dict(evidence_id='E2',version_id=1,chunk_id='l2')]
        result=bind_graph_context([context],evidence)
        self.assertEqual({e['type'] for e in result['edges']},{'NEXT','HAS_EVIDENCE','LEFT_OF','ABOVE','NEAR'})
        self.assertEqual(result['filtered_spatial_edges'],0)
        self.assertFalse(result['partial'])
        next_edge=next(e for e in result['edges'] if e['type']=='NEXT')
        self.assertIn('not_same_paragraph',next_edge['meaning'])
        self.assertEqual(next_edge['origin'],'source_bbox_geometry_not_semantics')

    def test_images_are_used_for_visual_questions_and_primary_visual_hits(self):
        text_source=dict(text='Guidance runs on a PC.',image_path='page.png',expansion_reason='query_match')
        figure_context=dict(text='',image_path='figure.png',expansion_reason='neo4j_source_relation')
        self.assertFalse(needs_source_images({},[text_source,figure_context]))
        self.assertTrue(needs_source_images(dict(intent=dict(needs_images=True)),[text_source]))
        self.assertTrue(needs_source_images({},[dict(figure_context,expansion_reason='query_match')]))

    def test_unresolved_table_includes_source_image_even_with_ocr_text(self):
        table=dict(layout_type='table', text='J6 0.05 kgm2', image_path='table.png',
                   structure_metadata=dict(table_structure_status='unresolved'))
        self.assertTrue(needs_source_images({},[table]))
        highres=dict(table,image_path='',structure_metadata=dict(table_structure_status='unresolved',
                                                                 structure_image_path='page.png'))
        self.assertTrue(needs_source_images({},[highres]))
        resolved=dict(table,structure_metadata=dict(table_structure_status='cell_aware'))
        self.assertFalse(needs_source_images({},[resolved]))

    def test_graph_context_cannot_bind_foreign_or_unsent_evidence(self):
        context=dict(state='ready',version_id=1,nodes=[
            dict(id='v1:n1',kind='section',label='Setup',source_leaf_ids=['leaf1']),
            dict(id='v1:n2',kind='evidence',label='PC',source_leaf_ids=['leaf1']),
            dict(id='v1:n3',kind='evidence',label='Unsent',source_leaf_ids=['leaf2'])],edges=[
                dict(from_id='v1:n1',to_id='v1:n2',type='HAS_EVIDENCE',source_leaf_ids=[]),
                dict(from_id='v1:n1',to_id='v1:n3',type='HAS_EVIDENCE',source_leaf_ids=[])])
        evidence=[dict(evidence_id='E1',version_id=1,chunk_id='leaf1')]
        result=bind_graph_context([context],evidence)
        self.assertEqual(len(result['edges']),1)
        self.assertEqual(result['edges'][0]['evidence_ids'],['E1'])
        self.assertTrue(result['partial'])
        self.assertNotIn('Unsent',str(result))
        foreign=bind_graph_context([context],[dict(evidence_id='E1',version_id=2,chunk_id='leaf1')])
        self.assertEqual(foreign['nodes'],[])
        bounded=bind_graph_context([context],evidence,max_chars=100)
        self.assertEqual(bounded['edges'],[])
        self.assertTrue(bounded['partial'])

    def test_pdf_rows_keep_correct_columns_and_units_after_splitting(self):
        rows = [['Device', 'Voltage (V)', 'Material'], ['AX-203', '24', 'SUS304'], ['BX-204', '48', 'Aluminum']]
        record = dict(chunk_id='block:1', text='\n'.join('\t'.join(r) for r in rows), layout_type='table',
                      structure_metadata=dict(table_headers=rows[0], table_rows=rows))
        units = [dict(record,table_source_row=i) for i in range(len(rows))]
        first, second = table_context(units[1]), table_context(units[2])
        self.assertEqual(first['cells'][1], dict(column=2, header='Voltage (V)', value='24'))
        self.assertEqual(first['row_context'], 'AX-203')
        self.assertEqual(second['cells'][1]['value'], '48')
        self.assertNotIn('24', [c['value'] for c in second['cells']])

    def test_ragged_rows_do_not_invent_columns(self):
        result = table_context(dict(structure_metadata=dict(table_headers=['Device', 'Voltage'], table_values=['A', '24', 'unknown'])))
        self.assertEqual(result['status'], 'ambiguous_columns')
        self.assertNotIn('cells', result)


    @override_settings(A3_KNOWLEDGE_ENABLED=True)
    def test_context_reaches_ner_and_qualification_stays_in_original_source(self):
        text = 'A supports B if accepted.'
        leaf = dict(chunk_id='u1', text=text, ancestor_ids=['root'], node_id='root',
                    structure_metadata=dict(table_headers=['Party', 'Duty'], table_values=['A', text]))
        def model(units):
            from ai_extract.services.gliner_entity_service import mention_key
            self.assertEqual(units[0]['text'],text)
            entities=[dict(name=name,source_id='u1',start=text.index(name),end=text.index(name)+1,
                entity_type='ENTITY',confidence=.95) for name in ('A','B')]
            return dict(entities=entities,relations=[])
        tree = dict(leaves=[leaf], nodes=[dict(id='root', kind='document')], source_fingerprint='test')
        tree['knowledge_graph'] = infer_relations(tree, model)
        self.assertEqual(tree['knowledge_graph']['status'], 'complete')
        run = SimpleNamespace(pk=1, version_id=1, version=SimpleNamespace(contract=SimpleNamespace(company_id=1, pk=1)))
        _, _, edges = projection(run, tree)
        self.assertFalse(any(e['type']=='SUPPORTS' for e in edges))
        self.assertEqual(sum(e['type']=='MENTIONS' for e in edges),2)
        self.assertEqual(tree['leaves'][0]['text'],text)


class EdgeBindingDefaultsTests(SimpleTestCase):
    def test_explicit_binding_is_kept_when_it_differs_from_endpoint_default(self):
        graph=dict(nodes=[dict(id=1,evidence_ids=['E1','E2']),dict(id=2,evidence_ids=['E1','E2'])],
            edges=[dict(type='REFERENCES',from_id=1,to_id=2,evidence_ids=['E1'])])
        wire=model_graph_context(graph,[])
        group=wire['relations']['REFERENCES'][0]
        edge=group['constants'] if 'links' in group else dict(group['constants'],**dict(zip(group['columns'],group['rows'][0])))
        self.assertEqual(edge['evidence_ids'],['E1'])
        self.assertNotIn('edge_source_default',wire)

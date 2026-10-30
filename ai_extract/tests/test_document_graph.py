from copy import deepcopy
import os
from uuid import uuid4
from unittest import skipUnless
from types import SimpleNamespace
from unittest.mock import patch
from django.test import SimpleTestCase, override_settings
from ai_extract.services import document_graph_service as graph
from ai_extract.services.contract_pipeline_service import structural_retrieve, retrieve_hybrid
from ai_extract.repositories import document_graph_repository as repository


class StructureReviewTests(SimpleTestCase):
    @override_settings(A3_GRAPH_ENABLED=True, A3_KNOWLEDGE_ENABLED=True, NEO4J_URI='bolt://test', NEO4J_PASSWORD='test')
    def test_temporary_graph_failure_is_retried_without_changing_projection(self):
        from neo4j.exceptions import ServiceUnavailable
        run, _ = fixture()
        with patch.object(graph, 'validate_prepared'), patch.object(graph.time, 'sleep'), \
             patch.object(repository, 'replace_snapshot', side_effect=[ServiceUnavailable('temporary'), None]) as publish:
            result = graph.sync_graph(run)
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(publish.call_count, 2)
        self.assertEqual(publish.call_args_list[0], publish.call_args_list[1])

    @override_settings(A3_GRAPH_ENABLED=True, A3_KNOWLEDGE_ENABLED=True, NEO4J_URI='bolt://test', NEO4J_PASSWORD='test')
    def test_persistent_graph_failure_is_bounded_and_never_ready(self):
        from neo4j.exceptions import ServiceUnavailable
        run, _ = fixture()
        with patch.object(graph, 'validate_prepared'), patch.object(graph.time, 'sleep'), \
             patch.object(repository, 'replace_snapshot', side_effect=ServiceUnavailable('offline')) as publish:
            result = graph.sync_graph(run)
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(publish.call_count, 3)

    @override_settings(A3_GRAPH_ENABLED=True)
    def test_unresolved_table_exposes_warning_without_cells_to_llm(self):
        run,tree=fixture();leaf=tree['leaves'][0]
        leaf['layout_type']='table'
        leaf['structure_metadata']=dict(table_structure_status='unresolved',
            table_structure_review=dict(reason='Independent column predictions disagree'),table_rows=[['Untrusted']])
        tree['graph_index']=dict(status='ready',fingerprint=tree['source_fingerprint'])
        _,nodes,edges=graph.projection(run,tree)
        self.assertFalse(any(n['category']=='table_cell' for n in nodes))
        with patch.object(repository,'context_edges',return_value=dict(edges=edges,truncated=False)):
            result=graph.read_context_graph(run,tree,[leaf])
        source=next(n for n in result['nodes'] if n['kind']=='evidence' and leaf['chunk_id'] in n['source_leaf_ids'])
        self.assertEqual(source['table_structure_status'],'unresolved')
        self.assertIn('column predictions',source['table_structure_reason'])

    @override_settings(A3_GRAPH_ENABLED=True)
    def test_table_header_role_survives_projection_and_llm_context(self):
        run,tree=fixture()
        leaf=tree['leaves'][0]
        leaf['structure_metadata']=dict(table_rows=[['Item','Value']],table_cell_bbox_scope='page_normalized_1000',table_cells=[
            dict(row=1,column=1,text='Item',header=True,bbox=[10,20,30,40]),
            dict(row=1,column=2,text='Value',header=False)],table_header_origin='surya_html_unreviewed')
        tree['graph_index']=dict(status='ready',fingerprint=tree['source_fingerprint'])
        _,nodes,edges=graph.projection(run,tree)
        cells=[n for n in nodes if n.get('category')=='table_cell']
        self.assertEqual({n['text']:n['header'] for n in cells},{'Item':True,'Value':False})
        with patch.object(repository,'context_edges',return_value=dict(edges=edges,truncated=False)):
            result=graph.read_context_graph(run,tree,[leaf])
        self.assertEqual({n['label']:n['header'] for n in result['nodes'] if n.get('kind')=='table_cell'},
                         {'Item':True,'Value':False})
        self.assertEqual(next(n['bbox'] for n in result['nodes'] if n.get('label')=='Item'),[10,20,30,40])

    def test_foreign_graph_nodes_or_edges_are_rejected_before_any_write(self):
        scope=dict(key='local',entity_compacted=True)
        local=dict(key='local:one',snapshot_key='local')
        foreign=dict(key='other:two',snapshot_key='other')
        with patch.object(repository,'connection') as connect:
            with self.assertRaisesRegex(ValueError,'same document'):
                repository.replace_snapshot(scope,[local,foreign],[])
            with self.assertRaisesRegex(ValueError,'Cross-document'):
                repository.replace_snapshot(scope,[local],[dict(type='NEXT',from_key=local['key'],to_key=foreign['key'])])
            connect.assert_not_called()

    def test_shared_registry_cannot_reintroduce_cross_document_identity(self):
        with patch.object(repository,'connection') as connect:
            self.assertEqual(repository.shared_entity_sources([dict(key='one')],['two']),[])
            with self.assertRaisesRegex(ValueError,'Shared entity registry'):
                repository.replace_snapshot(dict(key='one'),[dict(key='one:entity',snapshot_key='one',category='entity',canonical_key='shared')],[])
            connect.assert_not_called()

    def test_oversized_source_is_explicitly_unreviewed_instead_of_silently_clipped(self):
        from unittest.mock import Mock
        run,tree=self.tree_with_order()
        tree['leaves'][0]['text']='Source paragraph. '*120
        model=Mock(side_effect=self.model)
        result=graph.review_surya_relations(run,model)
        self.assertTrue(any(v.get('reason')=='source_exceeds_review_budget' for v in result['relations'].values()))
        for call in model.call_args_list:
            self.assertTrue(all(len(p[side])<=1800 for p in call.args[0] for side in ('left','right')))

    def test_review_policy_change_invalidates_cached_verdicts(self):
        from unittest.mock import Mock
        from ai_extract.services.gliner_entity_service import structure_review_policy
        run,tree=self.tree_with_order();model=Mock(side_effect=self.model)
        graph.review_surya_relations(run,model)
        calls=model.call_count
        policy=structure_review_policy();policy['acceptance_threshold']=.95
        with patch('ai_extract.services.gliner_entity_service.structure_review_policy',return_value=policy):
            graph.review_surya_relations(run,model)
        self.assertGreater(model.call_count,calls)

    @override_settings(A3_GRAPH_ENABLED=True,A3_KNOWLEDGE_ENABLED=True,NEO4J_URI='bolt://test',NEO4J_PASSWORD='test')
    def test_unprepared_document_cannot_be_written_to_neo4j(self):
        run,tree=fixture()
        with patch.object(repository,'replace_snapshot') as publish:
            result=graph.sync_graph(run)
        publish.assert_not_called()
        self.assertEqual(result['status'],'unavailable')

    def tree_with_order(self):
        run,tree=fixture()
        for i,leaf in enumerate(tree['leaves']):
            leaf.update(page_number=1,structure_metadata=dict(reading_order=i,
                reading_order_source='surya_layout_model',reading_order_scope='page'))
        return run,tree

    def model(self,pairs):
        return dict(decisions=[dict(id=p['id'],status='unresolved',supported=0.,reverse=0.,evidence=[]) for p in pairs])

    def test_every_source_relation_gets_a_review_and_cached_results_are_reused(self):
        from unittest.mock import Mock
        run,tree=self.tree_with_order();model=Mock(side_effect=self.model)
        result=graph.review_surya_relations(run,model)
        _,_,edges=graph.projection(run,tree,include_review=False)
        self.assertEqual(result['relations_total'],sum(e['type'] in graph.SOURCE_RELATIONS for e in edges))
        self.assertEqual(sum(result['counts'].values()),result['relations_total'])
        self.assertGreater(result['counts']['unresolved'],0)
        calls=model.call_count
        graph.review_surya_relations(run,model)
        self.assertEqual(model.call_count,calls)
        tree['source_fingerprint']='changed'
        graph.review_surya_relations(run,model)
        self.assertGreater(model.call_count,calls)

    def test_runtime_failure_keeps_pending_review_instead_of_claiming_confirmation(self):
        from unittest.mock import Mock
        run,tree=self.tree_with_order()
        result=graph.review_surya_relations(run,Mock(side_effect=RuntimeError('offline')))
        self.assertEqual(result['status'],'partial')
        self.assertGreater(result['counts']['pending'],0)
        _,_,edges=graph.projection(run,tree)
        self.assertTrue(any(e['type']=='NEXT' and e.get('review_status')=='model_pending' for e in edges))

    def test_invalid_source_endpoint_is_audited_as_invalid(self):
        run,tree=self.tree_with_order()
        _,nodes,edges=graph.projection(run,tree,include_review=False)
        broken=dict(from_key='missing-source',to_key=nodes[0]['key'],type='NEXT',origin='surya_layout_model')
        original=graph.projection
        def project(*args,**kwargs):
            if kwargs.get('include_review') is False:return {},nodes,[*edges,broken]
            return original(*args,**kwargs)
        with patch.object(graph,'projection',side_effect=project):
            result=graph.review_surya_relations(run,self.model)
        self.assertEqual(result['relations'][graph.relation_review_key(broken)]['status'],'invalid')
        self.assertEqual(result['counts']['invalid'],1)

    def test_model_dispute_is_exposed_without_overwriting_source_order(self):
        run,tree=self.tree_with_order()
        def model(pairs):
            return dict(decisions=[dict(id=p['id'],status='disputed',supported=0.,reverse=.98,
                evidence=[dict(predicate='reading follows',head=p['left'],tail=p['right'],confidence=.98)]) for p in pairs])
        original=[l['structure_metadata']['reading_order'] for l in tree['leaves']]
        graph.review_surya_relations(run,model)
        _,_,edges=graph.projection(run,tree)
        self.assertTrue(any(e['type']=='NEXT' and e.get('review_status')=='model_disputed' for e in edges))
        self.assertEqual([l['structure_metadata']['reading_order'] for l in tree['leaves']],original)


class SharedIdentityTests(SimpleTestCase):
    @override_settings(NEO4J_URI='bolt://pool-test:7687',NEO4J_USER='test',NEO4J_PASSWORD='test')
    def test_connections_reuse_driver_but_open_independent_sessions(self):
        with patch.object(repository,'_driver',None),patch.object(repository,'_driver_settings',None), \
             patch('neo4j.GraphDatabase.driver') as factory:
            with repository.connection():pass
            with repository.connection():pass
            factory.assert_called_once()
            self.assertEqual(factory.return_value.session.call_count,2)
            factory.return_value.close.assert_not_called()
            repository.close_driver()
            factory.return_value.close.assert_called_once()

    def test_idle_entity_worker_does_not_write_transaction(self):
        with patch.object(repository,'connection') as connect:
            session=connect.return_value.__enter__.return_value
            session.run.return_value.single.return_value=None
            self.assertIsNone(repository.claim_entity_job())
            session.execute_write.assert_not_called()

    def test_retired_graph_first_identity_rejects_even_injected_models(self):
        from unittest.mock import Mock
        model=Mock()
        with self.assertRaisesMessage(RuntimeError,'Graph-first entity resolution is retired'):
            graph.resolve_shared_entities(dict(company_id=1),request_model=model)
        model.assert_not_called()


def fixture():
    leaves = [dict(chunk_id='u1', node_id='u1', source_chunk_id='block:10', text='Pump A contains Motor B.',
                   heading_path=['Assembly'], ancestor_ids=['root','section','u1'], page_number=1,
                   bbox=[0,0,100,100], related_chunk_ids=[]),
              dict(chunk_id='u2', node_id='u2', source_chunk_id='block:20', text='Motor B maintenance interval is 90 days.',
                   heading_path=['Maintenance'], ancestor_ids=['root','other','u2'], page_number=2,
                   bbox=[0,0,100,100], related_chunk_ids=[])]
    tree = dict(source_fingerprint='abc', structure_status='model_complete',
                nodes=[dict(id='root',parent_id=None,kind='document',title='Manual'),
                       dict(id='section',parent_id='root',kind='section',title='Assembly'),
                       dict(id='other',parent_id='root',kind='section',title='Maintenance'),
                       dict(id='u1',parent_id='section',kind='paragraph',title=''),
                       dict(id='u2',parent_id='other',kind='paragraph',title='')], leaves=leaves, relations=[])
    run = SimpleNamespace(pk=7, version_id=3, structure=tree,
                          version=SimpleNamespace(contract=SimpleNamespace(pk=2,company_id=1)))
    return run, tree


def answer(units):
    from ai_extract.services.gliner_entity_service import mention_key
    text='Pump A contains Motor B.'
    entities=[dict(name=name,source_id='u1',start=text.index(name),end=text.index(name)+len(name),
                   entity_type='ENTITY',confidence=.95) for name in ('Pump A','Motor B')]
    return dict(entities=entities,relations=[dict(from_key=mention_key(entities[1]),to_key=mention_key(entities[0]),
        predicate='part_of',source_id='u1',quote=text,confidence=.95)])


@override_settings(A3_KNOWLEDGE_ENABLED=True, A3_GRAPH_ENABLED=True, NEO4J_URI='bolt://example', NEO4J_PASSWORD='test')
class DocumentGraphTests(SimpleTestCase):
    def test_figure_text_edges_use_only_same_page_surya_parent(self):
        run,tree=fixture()
        figure,child=tree['leaves']
        figure.update(layout_type='figure',text='',page_number=1,
                      structure_metadata={'surya_region_id':'fig'})
        child.update(page_number=1,structure_metadata={'surya_parent_region_id':'fig'})
        foreign=deepcopy(child); foreign.update(chunk_id='foreign',source_chunk_id='block:30',page_number=2)
        tree['leaves'].append(foreign)
        _,_,edges=graph.projection(run,tree)
        visual=[e for e in edges if e['type']=='HAS_FIGURE_TEXT']
        self.assertEqual(len(visual),1)
        self.assertTrue(visual[0]['from_key'].endswith(':leaf:u1'))
        self.assertTrue(visual[0]['to_key'].endswith(':leaf:u2'))
        self.assertEqual(visual[0]['review_status'],'unreviewed')
        self.assertEqual(visual[0]['relation_semantics'],'source_parent_not_verified_reference')

    def test_nearby_figure_caption_is_a_candidate_not_a_confirmed_reference(self):
        run,tree=fixture()
        fig,caption=tree['leaves']
        fig.update(layout_type='figure',text='',page_number=1,bbox=[100,100,300,300])
        caption.update(layout_type='figure_caption',page_number=1,bbox=[110,305,290,325])
        _,_,edges=graph.projection(run,tree)
        found=[e for e in edges if e['type']=='FIGURE_CAPTION_CANDIDATE']
        self.assertEqual(len(found),1)
        self.assertEqual(found[0]['review_status'],'candidate')
        self.assertEqual(found[0]['relation_semantics'],'layout_proximity_only_not_verified_ownership')
        caption['page_number']=2
        _,_,edges=graph.projection(run,tree)
        self.assertFalse(any(e['type']=='FIGURE_CAPTION_CANDIDATE' for e in edges))

    def test_nearby_paragraph_is_a_candidate_even_without_caption_label(self):
        run,tree=fixture()
        fig,text=tree['leaves']
        fig.update(layout_type='figure',text='',page_number=1,bbox=[100,100,300,300])
        text.update(layout_type='paragraph',page_number=1,bbox=[110,305,290,325])
        _,_,edges=graph.projection(run,tree)
        found=[e for e in edges if e['type']=='FIGURE_NEAR_TEXT_CANDIDATE']
        self.assertEqual(len(found),1)
        self.assertEqual(found[0]['review_status'],'candidate')

    def test_compaction_preserves_distinct_evidence_and_semantics(self):
        specific = dict(from_key='a', to_key='b', type='APPLIES_TO',
                        predicate='applies_to', quote='Applies under condition X.',
                        source_leaf_id='u1', origin='gliner_relex_source_span',
                        condition_quotes=['condition X'], review_status='unreviewed')
        duplicate = dict(specific, type='RELATED')
        other_source = dict(duplicate, source_leaf_id='u2')
        other_condition = dict(duplicate, condition_quotes=['condition Y'])
        other_predicate = dict(duplicate, predicate='depends_on')
        near = dict(from_key='a', to_key='b', type='NEAR')
        above = dict(near, type='ABOVE')
        left = dict(near, type='LEFT_OF')
        edges = [specific, duplicate, other_source, other_condition, other_predicate, near, above, left]
        result = graph.compact_relations(edges)
        self.assertEqual(result, [specific, other_source, other_condition, other_predicate, above, left])
        self.assertEqual(len(edges), 8)

    def test_compaction_removes_only_provenance_equivalent_direct_cells(self):
        row = dict(from_key='table', to_key='row', type='HAS_ROW')
        cell = dict(from_key='row', to_key='cell', type='HAS_CELL', origin='source', review_status='unreviewed')
        direct = dict(cell, from_key='table')
        separate = dict(direct, review_status='reviewed')
        self.assertEqual(graph.compact_relations([row,cell,direct,separate]), [row,cell,separate])

    def test_projection_omits_near_but_preserves_bounding_boxes(self):
        run, tree = fixture()
        tree['leaves'][1].update(page_number=1, bbox=[110,110,210,210])
        _, nodes, edges = graph.projection(run, tree)
        self.assertNotIn('NEAR', {e['type'] for e in edges})
        self.assertTrue(any(n.get('bbox_json') == '[110, 110, 210, 210]' for n in nodes))

    def test_old_graph_cleanup_requires_completed_ready_matching_snapshot(self):
        run,tree=fixture()
        run.status='COMPLETED'
        tree['graph_index']=dict(status='ready',fingerprint=tree['source_fingerprint'])
        with patch.object(repository,'prune_old_runs',return_value=2) as prune:
            self.assertEqual(graph.cleanup_old_run_graphs(run),dict(status='completed',deleted_snapshots=2))
            self.assertEqual(prune.call_args.args[0],graph.graph_scope(run,tree))
            prune.reset_mock()
            for status in ('RUNNING','FAILED'):
                run.status=status
                self.assertEqual(graph.cleanup_old_run_graphs(run)['status'],'skipped')
            run.status='COMPLETED'
            tree['graph_index']['status']='unavailable'
            self.assertEqual(graph.cleanup_old_run_graphs(run)['status'],'skipped')
            tree['graph_index']=dict(status='ready',fingerprint='old')
            self.assertEqual(graph.cleanup_old_run_graphs(run)['status'],'skipped_stale_graph')
            prune.assert_not_called()

    def test_cleanup_failure_preserves_successful_extraction(self):
        run,tree=fixture()
        run.status='COMPLETED'
        tree['graph_index']=dict(status='ready',fingerprint=tree['source_fingerprint'])
        with patch.object(repository,'prune_old_runs',side_effect=RuntimeError('offline')):
            self.assertEqual(graph.cleanup_old_run_graphs(run)['status'],'failed')
        self.assertEqual(run.status,'COMPLETED')
        self.assertEqual(tree['graph_index']['status'],'ready')

    def test_compact_edges_preserve_source_reachability(self):
        run, tree = fixture()
        tree['knowledge_graph'] = dict(entities=[dict(key='pump', name='Pump A',
            origin='gliner_source_span', source_ids=['u1'])], relations=[])
        _, nodes, edges = graph.projection(run, tree)
        suffix = lambda key: key.split(':', 1)[1]
        links = {(suffix(e['from_key']), suffix(e['to_key']), e['type']) for e in edges}
        self.assertIn(('leaf:u1', 'entity:pump', 'MENTIONS'), links)
        self.assertNotIn(('entity:pump', 'leaf:u1', 'HAS_EVIDENCE'), links)
        self.assertNotIn(('node:section', 'leaf:u1', 'HAS_EVIDENCE'), links)
        self.assertIn(('node:section', 'node:u1', 'CONTAINS'), links)
        self.assertIn(('node:u1', 'leaf:u1', 'HAS_EVIDENCE'), links)
        # The compact graph still hydrates entities and every table cell from
        # selected evidence through the remaining canonical source paths.
        tree['graph_index'] = dict(status='ready', fingerprint=tree['source_fingerprint'])
        with patch.object(repository, 'context_edges', return_value=dict(edges=edges, truncated=False)):
            context = graph.read_context_graph(run, tree, tree['leaves'])
        self.assertEqual(context['state'], 'ready')
        self.assertTrue(any(n['kind']=='entity' for n in context['nodes']))

    def test_projection_preserves_source_provenance_and_model_origin(self):
        run, tree = fixture()
        tree['knowledge_graph'] = graph.infer_relations(tree, lambda units:dict(answer(units),relations=[]))
        self.assertEqual(tree['knowledge_graph']['status'], 'complete')
        scope, nodes, edges = graph.projection(run,tree)
        self.assertEqual(scope['company_id'],1)
        self.assertTrue(all(n['snapshot_key']==scope['key'] for n in nodes))
        relation = next(e for e in edges if e['type']=='MENTIONS')
        self.assertEqual(relation['source_leaf_id'],'u1')
        self.assertIn(relation['quote'],tree['leaves'][0]['text'])
        self.assertEqual(relation['origin'],'gliner_source_span')
        self.assertFalse(any(e['type']=='PART_OF' for e in edges))
        self.assertTrue(any(n.get('bbox_json')=='[0, 0, 100, 100]' for n in nodes))

    def test_invented_names_quotes_ids_and_predicates_reject_batch(self):
        _, tree = fixture()
        for field, value in [('source_id','foreign'),('quote','Fabricated connection'),('predicate','DELETE')]:
            bad=answer([]); bad['relations'][0][field]=value
            result=graph.infer_relations(tree,lambda units:bad)
            self.assertEqual(result['relations'],[])
            self.assertEqual(result['status'],'unavailable')
        bad=answer([]); bad['entities'][0]['name']='Invented machine'
        self.assertEqual(graph.infer_relations(tree,lambda units:bad)['entities'],[])

    def test_outage_stops_without_repeated_calls_and_preserves_structure(self):
        _, tree=fixture()
        tree['leaves']=[dict(tree['leaves'][0],chunk_id=str(i),text='x'*8000) for i in range(5)]
        before=deepcopy(tree)
        with patch.object(graph,'infer_relations',wraps=graph.infer_relations):
            calls=[]
            def fail(units):
                calls.append(units); raise RuntimeError('offline')
            result=graph.infer_relations(tree,fail)
        self.assertEqual(len(calls),1)
        self.assertEqual(result['status'],'unavailable')
        self.assertEqual(tree,before)

    def test_tenant_version_run_and_fingerprint_change_snapshot_key(self):
        run,tree=fixture(); original=graph.graph_scope(run,tree)['key']
        for field,value in [('company_id',2),('version_id',4),('pk',8)]:
            altered=deepcopy(run)
            if field=='company_id': altered.version.contract.company_id=value
            else: setattr(altered,field,value)
            self.assertNotEqual(graph.graph_scope(altered,tree)['key'],original)
        self.assertNotEqual(graph.graph_scope(run,dict(tree,source_fingerprint='changed'))['key'],original)
        run.version.contract.company_id=None
        with self.assertRaises(ValueError): graph.graph_scope(run,tree)

    def test_write_failure_never_reports_ready(self):
        run,_=fixture()
        with patch.object(graph,'validate_prepared'), \
             patch.object(repository,'replace_snapshot',side_effect=RuntimeError('secret details')):
            result=graph.sync_graph(run)
        self.assertEqual(result,dict(status='unavailable',error_type='RuntimeError'))

    def test_graph_expands_only_current_leaves_and_never_uses_graph_text(self):
        run,tree=fixture(); tree['graph_index']=dict(status='ready',fingerprint='abc')
        with patch.object(repository,'neighbors',return_value=[dict(seed_id='u1',leaf_id='u2',text='FAKE'),dict(leaf_id='foreign')]) as call:
            expanded,diagnostics=graph.expand_graph(run,tree,[tree['leaves'][0]])
        self.assertEqual(call.call_args.args[0]['company_id'],1)
        selected,_=structural_retrieve('Pump A',expanded,seeds=[tree['leaves'][0]])
        source=next(c for c in selected if c['chunk_id']=='u2')
        self.assertEqual(source['text'],tree['leaves'][1]['text'])
        self.assertEqual(source['expansion_reason'],'neo4j_source_relation')
        self.assertEqual(diagnostics['candidates'],1)

    def test_graph_expansion_keeps_each_seeds_own_neighbors_beyond_sixteen(self):
        run,tree=fixture();tree['graph_index']=dict(status='ready',fingerprint='abc')
        for i in range(20):
            tree['leaves'].append(dict(tree['leaves'][1],chunk_id=f'related-{i}'))
        rows=[dict(seed_id='u1',leaf_id=f'related-{i}') for i in range(20)]
        with patch.object(repository,'neighbors',return_value=rows):
            expanded,state=graph.expand_graph(run,tree,tree['leaves'][:2])
        first,second=expanded['leaves'][:2]
        self.assertEqual(len(first['graph_related_chunk_ids']),20)
        self.assertEqual(second['graph_related_chunk_ids'],[])
        self.assertEqual(state['candidates'],20)

    def test_shared_entity_expansion_retains_source_text_and_reason(self):
        run,tree=fixture(); tree['graph_index']=dict(status='ready',fingerprint='abc')
        rows=[dict(seed_id='u1',leaf_id='u2',reason='shared_entity_mention'),
              dict(seed_id='u1',leaf_id='foreign',reason='shared_entity_mention')]
        with patch.object(repository,'neighbors',return_value=rows):
            expanded,_=graph.expand_graph(run,tree,[tree['leaves'][0]])
        selected,_=structural_retrieve('Pump A',expanded,seeds=[tree['leaves'][0]])
        related=next(item for item in selected if item['chunk_id']=='u2')
        self.assertEqual(related['text'],tree['leaves'][1]['text'])
        self.assertEqual(related['expansion_reason'],'shared_entity_mention')
        self.assertNotIn('foreign',[item['chunk_id'] for item in selected])

    def test_default_retrieval_keeps_all_source_bound_neighbors_without_character_cutoff(self):
        run,tree=fixture()
        related=[]
        for i in range(30):
            identity=f'linked-{i}'
            tree['leaves'].append(dict(tree['leaves'][1],chunk_id=identity,text='Condition detail. '*80))
            related.append(identity)
        tree['leaves'][0]['graph_related_chunk_ids']=related
        selected,coverage=structural_retrieve('Pump',tree,seeds=[tree['leaves'][0]])
        self.assertEqual({leaf['chunk_id'] for leaf in selected},{'u1',*related})
        self.assertEqual(coverage['omitted_source_ids'],[])
        selected,coverage=structural_retrieve('Pump',tree,seeds=[tree['leaves'][0]],budget_chars=500)
        self.assertTrue(coverage['partial'])
        self.assertTrue(coverage['omitted_source_ids'])

    def test_indexed_retrieval_does_not_expand_unconnected_heading_siblings(self):
        run,tree=fixture();tree['graph_index']=dict(status='ready',fingerprint='abc')
        sibling=dict(tree['leaves'][0],chunk_id='unconnected',text='Another function.')
        tree['leaves'].append(sibling)
        selected,_=structural_retrieve('Pump',tree,seeds=[tree['leaves'][0]])
        self.assertEqual([leaf['chunk_id'] for leaf in selected],['u1'])

    def test_stale_and_offline_graph_fall_back_without_querying_foreign_snapshot(self):
        run,tree=fixture(); tree['graph_index']=dict(status='ready',fingerprint='old')
        with patch.object(repository,'neighbors') as call:
            _,state=graph.expand_graph(run,tree,[tree['leaves'][0]])
            call.assert_not_called()
            self.assertEqual(state['state'],'stale')
        tree['graph_index']['fingerprint']='abc'
        with patch.object(repository,'neighbors',side_effect=RuntimeError):
            expanded,state=graph.expand_graph(run,tree,[tree['leaves'][0]])
        self.assertEqual(state['state'],'unavailable')
        self.assertEqual(expanded,tree)

    def test_retrieval_uses_graph_after_selecting_seed_evidence(self):
        run,tree=fixture(); tree['graph_index']=dict(status='ready',fingerprint='abc')
        version=run.version
        version.layout_extraction_runs=SimpleNamespace(filter=lambda **kwargs:SimpleNamespace(first=lambda:run))
        with override_settings(A3_HYBRID_ENABLED=False), patch.object(repository,'neighbors',return_value=[dict(seed_id='u1',leaf_id='u2')]), patch.object(repository,'context_edges',return_value=dict(edges=[],truncated=False)):
            selected,coverage=retrieve_hybrid('Pump A',tree,version)
        self.assertEqual(coverage['graph']['state'],'ready')
        self.assertTrue(any(c['chunk_id']=='u2' for c in selected))

    def test_graph_context_hydrates_current_nodes_and_binds_source_leaves(self):
        run,tree=fixture(); tree['graph_index']=dict(status='ready',fingerprint='abc')
        scope,nodes,edges=graph.projection(run,tree)
        stored=[dict(edge, label='FAKE NEO4J TEXT') for edge in edges]
        stored.append(dict(from_key='foreign',to_key='foreign',type='RELATED',predicate='',source_leaf_id='u1'))
        with patch.object(repository,'context_edges',return_value=dict(edges=stored,truncated=False)):
            context=graph.read_context_graph(run,tree,[tree['leaves'][0]])
        self.assertEqual(context['state'],'ready')
        self.assertNotIn('FAKE',str(context))
        self.assertTrue(context['edges'])
        self.assertTrue(all(n['source_leaf_ids']==['u1'] for n in context['nodes']))
        self.assertTrue({'CONTAINS','HAS_EVIDENCE'} <= {e['type'] for e in context['edges']})
        self.assertNotIn('Maintenance',{n['label'] for n in context['nodes']})

    def test_context_graph_uses_primary_seeds_without_recursively_expanding_neighbors(self):
        run,tree=fixture();tree['graph_index']=dict(status='ready',fingerprint='abc')
        with patch.object(repository,'context_edges',return_value=dict(edges=[],truncated=False)) as read:
            result=graph.read_context_graph(run,tree,tree['leaves'],seed_ids=['u1'])
        self.assertEqual(read.call_args.args[1],['u1','u2'])
        self.assertEqual(read.call_args.kwargs['seed_ids'],['u1'])
        self.assertEqual(result['context_scope'],'direct_seed_relations_and_source_paths')

    def test_graph_context_does_not_read_stale_snapshot(self):
        run,tree=fixture(); tree['graph_index']=dict(status='ready',fingerprint='changed')
        with patch.object(repository,'context_edges') as read:
            result=graph.read_context_graph(run,tree,tree['leaves'])
        self.assertEqual(result['state'],'stale')
        read.assert_not_called()

    def test_shared_page_image_does_not_bind_other_blocks_to_same_paragraph(self):
        run,tree=fixture()
        tree['graph_index']=dict(status='ready',fingerprint='abc')
        tree['leaves'][0]['image_path']='same-page.png'
        tree['leaves'][1]['image_path']='same-page.png'
        tree['leaves'][1]['page_number']=1
        _,_,edges=graph.projection(run,tree)
        with patch.object(repository,'context_edges',return_value=dict(edges=edges,truncated=False)):
            result=graph.read_context_graph(run,tree,[tree['leaves'][0]])
        self.assertNotIn('Motor B maintenance',str(result))
        self.assertEqual(sum(n['kind']=='evidence' for n in result['nodes']),1)

    def test_repository_atomic_replace_is_parameterized_and_idempotent(self):
        from contextlib import contextmanager
        from unittest.mock import Mock
        run,tree=fixture(); scope,nodes,edges=graph.projection(run,tree)
        session,tx=Mock(),Mock()
        session.execute_write.side_effect=lambda callback:callback(tx)
        @contextmanager
        def connection(): yield session
        with patch.object(repository,'connection',connection):
            repository.replace_snapshot(scope,nodes,edges)
        self.assertEqual(session.execute_write.call_count,1)
        self.assertIn('DETACH DELETE',tx.run.call_args_list[1].args[0])
        self.assertTrue(any(c.kwargs.get('rows')==nodes for c in tx.run.call_args_list))
        self.assertTrue(all(scope['key'] not in c.args[0] for c in tx.run.call_args_list))

    def test_ingestion_enriches_then_saves_graph_state(self):
        run,tree=fixture()
        order=[]
        def prepare(run,progress):
            order.append('analyze')
            run.structure.update(knowledge_graph=dict(status='complete',entities=[],relations=[]),
                pregraph_analysis=dict(schema=graph.PREGRAPH_PIPELINE_SCHEMA,status='complete',source_fingerprint='abc'),
                reading_analysis=dict(status='complete',source_fingerprint='abc'),
                identity_resolution=dict(schema='test',status='complete',source_fingerprint='abc',model='test',
                    revision='test',mentions=0,linked=0,unresolved=0,canonical_entities=0))
        with patch('ai_extract.services.layout_service.prepare_document',side_effect=prepare), \
             patch.object(repository,'replace_snapshot') as write:
            write.side_effect=lambda *args:order.append('publish')
            graph.enrich_and_sync(run)
        write.assert_called_once()
        self.assertEqual(order,['analyze','publish'])
        self.assertEqual(run.structure['graph_index']['status'],'ready')
        self.assertEqual(run.structure['knowledge_graph']['status'],'complete')




@skipUnless(os.getenv('NEO4J_LIVE_SMOKE') == '1', 'Opt-in isolated synthetic data + live Neo4j smoke test')
class LiveGraphTests(SimpleTestCase):
    def test_shared_entity_mentions_stay_in_snapshot(self):
        run,tree=fixture()
        identity=-int(uuid4().hex[:12],16)
        run.pk=run.version_id=identity
        run.version.contract.company_id=identity
        run.version.contract.pk=identity
        tree['knowledge_graph']=dict(entities=[dict(key='shared',name='Pump A',source_ids=['u1','u2'])],relations=[])
        with override_settings(A3_GRAPH_ENABLED=True, NEO4J_URI=os.getenv('NEO4J_URI',''),
                               NEO4J_USER=os.getenv('NEO4J_USER','neo4j'),
                               NEO4J_PASSWORD=os.getenv('NEO4J_PASSWORD',''),
                               NEO4J_DATABASE=os.getenv('NEO4J_DATABASE','neo4j')):
            scope,nodes,edges=graph.projection(run,tree)
            try:
                repository.replace_snapshot(scope,nodes,edges)
                rows=repository.neighbors(scope,['u1'])
                self.assertTrue(any(row['leaf_id']=='u2' and row['reason']=='shared_entity_mention' for row in rows))
                self.assertEqual(repository.neighbors(dict(scope,company_id=identity-1),['u1']),[])
                edges=repository.context_edges(scope,['u1','u2'],seed_ids=['u1'])['edges']
                self.assertEqual(sum(edge['type']=='MENTIONS' for edge in edges),2)
            finally:
                with repository.connection() as session:
                    session.run('MATCH (n:A3Node {snapshot_key:$key}) DETACH DELETE n',key=scope['key']).consume()
                    session.run('MATCH (s:A3Snapshot {key:$key}) DETACH DELETE s',key=scope['key']).consume()

    def test_real_write_read_idempotency_and_tenant_isolation(self):
        # No SQL database or user document is touched. Unique negative IDs
        # distinguish this synthetic graph from all application data.
        run,tree=fixture()
        identity=-int(uuid4().hex[:12],16)
        run.pk=run.version_id=identity
        run.version.contract.company_id=identity
        run.version.contract.pk=identity
        tree['leaves'][0]['image_path']='synthetic-page-1.png'
        tree['leaves'][0]['structure_metadata']=dict(table_id='synthetic-table',table_rows=[['Part','Voltage'],['Pump A','24 V']])
        tree['leaves'].append(dict(tree['leaves'][0],chunk_id='u3',source_chunk_id='block:30',
                                  bbox=[120,0,220,100],structure_metadata={}))
        tree['relations']=[dict(from_id='u1',to_id='u2',kind='reference')]
        tree['knowledge_graph']=dict(entities=[dict(key='a',name='Pump A',source_ids=['u1']),
                                              dict(key='b',name='Motor B',source_ids=['u2'])],
                                    relations=[dict(from_key='a',to_key='b',predicate='part_of',
                                                    source_id='u1',quote=tree['leaves'][0]['text'],
                                                    origin='synthetic_live_test')])
        tree['visual_graph']=dict(images=[dict(key='image',image_path='synthetic-live.png',sha256='synthetic',
                                              width=200,height=100,source_ids=['u1'])],
                                  objects=[dict(key='object',name='Pump A',image_key='image',bbox=[0,0,100,100],
                                                origin='synthetic_live_test',source_ids=['u1'])],relations=[],
                                  text_links=[dict(source_id='u2',object_key='object',kind='describes',
                                                   quote=tree['leaves'][1]['text'],origin='synthetic_live_test')])
        tree['visual_graph']['objects'].append(dict(key='second-object',name='Motor B',image_key='image',
                                                   bbox=[120,0,220,100],origin='synthetic_live_test',source_ids=['u1']))
        from ai_extract.services.document_graph_service import spatial_relations
        tree['visual_graph']['relations']=spatial_relations(tree['visual_graph']['objects'])
        with override_settings(A3_GRAPH_ENABLED=True, NEO4J_URI=os.getenv('NEO4J_URI',''),
                               NEO4J_USER=os.getenv('NEO4J_USER','neo4j'),
                               NEO4J_PASSWORD=os.getenv('NEO4J_PASSWORD',''),
                               NEO4J_DATABASE=os.getenv('NEO4J_DATABASE','neo4j')):
            scope,nodes,edges=graph.projection(run,tree)
            try:
                repository.replace_snapshot(scope,nodes,edges)
                repository.replace_snapshot(scope,nodes,edges)
                neighbors=repository.neighbors(scope,['u1'])
                self.assertTrue(any(r['leaf_id']=='u2' for r in neighbors))
                self.assertTrue(any(r['leaf_id']=='u2' and r['reason']=='semantic_relation' for r in neighbors))
                self.assertTrue(any(r['leaf_id']=='u2' and r['reason']=='visual_explanation' for r in neighbors))
                self.assertEqual(repository.neighbors(dict(scope,company_id=identity-1),['u1']),[])
                self.assertEqual(repository.neighbors(dict(scope,fingerprint='changed'),['u1']),[])
                context=repository.context_edges(scope,['u1','u2'])
                self.assertTrue(context['edges'])
                self.assertTrue({'CONTAINS','HAS_EVIDENCE','HAS_ROW','HAS_CELL'} <= {e['type'] for e in context['edges']})
                for field,value in [('company_id',identity-1),('contract_id',identity-1),('version_id',identity-1),('run_id',identity-1),('fingerprint','changed')]:
                    self.assertEqual(repository.context_edges(dict(scope,**{field:value}),['u1'])['edges'],[])
                preview=repository.visualization(scope, ['HAS_ROW','HAS_CELL'], page=1, limit=3)
                self.assertEqual(preview['status'], 'ready')
                self.assertEqual(len(preview['edges']),3)
                self.assertGreater(preview['matched_edges'],3)
                self.assertTrue(all(n['snapshot_key']==scope['key'] for n in preview['nodes']))
                self.assertEqual(repository.visualization(scope, ['HAS_CELL'], page=2)['edges'],[])
                self.assertEqual(preview['page_counts']['HAS_CELL'],4)
                mixed=repository.visualization(scope, ['APPEARS_IN','CONTAINS','HAS_EVIDENCE','HAS_ROW','HAS_CELL'],limit=5)
                self.assertEqual([e['type'] for e in mixed['edges']],
                                 ['APPEARS_IN','CONTAINS','HAS_EVIDENCE','HAS_ROW','HAS_CELL'])
                second=repository.visualization(scope, ['APPEARS_IN','CONTAINS','HAS_EVIDENCE','HAS_ROW','HAS_CELL'],offset=5,limit=5)
                edge_identity=lambda e:(e['from_key'],e['to_key'],e['type'])
                self.assertFalse(set(map(edge_identity,mixed['edges'])) & set(map(edge_identity,second['edges'])))
                page_graph=repository.visualization(scope, ['HAS_EVIDENCE','RELATED','CONTAINS'],page=1)
                self.assertTrue(all(n.get('page_number') in (None,1) for n in page_graph['nodes']))
                self.assertNotIn('RELATED',{e['type'] for e in page_graph['edges']})
                focused_key=scope['key']+':node:section'
                focused=repository.visualization(scope,['CONTAINS','HAS_EVIDENCE'],page=1,focus=focused_key)
                self.assertTrue(focused['edges'])
                self.assertTrue(all(focused_key in (e['from_key'],e['to_key']) for e in focused['edges']))
                self.assertTrue(any(e['type']=='HAS_EVIDENCE' for e in focused['edges']))
                self.assertEqual(repository.visualization(scope,['HAS_EVIDENCE'],focus='foreign:node:section')['edges'],[])
                for field, value in [('company_id',identity-1),('contract_id',identity-1),
                                     ('version_id',identity-1),('run_id',identity-1),('fingerprint','changed')]:
                    hidden=repository.visualization(dict(scope, **{field:value}), ['HAS_EVIDENCE'])
                    self.assertEqual(hidden['status'],'not_indexed')
                    self.assertEqual(hidden['nodes'],[])
                    self.assertEqual(hidden['counts'],{})
                with repository.connection() as session:
                    count=session.run('MATCH (n:A3Node {snapshot_key:$key}) RETURN count(n) AS count',key=scope['key']).single()['count']
                    positions=session.run('MATCH (a:A3Node {snapshot_key:$key})-[r:NEAR]->(b:A3Node) RETURN count(r) AS count',key=scope['key']).single()['count']
                    actual_counts={r['kind']:r['count'] for r in session.run(
                        'MATCH (a:A3Node {snapshot_key:$key})-[r]->(b:A3Node) RETURN type(r) AS kind, count(r) AS count',key=scope['key'])}
                self.assertEqual(count,len(nodes))
                from collections import Counter
                self.assertEqual(actual_counts, dict(Counter(e['type'] for e in edges)))
                self.assertEqual(positions,sum(e['type']=='NEAR' for e in edges))
                self.assertTrue({'HAS_ROW','HAS_CELL','NEXT','APPEARS_IN','HAS_VISUAL'}<=actual_counts.keys())
            finally:
                with repository.connection() as session:
                    def cleanup(tx):
                        tx.run('MATCH (s:A3Snapshot {key:$key, company_id:$company_id}) '
                               'MATCH (n:A3Node {snapshot_key:s.key}) DETACH DELETE n',
                               key=scope['key'],company_id=identity).consume()
                        tx.run('MATCH (s:A3Snapshot {key:$key, company_id:$company_id}) DELETE s',
                               key=scope['key'],company_id=identity).consume()
                    session.execute_write(cleanup)

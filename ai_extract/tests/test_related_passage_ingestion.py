from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock,patch

from django.test import SimpleTestCase,override_settings

from ai_extract.services.document_graph_service import infer_related_passages,projection,read_context_graph
from ai_extract.repositories import document_graph_repository as repository
from ai_extract.services.layout_service import prepare_related_passages
from ai_extract.services.inference_runtime_service import PipelineProgress
from NewIdea.services.knowledge_answer_service import bind_graph_context,answer_source_pods,render_source_pods


class RelatedPassageIngestionTests(SimpleTestCase):
    def test_table_structure_warnings_are_visible_in_llm_sources(self):
        source = dict(evidence_id='E1', text='model A model B', heading_path=[],
                      structure_metadata=dict(table_structure_status='cell_aware',
                          table_structure_review=dict(structure_warnings=['source_column_count_differs_from_reconstructed_grid'])))
        packets = answer_source_pods([source], {'nodes': [], 'edges': []})
        self.assertIn('source_column_count_differs_from_reconstructed_grid', render_source_pods(packets))

    def setUp(self):
        from ai_extract.services.gliner_identity_classifier import PASSAGE_LINK_LABELS,PASSAGE_REVIEW_SCHEMA,MODEL_ID,REVISION,passage_pair_digest
        def review(pairs):
            rows={k:.97 if k=='specific_subject_link' else .01 for k in PASSAGE_LINK_LABELS}
            return dict(schema=PASSAGE_REVIEW_SCHEMA,model=MODEL_ID,revision=REVISION,decisions=[dict(
                id=p['id'],pair_digest=passage_pair_digest(p),accepted=True,scores=[dict(rows),dict(rows)]) for p in pairs])
        self.topic_patch=patch('ai_extract.services.gliner_identity_classifier.request_passage_links',side_effect=review)
        self.topic_review=self.topic_patch.start();self.addCleanup(self.topic_patch.stop)
        self.leaves=[dict(chunk_id=str(i),source_chunk_id='block:'+str(i),page_number=i+1,
            node_id='root',ancestor_ids=['root'],heading_path=['Manual'],layout_type='paragraph',text=text)
            for i,text in enumerate(('Assembly instructions.','An explanation of assembly.','Unrelated pricing.'))]
        self.tree=dict(source_fingerprint='original',nodes=[dict(id='root',kind='document',parent_id=None)],leaves=self.leaves)
        self.backend=Mock()
        self.backend.embedding.return_value=[[1.,0.]+[0.]*1022,[.99,.1]+[0.]*1022,[0.,1.]+[0.]*1022]
        def rank(groups):
            return [[dict(c,neural_logit=2. if query.endswith(self.leaves[0]['text']) and c['chunk_id']=='1' else -2.,
                          text='Untrusted ranking payload') for c in candidates] for query,candidates in groups]
        self.backend.rerank_many.side_effect=rank

    def test_neural_scores_create_source_links_without_entity_or_llm_inference(self):
        original=deepcopy(self.tree)
        with patch('ai_extract.services.gliner_entity_service.infer_entities') as gliner, \
             patch('ai_extract.services.contract_pipeline_service.request_qwen') as llm:
            result=infer_related_passages(self.tree,self.backend,candidate_limit=2,link_limit=1)
        self.assertEqual(self.tree,original)
        self.assertEqual(len(result['links']),1)
        link=result['links'][0]
        self.assertEqual((link['from_id'],link['to_id']),('0','1'))
        self.assertEqual(link['quote'],self.leaves[0]['text'])
        self.assertEqual(link['target_quote'],self.leaves[1]['text'])
        self.assertEqual(link['review_status'],'candidate')
        self.assertNotIn('confidence',link)
        self.assertEqual(result['scoring_input'],'source_text_only')
        self.assertEqual(self.backend.embedding.call_args.args[0],[l['text'] for l in self.leaves])
        for query,candidates in self.backend.rerank_many.call_args.args[0]:
            self.assertIn(query,[l['text'] for l in self.leaves])
            self.assertTrue(all(c['heading_path']==[] for c in candidates))
        gliner.assert_not_called();llm.assert_not_called()

    def test_topic_review_withholds_surface_overlap_without_refilling_or_changing_sources(self):
        original=deepcopy(self.tree);review=self.topic_review.side_effect
        def reject(pairs):
            response=review(pairs)
            for d in response['decisions']:
                d.update(accepted=False,scores=[dict(specific_subject_link=.01,surface_overlap=.97,different_subjects=.01,uncertain=.01)]*2)
            return response
        self.topic_review.side_effect=reject
        result=infer_related_passages(self.tree,self.backend)
        self.assertEqual(result['links'],[])
        self.assertEqual(result['topic_review']['withheld'],1)
        self.assertEqual(self.tree,original)

    def test_wrong_pair_digest_or_forged_acceptance_cannot_create_edge(self):
        review=self.topic_review.side_effect
        for bad_digest in (False,True):
            def invalid(pairs):
                response=review(pairs)
                decision=response['decisions'][0]
                if bad_digest:decision['pair_digest']='different-source'
                else:decision['accepted']=False
                return response
            self.topic_review.side_effect=invalid
            with self.assertRaisesRegex(ValueError,'Ungrounded or inconsistent'):
                infer_related_passages(self.tree,self.backend)

    def test_same_block_and_self_pairs_are_not_ranked(self):
        self.leaves[1]['source_chunk_id']='block:0'
        infer_related_passages(self.tree,self.backend,candidate_limit=2,link_limit=1)
        for query,candidates in self.backend.rerank_many.call_args.args[0]:
            source=next(l for l in self.leaves if query.endswith(l['text']))
            self.assertTrue(all(c['source_chunk_id']!=source['source_chunk_id'] for c in candidates))

    def test_ingestion_rejects_identical_quotes_and_reverse_candidates_before_admission(self):
        def rank(groups):
            return [[dict(c,neural_logit=5. if {query,c['text']}==set(l['text'] for l in self.leaves[:2]) else -5.)
                     for c in candidates] for query,candidates in groups]
        self.backend.rerank_many.side_effect=rank
        result=infer_related_passages(self.tree,self.backend)
        self.assertEqual(len(result['links']),1)
        self.assertEqual(result['rejected_duplicate_candidates']['reciprocal_or_duplicate'],1)
        self.assertNotIn('removed_redundant_links',result)
        self.leaves[1]['text']='Assembly   instructions.'
        result=infer_related_passages(self.tree,self.backend)
        self.assertEqual(result['links'],[])
        self.assertEqual(result['rejected_duplicate_candidates']['identical_text'],2)

    def test_duplicate_batch_is_rejected_without_modifying_input(self):
        def link(a,b,q,t,score):return dict(from_id=a,to_id=b,quote=q,target_quote=t,rerank_logit=score)
        raw=[link('a','b','Source','Target',1.),link('b','a','Target','Source',3.),
             link('c','d','Other','Repeated',2.),link('c','e','Other','Repeated',1.)]
        original=deepcopy(raw)
        with self.assertRaisesRegex(ValueError,'reciprocal_or_duplicate'):
            repository.validate_passage_links(raw)
        self.assertEqual(raw,original)
        with self.assertRaisesRegex(ValueError,'repeated_target_text'):
            repository.validate_passage_links(raw[2:])

    def test_writer_rejects_duplicate_batch_before_opening_database_or_deleting_snapshot(self):
        valid=dict(type='RELATED_PASSAGE',from_key='a',to_key='b',quote='Source',target_quote='Target')
        cases=[[valid,dict(valid)],
               [valid,dict(valid,from_key='b',to_key='a',quote='Target',target_quote='Source')],
               [dict(valid,target_quote='Source')],
               [valid,dict(valid,to_key='c')]]
        with patch.object(repository,'connection') as connect:
            for batch in cases:
                original=deepcopy(batch)
                with self.assertRaisesRegex(ValueError,'Rejected RELATED_PASSAGE'):
                    repository.replace_snapshot({},[],batch)
                self.assertEqual(batch,original)
            connect.assert_not_called()

    def test_writer_rejects_duplicate_nodes_before_any_database_work(self):
        with patch.object(repository,'connection') as connect:
            with self.assertRaisesRegex(ValueError,'Duplicate graph node keys'):
                repository.replace_snapshot({},[dict(key='same'),dict(key='same')],[])
            connect.assert_not_called()

    def test_unreadable_and_display_only_leaves_are_excluded_before_embedding(self):
        self.leaves[1]['structure_metadata']=dict(ocr_status='failed')
        self.leaves[2]['structure_metadata']=dict(display_only=True)
        result=infer_related_passages(self.tree,self.backend)
        self.assertEqual(result['status'],'empty')
        self.backend.embedding.assert_not_called()

    def test_invalid_vectors_and_incomplete_ranking_do_not_create_links(self):
        for vectors in ([[0.]*1024]*3,[[float('nan')]*1024]*3,[[1.,0.]]*3):
            self.backend.embedding.return_value=vectors
            with self.assertRaises(ValueError):infer_related_passages(self.tree,self.backend)
        self.backend.embedding.return_value=[[1.]+[0.]*1023]*3
        self.backend.rerank_many.side_effect=lambda groups:[]
        with self.assertRaises(ValueError):infer_related_passages(self.tree,self.backend)

    def test_missing_or_foreign_ranking_ids_are_not_accepted(self):
        self.backend.rerank_many.side_effect=lambda groups:[
            [dict(chunk_id='foreign',neural_logit=4.) for c in candidates] for _,candidates in groups]
        with self.assertRaises(ValueError):infer_related_passages(self.tree,self.backend)

    def test_graph_projection_preserves_relevance_and_both_source_bindings(self):
        self.tree['related_passages']=infer_related_passages(self.tree,self.backend,candidate_limit=2,link_limit=1)
        run=SimpleNamespace(pk=1,version_id=2,version=SimpleNamespace(contract=SimpleNamespace(pk=3,company_id=4)))
        _,nodes,edges=projection(run,self.tree)
        edge=next(e for e in edges if e['type']=='RELATED_PASSAGE')
        self.assertEqual(edge['source_leaf_id'],'0')
        self.assertEqual(edge['target_source_leaf_id'],'1')
        self.assertEqual(edge['rerank_logit'],2.)
        self.assertEqual(edge['relation_semantics'],'content_similarity_navigation_only')
        reverse=dict(self.tree['related_passages']['links'][0],from_id='1',to_id='0',
                     quote=self.leaves[1]['text'],target_quote=self.leaves[0]['text'],rerank_logit=3.)
        self.tree['related_passages']['links'].append(reverse)
        original=deepcopy(self.tree)
        with self.assertRaisesRegex(ValueError,'reciprocal_or_duplicate'):
            projection(run,self.tree)
        self.assertEqual(self.tree,original)
        self.tree['related_passages']['links'].pop()
        self.tree['related_passages']['links'][0]['target_quote']='Invented source quote'
        with self.assertRaises(ValueError):projection(run,self.tree)

    def test_source_links_reach_llm_pods_as_retrieval_candidates(self):
        sources=[dict(evidence_id='E'+str(i+1),version_id=2,chunk_id=str(i),text=l['text'],heading_path=['Manual'])
                 for i,l in enumerate(self.leaves[:2])]
        context=dict(state='ready',version_id=2,nodes=[
            dict(id='a',kind='evidence',source_leaf_ids=['0']),dict(id='b',kind='evidence',source_leaf_ids=['1'])],
            edges=[dict(type='RELATED_PASSAGE',from_id='a',to_id='b',source_leaf_ids=['0','1'],
                        predicate='content_related',review_status='candidate',rerank_logit=2.,retrieval_rank=1,
                        relation_semantics='content_similarity_navigation_only',score_semantics='logit_not_probability')])
        graph=bind_graph_context([context],sources)
        edge=graph['edges'][0]
        self.assertEqual(edge['role'],'retrieval_candidate')
        self.assertIn('not a verified reference',graph['relation_legend']['RELATED_PASSAGE'])
        packets=answer_source_pods(sources,graph)
        self.assertEqual(packets['source_links'][0]['from_sources'],['E1'])
        self.assertEqual(packets['source_links'][0]['to_sources'],['E2'])
        text=render_source_pods(packets)
        self.assertIn('RELATED_PASSAGE',text)
        self.assertIn('content_similarity_navigation_only',text)
        self.assertIn('Assembly instructions.',text)
        self.assertIn('An explanation of assembly.',text)

    @override_settings(A3_GRAPH_ENABLED=True)
    def test_graph_read_keeps_zero_relevance_scores(self):
        self.tree['related_passages']=infer_related_passages(self.tree,self.backend,candidate_limit=2,link_limit=1)
        self.tree['related_passages']['links'][0].update(dense_similarity=0.,rerank_logit=0.)
        self.tree['graph_index']=dict(status='ready',fingerprint='original')
        run=SimpleNamespace(pk=1,version_id=2,version=SimpleNamespace(contract=SimpleNamespace(pk=3,company_id=4)))
        _,_,edges=projection(run,self.tree)
        stored=[e for e in edges if e['type']=='RELATED_PASSAGE']
        with patch('ai_extract.services.document_graph_service.repository.context_edges',
                   return_value=dict(edges=stored,truncated=False)):
            context=read_context_graph(run,self.tree,self.leaves[:2])
        link=next(e for e in context['edges'] if e['type']=='RELATED_PASSAGE')
        self.assertEqual(link['rerank_logit'],0.)
        self.assertEqual(link['dense_similarity'],0.)
        sources=[dict(l,evidence_id='E'+str(i+1),version_id=2) for i,l in enumerate(self.leaves[:2])]
        packets=answer_source_pods(sources,bind_graph_context([context],sources))
        self.assertEqual(packets['source_links'][0]['rerank_logit'],0.)
        self.assertIn('undirected_navigation',packets['source_links'][0]['direction_semantics'])

    def test_failed_neural_stage_is_explicit_and_keeps_original_sources(self):
        run=SimpleNamespace(id=1,total_pages=1,structure=self.tree)
        original=deepcopy(self.leaves)
        with patch('ai_extract.services.document_graph_service.infer_related_passages',side_effect=RuntimeError('Unavailable')), \
             patch('ai_extract.services.layout_service.layout_repository.save_run') as save:
            prepare_related_passages(run,PipelineProgress(1))
        self.assertEqual(run.structure['related_passages']['status'],'unavailable')
        self.assertEqual(run.structure['related_passages']['links'],[])
        self.assertEqual(self.leaves,original)
        save.assert_called_once_with(run,['structure'])

    @override_settings(A3_GRAPH_ENABLED=True)
    def test_similarity_expansion_needs_query_relevance_but_source_context_is_preserved(self):
        from ai_extract.services.contract_pipeline_service import expand_retrieval_context
        self.tree['graph_index']=dict(status='ready',fingerprint='original')
        run=SimpleNamespace(pk=1,version_id=2,version=SimpleNamespace(contract=SimpleNamespace(pk=3,company_id=4)))
        rows=[dict(seed_id='0',leaf_id='1',reason='retrieval_similarity'),
              dict(seed_id='0',leaf_id='2',reason='direct_structure_relation')]
        for scores,expected in (([-3.],{'0','2'}),([3.],{'0','1','2'}),([],{'0','2'}),(None,{'0','2'})):
            backend=Mock()
            if scores is None:backend.rerank.side_effect=RuntimeError('Offline')
            else:backend.rerank.side_effect=lambda query,items:[dict(c,neural_logit=s) for c,s in zip(items,scores)]
            with patch('ai_extract.services.contract_pipeline_service.layout_repository.latest_run',return_value=run), \
                 patch('ai_extract.services.document_graph_service.repository.neighbors',return_value=rows), \
                 patch('ai_extract.services.document_graph_service.read_context_graph',return_value=dict(nodes=[],edges=[])):
                selected,coverage=expand_retrieval_context('Assembly',self.tree,run.version,[self.leaves[0]],{},backend=backend)
            self.assertEqual({s['chunk_id'] for s in selected},expected)
            self.assertEqual(self.tree['leaves'],self.leaves)

    @override_settings(A3_GRAPH_ENABLED=True)
    def test_structural_support_also_reaching_similarity_target_does_not_need_query_gate(self):
        from ai_extract.services.document_graph_service import expand_graph
        self.tree['graph_index']=dict(status='ready',fingerprint='original')
        run=SimpleNamespace(pk=1,version_id=2,version=SimpleNamespace(contract=SimpleNamespace(pk=3,company_id=4)))
        rows=[dict(seed_id='0',leaf_id='1',reason=reason) for reason in ('retrieval_similarity','direct_structure_relation')]
        with patch('ai_extract.services.document_graph_service.repository.neighbors',return_value=rows):
            tree,_=expand_graph(run,self.tree,[self.leaves[0]])
        self.assertEqual(tree['leaves'][0]['graph_related_chunk_ids'],['1'])
        self.assertEqual(tree['leaves'][0]['graph_similarity_only_chunk_ids'],[])

    @override_settings(A3_RELATED_PASSAGES_ENABLED=True)
    def test_pregraph_pipeline_persists_links_before_complete_proof(self):
        from ai_extract.services.layout_service import prepare_document
        run=SimpleNamespace(id=1,total_pages=1,structure=self.tree)
        def reading(run):run.structure['reading_analysis']=dict(status='complete',source_ids=[])
        identity=dict(status='complete',mentions=0,linked=0,unresolved=0,canonical_entities=0)
        knowledge=dict(status='complete',entities=[],relations=[],source_relations=[],source_coverage=dict(completed_ids=[]))
        with patch('ai_extract.services.document_structure_service.prepare_reading',side_effect=reading), \
             patch('ai_extract.services.gliner_entity_service.extract_graph_entities',return_value=knowledge), \
             patch('ai_extract.services.gliner_identity_classifier.resolve_identities',return_value=identity), \
             patch('ai_extract.services.document_graph_service.infer_related_passages',return_value=dict(status='complete',links=[])) as related, \
             patch('ai_extract.services.layout_service.layout_repository.save_run'):
            prepare_document(run,PipelineProgress(1))
        related.assert_called_once()
        self.assertEqual(self.tree['related_passages']['status'],'complete')
        self.assertIn('related_passages',self.tree['pregraph_analysis']['stages'])

import tempfile
import hashlib
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
from PIL import Image
from django.test import SimpleTestCase,override_settings
from ai_extract.services.document_graph_service import valid_bbox,spatial_relations,visual_sources_fresh
from ai_extract.services.document_graph_service import projection,expand_graph
from ai_extract.tests.test_document_graph import fixture


@override_settings(A3_KNOWLEDGE_ENABLED=True)
class VisualRelationsTests(SimpleTestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.media=override_settings(MEDIA_ROOT=self.directory.name)
        self.media.enable(); self.addCleanup(self.media.disable)
        Image.new('RGB',(200,100),'white').save(Path(self.directory.name)/'page.png')
        self.run,self.tree=fixture()
        for leaf in self.tree['leaves']: leaf['image_path']='page.png'
        # Historical visual observations remain readable after retiring their
        # inference adapter. These fixtures exercise projection and freshness.
        objects=[dict(key=key,name=name,bbox=box,image_key='image',source_ids=['u1'],
                      origin='vision_model_proposal',review_status='unreviewed')
                 for key,name,box in [('a','Pump A',[10,10,100,100]),('b','Motor B',[120,10,220,100])]]
        self.tree['visual_graph']=dict(images=[dict(key='image',image_path='page.png',width=200,height=100,
            sha256=hashlib.sha256((Path(self.directory.name)/'page.png').read_bytes()).hexdigest(),source_ids=['u1'])],
            objects=objects,relations=spatial_relations(objects),text_links=[dict(source_id='u1',object_key='a',
                kind='describes',quote='Pump A contains Motor B.',origin='multimodal_model_proposal_source_aligned')])


    def test_stored_visual_observations_preserve_projection_and_spatial_provenance(self):
        result=self.tree['visual_graph']
        self.assertEqual({r['kind'] for r in result['relations']},{'LEFT_OF'})
        self.assertTrue(all(o['review_status']=='unreviewed' for o in result['objects']))
        self.tree['visual_graph']=result
        _,nodes,edges=projection(self.run,self.tree)
        self.assertEqual(len([n for n in nodes if n['category']=='visual_object']),2)
        self.assertTrue({'APPEARS_IN','HAS_VISUAL','EXPLAINS','LEFT_OF'}<={e['type'] for e in edges})

    def test_invalid_spatial_boxes_are_rejected(self):
        for box in ([0,0,0,1],[-1,0,20,20],[0,0,1001,20],[0,0,float('nan'),20],[False,0,20,20]):
            with self.assertRaises(ValueError): valid_bbox(box)


    def test_geometry_does_not_assert_connectivity_and_overlap_is_not_near(self):
        a=dict(key='a',bbox=[0,0,100,100]); b=dict(key='b',bbox=[50,50,150,150])
        self.assertEqual(spatial_relations([a,b]),[])
        b['bbox']=[0,200,100,300]
        self.assertEqual(spatial_relations([a,b],80),[])
        self.assertEqual({r['kind'] for r in spatial_relations([a,b],120)},{'ABOVE'})

    def test_diagonal_and_distant_boxes_have_no_directional_edges(self):
        a=dict(key='a',bbox=[0,0,100,100])
        diagonal=dict(key='diagonal',bbox=[120,120,220,220])
        far=dict(key='far',bbox=[400,0,500,100])
        self.assertEqual({r['kind'] for r in spatial_relations([a,diagonal])},{'NEAR'})
        self.assertEqual(spatial_relations([a,far]),[])
        sliver=dict(key='sliver',bbox=[120,99,220,199])
        self.assertNotIn('LEFT_OF',{r['kind'] for r in spatial_relations([a,sliver])})

    def test_intervening_block_prevents_skipped_horizontal_and_vertical_links(self):
        row=[dict(key=str(i),bbox=[i*30,0,i*30+20,20]) for i in range(3)]
        links={(r['from_key'],r['to_key']) for r in spatial_relations(row) if r['kind']=='LEFT_OF'}
        self.assertEqual(links,{('0','1'),('1','2')})
        column=[dict(key=str(i),bbox=[0,i*30,20,i*30+20]) for i in range(3)]
        links={(r['from_key'],r['to_key']) for r in spatial_relations(column) if r['kind']=='ABOVE'}
        self.assertEqual(links,{('0','1'),('1','2')})

    def test_local_graph_is_sparse_and_independent_of_input_order(self):
        rows=[dict(key=str(i),bbox=[i*25,0,i*25+20,20]) for i in range(20)]
        relations=spatial_relations(rows)
        self.assertEqual(sum(r['kind']=='LEFT_OF' for r in relations),19)
        degree={r['key']:0 for r in rows}
        for r in relations:
            if r['kind']=='NEAR':
                degree[r['from_key']]+=1; degree[r['to_key']]+=1
        self.assertLessEqual(max(degree.values()),2)
        self.assertEqual(relations,spatial_relations(list(reversed(rows))))

    def test_changed_image_blocks_graph_expansion(self):
        self.assertTrue(visual_sources_fresh(self.tree))
        Image.new('RGB',(200,100),'black').save(Path(self.directory.name)/'page.png')
        self.assertFalse(visual_sources_fresh(self.tree))
        self.tree['graph_index']=dict(status='ready',fingerprint='abc')
        with override_settings(A3_GRAPH_ENABLED=True),patch('ai_extract.repositories.document_graph_repository.neighbors') as read:
            _,diagnostic=expand_graph(self.run,self.tree,[self.tree['leaves'][0]])
        self.assertEqual(diagnostic['state'],'stale_visual_source'); read.assert_not_called()



    def test_page_furniture_keeps_layout_blocks_and_neighbor_edges(self):
        from ai_extract.services.document_graph_service import append_layout_edges
        leaves=[dict(chunk_id=kind,source_chunk_id=kind,layout_type=kind,page_number=1,
                     bbox=box,text=kind,structure_metadata=dict(reading_order=i,reading_order_source='surya_full_page_model'))
                for i,(kind,box) in enumerate([('page_header',[0,0,100,10]),('title',[0,20,100,30]),
                                              ('paragraph',[0,40,100,80]),('page_footer',[0,90,100,100])])]
        nodes={};edges=[]
        append_layout_edges(dict(leaves=leaves),lambda identity,**props:nodes.update({identity:props}),
            lambda a,b,kind,**props:edges.append((a,b,kind)))
        self.assertIn('layout-block:1:page_header',nodes)
        self.assertIn('layout-block:1:page_footer',nodes)
        self.assertIn(('layout-block:1:title','layout-block:1:paragraph','NEXT'),edges)

    def test_ner_only_withholds_source_to_source_semantics_even_with_both_quotes(self):
        from ai_extract.services.gliner_entity_service import extract_graph_entities,mention_key
        a,b=self.tree['leaves']
        entities=[dict(source_id=leaf['chunk_id'],start=0,end=len(name),name=name,entity_type='ENTITY',confidence=.95)
                  for leaf,name in ((a,'Pump A'),(b,'Motor B'))]
        relation=dict(source_id='u1',target_source_id='u2',from_key=mention_key(entities[0]),to_key=mention_key(entities[1]),
            predicate='supports',quote=a['text'],target_quote=b['text'],confidence=.95)
        proposal=dict(entities=entities,relations=[relation])
        units=[dict(id='u1',text=a['text'],context_windows=[dict(offset=0,text=a['text'],
            contexts=[dict(id='u2',offset=0,text=b['text'])])])]
        result=extract_graph_entities(self.tree,lambda batch:proposal,context_units=units)
        self.assertEqual(result['relations'],[])
        self.assertEqual(result['status'],'unavailable')
        self.tree['knowledge_graph']=result
        _,_,edges=projection(self.run,self.tree)
        self.assertFalse(any(e['type']=='SUPPORTS' for e in edges))
        invalid=deepcopy(proposal); invalid['relations'][0]['target_quote']='fake'
        rejected=extract_graph_entities(self.tree,lambda batch:invalid,context_units=units)
        self.assertEqual(rejected['status'],'unavailable')
        self.assertEqual(rejected['source_relations'],[])

    def test_embedded_image_objects_are_not_drawn_in_page_coordinate_frame(self):
        from ai_extract.services.graph_visualization_service import page_payload
        from types import SimpleNamespace
        page=SimpleNamespace(page_number=1,image_path='different-page.png',blocks=SimpleNamespace(all=lambda:[]))
        result=page_payload(self.run,page)
        self.assertTrue(result['visual']['objects'])
        self.assertTrue(all(not o['overlay_on_page'] for o in result['visual']['objects']))

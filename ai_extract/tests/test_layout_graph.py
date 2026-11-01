from django.test import SimpleTestCase
from ai_extract.services.document_graph_service import append_layout_edges
from ai_extract.services.document_structure_service import analyze_document_structure
from importlib.util import find_spec
from unittest import TestCase, skipUnless



class LayoutGraphTests(SimpleTestCase):
    def test_unresolved_table_keeps_source_layout_without_publishing_candidate_cells(self):
        from ai_extract.services.document_structure_service import table_context
        leaf=dict(chunk_id='source',source_chunk_id='block:1',text='Original table transcription',
            page_number=1,bbox=[0,0,500,500],image_path='p1.png',structure_metadata=dict(
                ocr_status='completed',table_structure_status='unresolved',
                table_rows=[['Wrong','pair']],table_headers=['A','B'],table_values=['Wrong','pair']))
        nodes,edges=[],[]
        append_layout_edges(dict(leaves=[leaf]),lambda k,**p:nodes.append(p),
            lambda a,b,k,**p:edges.append((a,b,k)))
        self.assertFalse(any(p.get('category') in {'table_cell','table_row'} for p in nodes))
        self.assertFalse(any(k in {'HAS_ROW','HAS_CELL'} for a,b,k in edges))
        self.assertTrue(any(k=='HAS_EVIDENCE' for a,b,k in edges))
        self.assertTrue(any(k=='HAS_VISUAL' for a,b,k in edges))
        self.assertEqual(table_context(leaf),{})
        self.assertEqual(leaf['text'],'Original table transcription')

    def test_skipped_image_keeps_node_ownership_and_source_image_relations(self):
        records=[dict(chunk_id='image',text='UI image label',layout_type='figure',page_number=1,
            image_path='p1.png',bbox=[0,0,100,100],structure_metadata=dict(
                ocr_status='skipped',surya_region_id='image',chunk_policy='surya_layout_region',
                reading_order=0,reading_order_source='surya_full_page_model')),
            dict(chunk_id='child',text='RC8',layout_type='paragraph',page_number=1,
                image_path='p1.png',bbox=[10,10,20,20],structure_metadata=dict(
                    ocr_status='completed',surya_region_id='child',surya_parent_region_id='image',
                    chunk_policy='surya_layout_region',reading_order=0,reading_order_source='surya_full_page_model'))]
        tree=analyze_document_structure(records,'Manual')
        image,child=tree['leaves']
        self.assertEqual(image['text'],'')
        self.assertEqual(child['text'],'RC8')
        self.assertEqual(child['parent_reading_stream_id'],image['reading_stream_id'])
        from ai_extract.tests.test_reading_streams import project
        _,nodes,edges=project(tree)
        self.assertTrue(any(n.get('kind')=='figure' for n in nodes))
        self.assertTrue(any(e['type']=='CONTAINS' for e in edges))
        self.assertTrue(any(e['type']=='HAS_VISUAL' for e in edges))
        self.assertTrue(any(e['type']=='APPEARS_IN' for e in edges))
    def test_model_reading_order_keeps_left_column_content_under_its_heading(self):
        records=[dict(chunk_id='h5',text='5. Optimization',page_number=7,image_path='p7.png'),
                 dict(chunk_id='h51',text='5.1\nOptimizations',page_number=8,image_path='p8.png'),
                 dict(chunk_id='h6',text='6.\nImplementation',page_number=8,image_path='p8.png'),
                 dict(chunk_id='dead',text='Dead data-stack store elimination.',page_number=8,image_path='p8.png'),
                 dict(chunk_id='impl',text='We implement compiled traces.',page_number=8,image_path='p8.png')]
        records=[records[i] for i in (0,1,3,2,4)]
        for i,r in enumerate(records):
            r.update(layout_type='title' if r['chunk_id'].startswith('h') else 'paragraph',
                structure_metadata=dict(layout_source='surya',reading_order=i,
                                        heading_level=2 if r['chunk_id']=='h51' else 1 if r['chunk_id'].startswith('h') else None))
        tree=analyze_document_structure(records,'Manual')
        leaf=next(l for l in tree['leaves'] if l['source_chunk_id']=='dead')
        self.assertEqual(leaf['heading_path'],['Manual','5. Optimization','5.1\nOptimizations'])
        impl=next(l for l in tree['leaves'] if l['source_chunk_id']=='impl')
        self.assertEqual(impl['heading_path'],['Manual','6.\nImplementation'])
        self.assertEqual([l['source_chunk_id'] for l in tree['leaves']],['h5','h51','dead','h6','impl'])

    def test_split_table_block_does_not_duplicate_containment(self):
        edges=[]
        leaves=[dict(chunk_id=k,source_chunk_id='block:1',page_number=1,bbox=[0,0,100,100],
                     structure_metadata=dict(table_id='table',table_rows=[['A','24 V']])) for k in ('a','b')]
        append_layout_edges(dict(leaves=leaves),lambda *a,**kw:None,lambda a,b,k,**p:edges.append((a,b,k)))
        self.assertEqual(sum(k=='CONTAINS' for a,b,k in edges),1)
        self.assertEqual(sum(k=='HAS_EVIDENCE' and a=='layout-table:1:table' for a,b,k in edges),2)

    def test_bbox_edges_do_not_become_semantic_relations(self):
        leaves = [dict(chunk_id='u1',source_chunk_id='block:1',text='caption',page_number=1,bbox=[0,0,100,100],image_path='p1.png'),
                  dict(chunk_id='u2',source_chunk_id='block:2',text='figure',page_number=1,bbox=[120,0,220,100],image_path='p1.png'),
                  dict(chunk_id='u3',source_chunk_id='block:3',text='other',page_number=2,bbox=[0,0,100,100],image_path='p2.png')]
        nodes,edges=[],[]
        append_layout_edges(dict(leaves=leaves),lambda k,**p:nodes.append((k,p)),lambda a,b,k,**p:edges.append((a,b,k)))
        kinds={e[2] for e in edges}
        self.assertTrue({'HAS_VISUAL','APPEARS_IN','LEFT_OF','NEXT'}<=kinds)
        self.assertNotIn('NEAR', kinds)  # LEFT_OF already records this local pair.
        self.assertFalse({'EXPLAINS','RELATED'}&kinds)
        self.assertFalse(any('block:3' in a+b for a,b,k in edges if k in {'LEFT_OF','ABOVE','NEAR','NEXT'}))

    def test_table_rows_and_cells_are_source_values(self):
        nodes,edges=[],[]
        append_layout_edges(dict(leaves=[dict(chunk_id='u1',text='24 V',source_chunk_id='b1',
            structure_metadata=dict(table_id='t1',table_rows=[['Device','Voltage'],['A','24 V']]))]),
            lambda k,**p:nodes.append((k,p)),lambda a,b,k,**p:edges.append((a,b,k)))
        self.assertEqual(sum(k=='HAS_ROW' for a,b,k in edges),2)
        self.assertEqual(sum(k=='HAS_CELL' for a,b,k in edges),4)
        self.assertIn('24 V',[p.get('text') for k,p in nodes])
        self.assertFalse(any(k=='HAS_EVIDENCE' and ':row:' in a for a,b,k in edges))

    def test_fragmented_table_retains_exact_row_sources(self):
        edges=[]
        leaves=[dict(chunk_id=key, source_chunk_id=key, structure_metadata=dict(
            table_id='t', table_row=i, table_rows=[[key]])) for i,key in enumerate(('a','b'),1)]
        append_layout_edges(dict(leaves=leaves),lambda *a,**p:None,
                            lambda a,b,k,**p:edges.append((a,b,k)))
        self.assertIn(('layout-table:1:t:row:1','leaf:a','HAS_EVIDENCE'),edges)
        self.assertIn(('layout-table:1:t:row:2','leaf:b','HAS_EVIDENCE'),edges)

    def test_model_table_grid_keeps_provenance_and_review_status(self):
        nodes,edges=[],[]
        append_layout_edges(dict(leaves=[dict(chunk_id='u1',text='24 V',structure_metadata=dict(
            table_id='t',table_rows=[['24 V']],table_header_origin='granite_first_row_unverified'))]),
            lambda k,**p:nodes.append(p),lambda a,b,k,**p:edges.append(dict(kind=k,**p)))
        cell=next(n for n in nodes if n['category']=='table_cell')
        self.assertEqual(cell['origin'],'granite_table_values_source_aligned')
        self.assertEqual(cell['review_status'],'unreviewed')
        self.assertEqual(next(e for e in edges if e['kind']=='HAS_CELL')['review_status'],'unreviewed')

    def test_source_structure_keeps_text_and_has_no_semantic_links(self):
        records=[dict(chunk_id='b1',text='Maintenance',layout_type='paragraph',page_number=1,image_path='p1.png',bbox=[0,0,200,100])]
        tree=analyze_document_structure(records,'Manual')
        self.assertEqual(tree['analysis']['provider'],'source')
        self.assertEqual(tree['leaves'][0]['text'],'Maintenance')
        self.assertFalse(tree.get('relations',[]))

    def test_every_segment_retains_source_block_evidence(self):
        edges=[]
        leaves=[dict(chunk_id=k, source_chunk_id='same', page_number=1, bbox=[0,0,100,100], text=k) for k in ('a','b')]
        append_layout_edges(dict(leaves=leaves),lambda *a,**kw:None,lambda a,b,k,**p:edges.append((a,b,k)))
        self.assertEqual({b for a,b,k in edges if k=='HAS_EVIDENCE'}, {'leaf:a','leaf:b'})

    def test_source_structure_preserves_table_values(self):
        records=[dict(chunk_id='b1',text='Motor 24 V',layout_type='table',page_number=1,image_path='p1.png',bbox=[0,0,200,100],
            structure_metadata=dict(table_rows=[['Motor','24 V']]))]
        tree=analyze_document_structure(records,'Manual')
        self.assertEqual(tree['leaves'][0]['structure_metadata']['table_rows'],[['Motor','24 V']])

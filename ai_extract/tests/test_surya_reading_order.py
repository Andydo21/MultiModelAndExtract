"""Native Surya order must survive persistence without inventing page/crop NEXT."""
from types import SimpleNamespace as NS
from django.test import SimpleTestCase
from ai_extract.services.surya_inference_service import native_reading_order
from ai_extract.services.document_structure_service import build_document_structure, order_source_regions
from ai_extract.services.document_graph_service import append_layout_edges


class NativeReadingOrderTests(SimpleTestCase):
    def record(self, identity, rank, page=1, parent=None, box=None):
        return dict(chunk_id=identity,source_chunk_id=identity,text='Source '+identity,
                    page_number=page,image_path=f'page{page}.png',layout_type='paragraph',
                    bbox=box or [10,10,100,50],
                    structure_metadata=dict(surya_region_id=identity,surya_parent_region_id=parent,
                        chunk_policy='surya_layout_region',reading_order=rank,
                        reading_order_source='surya_layout_model',reading_order_scope=parent or 'page',
                        reading_order_unresolved=False,reading_order_status='model_predicted'))

    def next_edges(self, records):
        edges=[]
        append_layout_edges(dict(leaves=records),lambda *a,**kw:None,
                            lambda a,b,k,**props:edges.append((a,b,k,props)))
        return [(a,b,p['origin']) for a,b,k,p in edges if k=='NEXT']

    def test_native_order_uses_model_ranks_not_vertical_position(self):
        blocks=[NS(reading_order=2,bbox=[200,0,300,50]),
                NS(reading_order=0,bbox=[0,0,100,50]),
                NS(reading_order=1,bbox=[0,200,100,250])]
        ordered=native_reading_order(blocks)
        self.assertEqual([b.reading_order for b,_ in ordered],[0,1,2])
        self.assertTrue(all(m['reading_order_status']=='model_predicted' for _,m in ordered))

    def test_bad_model_ranks_keep_evidence_but_mark_whole_scope_unresolved(self):
        for ranks in ([0,0],[None,1],[-1,0],[True,2]):
            ordered=native_reading_order([NS(reading_order=r) for r in ranks])
            self.assertEqual(len(ordered),2)
            self.assertTrue(all(m['reading_order_unresolved'] for _,m in ordered))

    def test_stored_rows_are_sorted_before_heading_assignment(self):
        heading=self.record('h',0); heading.update(text='1. Payment',layout_type='title')
        content=self.record('p',1)
        tree=build_document_structure([content,heading],'Contract')
        self.assertEqual([l['source_chunk_id'] for l in tree['leaves']],['h','p'])
        self.assertIn('1. Payment',tree['leaves'][1]['heading_path'])

    def test_main_page_and_image_text_have_separate_next_sequences(self):
        a=self.record('a',0)
        picture=self.record('picture',1); picture['layout_type']='figure'
        after=self.record('after',2)
        x=self.record('x',0,parent='picture')
        y=self.record('y',1,parent='picture')
        page2=self.record('other-page',0,page=2)
        records=[after,y,page2,a,x,picture]
        ordered=order_source_regions(records)
        self.assertEqual([r['chunk_id'] for r in ordered],['a','picture','x','y','after','other-page'])
        links=self.next_edges(ordered)
        self.assertEqual({(a,b) for a,b,_ in links},{
            ('layout-block:1:a','layout-block:1:picture'),
            ('layout-block:1:picture','layout-block:1:after'),
            ('layout-block:1:x','layout-block:1:y')})
        self.assertTrue(all(origin=='surya_layout_model' for _,_,origin in links))

    def test_duplicate_rank_in_graph_cannot_turn_into_next_from_input_order(self):
        a,b=self.record('a',0),self.record('b',0)
        self.assertEqual(self.next_edges([a,b]),[])
        ordered=order_source_regions([a,b])
        self.assertTrue(all(r['structure_metadata']['reading_order_unresolved'] for r in ordered))

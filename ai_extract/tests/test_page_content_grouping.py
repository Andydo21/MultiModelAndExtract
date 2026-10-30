"""Coordinates and punctuation must not invent structure or reading order."""
from copy import deepcopy
from django.test import SimpleTestCase
from ai_extract.services.document_structure_service import annotate_page_content, build_document_structure
from ai_extract.services.document_graph_service import append_layout_edges


def block(key,text,box,**metadata):
    return dict(chunk_id=key,text=text,bbox=box,page_number=1,image_path='p.png',
                layout_type='paragraph',structure_metadata=metadata)


class PageContentGroupingTests(SimpleTestCase):
    def columns(self):
        return [block('l1','The machine requires',[50,100,430,120]),
                block('r1','Another column.',[560,100,940,120]),
                block('l2','regular maintenance.',[50,125,430,145]),
                block('r2','Last item.',[560,150,940,170])]

    def test_bbox_cannot_override_input_order_or_join_source_regions(self):
        rows=self.columns(); original=deepcopy(rows)
        result=annotate_page_content(rows)
        self.assertEqual(rows,original)
        self.assertEqual(result,rows)
        tree=build_document_structure(rows)
        self.assertEqual([l['source_chunk_id'] for l in tree['leaves']],['l1','r1','l2','r2'])
        self.assertTrue(all('paragraph_group_id' not in l['structure_metadata'] for l in tree['leaves']))

    def test_supplied_model_order_and_ambiguity_are_preserved(self):
        rows=[block('a','Text',[0,0,100,20],reading_order=2,reading_order_source='surya_layout_model'),
              block('b','Text',[0,30,100,50],reading_order=1,reading_order_unresolved=True)]
        self.assertEqual(annotate_page_content(rows),rows)

    def test_old_inferred_groups_are_removed_without_mutating_source(self):
        rows=[block('a','Text',[0,0,100,20],paragraph_group_id='old',paragraph_group_status='likely_continuation',
                    layout_column=0,layout_column_count=2,layout_region_source='observed_bbox_gutters')]
        result=annotate_page_content(rows)
        self.assertEqual(result[0]['structure_metadata'],{})
        self.assertEqual(rows[0]['structure_metadata']['paragraph_group_id'],'old')

    def test_old_geometry_order_does_not_become_a_model_prediction(self):
        rows=[block('a','Text',[0,0,100,20],reading_order=1,reading_order_source='bbox_column_bands')]
        metadata=annotate_page_content(rows)[0]['structure_metadata']
        self.assertTrue(metadata['reading_order_unresolved'])
        self.assertIsNone(metadata['reading_order'])

    def test_graph_keeps_source_evidence_without_invented_paragraph_groups(self):
        tree=build_document_structure(self.columns())
        nodes,edges=[],[]
        append_layout_edges(tree,lambda k,**p:nodes.append((k,p)),lambda a,b,k,**p:edges.append((a,b,k)))
        self.assertFalse(any(p['category']=='paragraph_group' for _,p in nodes))
        self.assertEqual(sum(k=='HAS_EVIDENCE' for _,_,k in edges),4)

    def test_missing_bbox_or_font_changes_do_not_drop_or_merge_content(self):
        rows=[block('a','unfinished',[0,0,100,20],font_size=10),
              block('b','continuation',[0,25,100,45],font_size=14),
              dict(chunk_id='c',text='Without position',page_number=2)]
        result=annotate_page_content(rows)
        self.assertEqual([r['text'] for r in result],[r['text'] for r in rows])
        self.assertTrue(all('paragraph_group_id' not in r['structure_metadata'] for r in result))

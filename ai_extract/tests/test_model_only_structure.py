from django.test import SimpleTestCase
from ai_extract.services.document_structure_service import build_document_structure, HierarchicalChunker


class ModelOnlyStructureTests(SimpleTestCase):
    def test_keywords_numbers_and_markdown_cannot_classify_plain_text(self):
        text='Article 1. Payment\n1.1 Details\na) Item\n# Heading\nChapter 2. Scope\nĐiều 3. Thời hạn'
        tree=build_document_structure([dict(chunk_id='x',text=text,layout_type='paragraph')])
        self.assertEqual(len(tree['nodes']),1)
        self.assertEqual(tree['leaves'][0]['text'],text)

    def test_unknown_heading_levels_do_not_turn_numbering_into_nested_hierarchy(self):
        tree=build_document_structure([dict(chunk_id='a',text='5. 運転',layout_type='title'),
            dict(chunk_id='b',text='5.1 起動',layout_type='title'),dict(chunk_id='c',text='操作方法',page_number=2)])
        self.assertEqual([n['parent_id'] for n in tree['nodes'][1:]],['root','root'])
        self.assertTrue(all(n['heading_level'] is None for n in tree['nodes'][1:]))
        self.assertEqual(tree['leaves'][-1]['heading_path'],['Document','5.1 起動'])

    def test_explicit_levels_create_cross_page_hierarchy_without_reading_title_text(self):
        tree=build_document_structure([
            dict(chunk_id='a',text='運転',layout_type='title',heading_level=1,page_number=1),
            dict(chunk_id='b',text='起動',layout_type='title',heading_level=2,page_number=1),
            dict(chunk_id='c',text='操作方法',page_number=2)])
        self.assertEqual(tree['nodes'][2]['parent_id'],tree['nodes'][1]['id'])
        self.assertEqual(tree['leaves'][-1]['heading_path'],['Document','運転','起動'])

    def test_chunker_does_not_invent_a_heading_level(self):
        chunks=HierarchicalChunker.process_hierarchical_chunks([dict(layout_type='title',text='起動')])
        self.assertNotIn('heading_level',chunks[0])

    def test_source_text_is_not_stripped_by_a_prefix_pattern(self):
        text='[a|b]\nOriginal source'
        tree=build_document_structure([dict(chunk_id='x',text=text)])
        self.assertEqual(tree['leaves'][0]['text'],text)

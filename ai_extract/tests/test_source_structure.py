from unittest.mock import patch
from django.test import SimpleTestCase
from ai_extract.services.document_structure_service import analyze_document_structure, source_fingerprint


class SourceStructureTests(SimpleTestCase):
    def test_duplicate_source_id_is_rejected_before_structure_can_overwrite_ownership(self):
        records=[dict(chunk_id='same',text=text,layout_type='paragraph') for text in ('First','Second')]
        with self.assertRaisesRegex(ValueError,'Duplicate structure source IDs'):
            analyze_document_structure(records,'Document')
    def test_structure_uses_observed_headings_without_qwen(self):
        records=[dict(chunk_id='h',text='1. Payment',layout_type='title',page_number=1),
                 dict(chunk_id='body',text='Pay within 30 days.',layout_type='paragraph',page_number=2)]
        with patch('ai_extract.services.contract_pipeline_service.request_qwen') as qwen:
            tree=analyze_document_structure(records,'Contract')
        qwen.assert_not_called()
        self.assertEqual(tree['analysis']['provider'],'source')
        leaf=next(l for l in tree['leaves'] if l['source_chunk_id']=='body')
        self.assertEqual(leaf['text'],'Pay within 30 days.')
        self.assertEqual(leaf['heading_path'],['Contract','1. Payment'])

    def test_fingerprint_changes_with_source_text_or_bbox(self):
        original=dict(chunk_id='a',text='Original',bbox=[0,0,100,100])
        fingerprint=source_fingerprint([original])
        self.assertNotEqual(fingerprint,source_fingerprint([dict(original,text='Changed')]))
        self.assertNotEqual(fingerprint,source_fingerprint([dict(original,bbox=[0,0,200,100])]))

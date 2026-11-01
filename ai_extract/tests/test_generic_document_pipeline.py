"""Real Office containers and pixels; these tests do not measure model accuracy."""
import io
import tempfile
from pathlib import Path
from zipfile import ZipFile

from django.test import SimpleTestCase, override_settings
from PIL import Image

from ai_extract.services.document_structure_service import build_document_structure
from ai_extract.services.layout_extractor import extraction_metadata
from ai_extract.services.office_parser import extract_docx, extract_pptx, extract_spreadsheet, NS
from ai_extract.services.contract_pipeline_service import structural_retrieve, review_batches, retrieval_gate


class GenericDocumentTests(SimpleTestCase):
    def test_model_body_text_does_not_become_a_heading_from_reference_words(self):
        records=[dict(chunk_id='heading',text='References',layout_type='title',
                      structure_metadata=dict(granite_observation={'kind':'section_header'}))]
        for i,text in enumerate(('Section 9 and an outlook on future work is presented in Section 10.',
                                 'part of recording LIR. All of these optimizations reduce complex constructs.',
                                 '300. ACM New York, NY, USA, 1998.')):
            records.append(dict(chunk_id=str(i),text=text,layout_type='paragraph'))
        tree=build_document_structure(records)
        self.assertEqual([n['title'] for n in tree['nodes'][1:]],['References'])
        self.assertTrue(all(l['node_id']=='n1' for l in tree['leaves']))

    def test_technical_structure_and_cross_reference_include_original_image(self):
        tree = build_document_structure([
            dict(chunk_id='intro', text='Installation',layout_type='title',heading_level=1,page_number=1),
            dict(chunk_id='caption', text='Figure 2. Terminal wiring', layout_type='figure_caption', page_number=2,
                 bbox=[100,360,400,390],image_path='extracted_images/wiring.png'),
            dict(chunk_id='image', text='', layout_type='Picture', page_number=2,
                 bbox=[100,100,400,350],image_path='extracted_images/wiring.png'),
        ])
        selected, coverage = structural_retrieve('Installation wiring', tree)
        self.assertTrue(any(s['source_chunk_id'] == 'image' for s in selected))
        image = next(s for s in tree['leaves'] if s['source_chunk_id'] == 'image')
        self.assertEqual(image['text'], '')
        self.assertNotIn('reference_number', image)
        self.assertFalse(image.get('visual_candidate_ids'))
        self.assertNotIn('source_links',tree)
        self.assertEqual(tree['nodes'][1]['kind'], 'section')

    def test_native_levels_and_unstructured_text_do_not_invent_legal_tree(self):
        tree = build_document_structure([
            dict(chunk_id=1, text='Maintenance', layout_type='title', heading_level=1),
            dict(chunk_id=2, text='Calibration', layout_type='title', heading_level=2),
            dict(chunk_id=3, text='Read the gauge.'),
        ])
        self.assertEqual(tree['nodes'][2]['parent_id'], tree['nodes'][1]['id'])
        plain = build_document_structure([dict(chunk_id=1, text='An unnumbered plain paragraph.')])
        self.assertEqual(plain['structure_status'], 'unstructured')
        self.assertEqual(len(plain['nodes']), 1)

    def test_visual_review_batches_respect_gateway_image_limit(self):
        tree = build_document_structure([dict(chunk_id=i, text='', layout_type='figure', image_path=f'{i}.png') for i in range(7)])
        batches = review_batches(tree, budget_chars=10000)
        self.assertEqual([len(batch) for batch in batches], [3, 3, 1])

    def test_visual_gate_requires_real_retrieval_provenance_and_keeps_numeric_guard(self):
        evidence = [dict(text='', image_path='wiring.png', colpali_page_rank=1)]
        coverage = dict(retrieval_streams=dict(colpali=dict(state='ready')))
        self.assertTrue(retrieval_gate('Explain this wiring', evidence, coverage)['accepted'])
        self.assertFalse(retrieval_gate('Explain this wiring', evidence, {})['accepted'])
        self.assertFalse(retrieval_gate('Is the voltage 240 V?', evidence, coverage)['accepted'])

    def test_structural_expansion_keeps_visual_rank_for_the_gate(self):
        tree = build_document_structure([dict(chunk_id='image', text='', layout_type='figure', image_path='diagram.png')])
        selected, coverage = structural_retrieve('wiring', tree, seeds=[dict(tree['leaves'][0], colpali_page_rank=1)])
        coverage['retrieval_streams'] = dict(colpali=dict(state='ready'))
        self.assertTrue(retrieval_gate('wiring', selected, coverage)['accepted'])

    def test_reference_words_alone_do_not_run_a_rule_based_resolver(self):
        tree = build_document_structure([dict(chunk_id='section', text='## Wiring\nSee Figure 9.')])
        selected, coverage = structural_retrieve('Wiring', tree)
        self.assertNotIn('document_relation_status',coverage)
        self.assertEqual(coverage['unresolved_document_references'],[])
        self.assertTrue(retrieval_gate('Wiring', selected, coverage)['accepted'])

    def png(self):
        buffer = io.BytesIO()
        Image.new('RGB', (20, 12), (30, 80, 140)).save(buffer, format='PNG')
        return buffer.getvalue()

    def assert_original_image(self, chunk, root, blob):
        self.assertEqual((root / chunk['image_path']).read_bytes(), blob)
        self.assertEqual(chunk['text'], '')
        self.assertEqual(chunk['bbox'], [])
        with override_settings(MEDIA_ROOT=str(root)):
            self.assertEqual(extraction_metadata(chunk, root)['image_path'], chunk['image_path'])

    def test_word_keeps_native_heading_and_original_embedded_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'manual.docx'
            blob = self.png()
            w = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
            with ZipFile(path, 'w') as z:
                z.writestr('word/document.xml', f'<w:document xmlns:w="{w}" xmlns:a="{NS["a"]}" xmlns:r="{NS["r"]}"><w:body><w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Calibration</w:t></w:r></w:p><w:p><w:r><w:drawing><a:blip r:embed="img"/></w:drawing></w:r></w:p></w:body></w:document>')
                z.writestr('word/_rels/document.xml.rels', '<Relationships><Relationship Id="img" Target="media/photo.png"/></Relationships>')
                z.writestr('word/media/photo.png', blob)
            chunks = extract_docx(path, root / 'extracted_images', 'manual')
            self.assertEqual(chunks[0]['heading_level'], 2)
            self.assert_original_image(chunks[1], root, blob)

    def test_slide_image_retains_slide_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = self.png()
            path = root / 'diagram.pptx'
            with ZipFile(path, 'w') as z:
                z.writestr('ppt/presentation.xml', f'<p:presentation xmlns:p="{NS["p"]}" xmlns:r="{NS["r"]}"><p:sldIdLst><p:sldId r:id="s"/></p:sldIdLst></p:presentation>')
                z.writestr('ppt/_rels/presentation.xml.rels', '<Relationships><Relationship Id="s" Target="slides/slide1.xml"/></Relationships>')
                z.writestr('ppt/slides/slide1.xml', f'<p:sld xmlns:p="{NS["p"]}" xmlns:a="{NS["a"]}" xmlns:r="{NS["r"]}"><p:cSld><p:spTree><p:pic><p:blipFill><a:blip r:embed="img"/></p:blipFill></p:pic></p:spTree></p:cSld></p:sld>')
                z.writestr('ppt/slides/_rels/slide1.xml.rels', '<Relationships><Relationship Id="img" Target="../media/photo.png"/></Relationships>')
                z.writestr('ppt/media/photo.png', blob)
            chunks = extract_pptx(path, root / 'extracted_images', 'slide')
            self.assertEqual(chunks[0]['source_location'], 'Slide 1')
            self.assert_original_image(chunks[0], root, blob)

    def test_workbook_image_retains_cell_anchor(self):
        from openpyxl import Workbook
        from openpyxl.drawing.image import Image as SheetImage
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = self.png()
            picture = root / 'original.png'
            picture.write_bytes(blob)
            path = root / 'spec.xlsx'
            book = Workbook()
            book.active.title = 'Wiring'
            book.active.add_image(SheetImage(str(picture)), 'C4')
            book.save(path)
            book.close()
            chunks = extract_spreadsheet(path, root / 'extracted_images', 'sheet')
            self.assertEqual(chunks[0]['source_location'], 'Sheet Wiring, image at C4')
            self.assert_original_image(chunks[0], root, blob)

    def test_external_or_outside_media_path_is_not_attached(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertNotIn('image_path', extraction_metadata(dict(image_path='../outside.png'), root))

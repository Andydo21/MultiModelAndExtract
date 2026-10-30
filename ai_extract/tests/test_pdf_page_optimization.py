from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
import fitz
from PIL import Image
from django.test import SimpleTestCase, override_settings
from ai_extract.services.layout_extractor import SuryaExtractor
from ai_extract.services.contract_pipeline_service import remote_page


class PDFPageOptimizationTests(SimpleTestCase):
    def test_streaming_main_pipeline_delivers_pages_without_retaining_second_document_copy(self):
        from ai_extract.services.contract_pipeline_service import A3ContractPipeline
        path=Path(self.directory.name)/'stream.pdf'
        with self.pdf() as document:
            document.new_page();document.save(path)
        def ready(extractor):extractor.surya_layout_model=Mock()
        def ocr(extractor,**kwargs):
            return [dict(chunk_id=kwargs['page_num'],text='Original source',layout_type='paragraph',
                         page_number=kwargs['page_num'])],kwargs['start_chunk_id']+1
        consumer=Mock()
        with override_settings(MEDIA_ROOT=self.directory.name), \
             patch.object(SuryaExtractor,'_load_surya_models',autospec=True,side_effect=ready), \
             patch.object(SuryaExtractor,'_extract_logged_page',autospec=True,side_effect=ocr):
            result=A3ContractPipeline().extract(path,document_key='stream_test',on_page=consumer,retain_pages=False)
        self.assertEqual(result.pages,[])
        self.assertEqual(result.page_count,2)
        self.assertEqual(consumer.call_count,2)
        self.assertTrue(all(call.args[0].text=='Original source' for call in consumer.call_args_list))

    def test_text_parser_does_not_load_surya(self):
        from ai_extract.services.contract_pipeline_service import A3ContractPipeline
        with TemporaryDirectory() as directory, override_settings(MEDIA_ROOT=directory):
            path=Path(directory)/'source.txt';path.write_text('Original source text.',encoding='utf-8')
            with patch.object(SuryaExtractor,'_load_surya_models') as load:
                result=A3ContractPipeline().extract(path,document_key='text_test')
            load.assert_not_called()
            self.assertEqual(len(result.pages),1)
            self.assertIn('Original source text.',result.pages[0].text)
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings_override = override_settings(A3_PAGE_CACHE_ROOT=Path(self.directory.name), RAG_INFERENCE_URL='http://ocr.test')
        self.settings_override.enable()
        self.addCleanup(self.settings_override.disable)

    def pdf(self):
        document = fitz.open()
        page = document.new_page()
        page.insert_text((50, 70), 'Payment is due within thirty days after acceptance and receipt of the invoice.')
        page.insert_text((50, 100), 'Warranty covers the supplied equipment for twenty four months after acceptance.')
        return document

    def test_streamed_pdf_does_not_repeat_whole_document_chunking(self):
        extractor = SuryaExtractor.__new__(SuryaExtractor)
        extractor.on_page_records = Mock()
        document = SimpleNamespace(category='pdf', id=123, file=SimpleNamespace(path='source.pdf'))
        raw = [dict(text='OCR page content')]
        with patch.object(extractor, '_extract_pdf_pure_surya_vision', return_value=raw), \
             patch('ai_extract.services.document_structure_service.HierarchicalChunker.process_hierarchical_chunks') as chunker:
            self.assertIs(extractor.extract_document(document), raw)
            chunker.assert_not_called()
            del extractor.on_page_records
            extractor.extract_document(document)
            chunker.assert_called_once_with(raw, doc_title='source.pdf')

    def test_failed_ocr_page_does_not_discard_other_pages_or_stop_document(self):
        pdf_path = Path(self.directory.name)/'three.pdf'
        with self.pdf() as document:
            document.new_page()
            document.new_page()
            document.save(pdf_path)
        extractor = SuryaExtractor.__new__(SuryaExtractor)
        extractor.output_img_dir = Path(self.directory.name)
        extractor.strict = True
        extractor.on_page_records, extractor.on_page_error = Mock(), Mock()
        def ocr(**kwargs):
            if kwargs['page_num'] == 2:
                raise RuntimeError('backend unavailable after recovery')
            return [dict(text='source', page_number=kwargs['page_num'])], kwargs['start_chunk_id']+1
        with patch.object(extractor, '_extract_logged_page', side_effect=ocr):
            result = extractor._extract_pdf_pure_surya_vision(pdf_path, 1)
        self.assertEqual([r['page_number'] for r in result], [1, 3])
        self.assertEqual([c.args[0] for c in extractor.on_page_records.call_args_list], [1, 3])
        extractor.on_page_error.assert_called_once()
        self.assertEqual(extractor.on_page_error.call_args.args[0], 2)

    @override_settings(A3_PDF_NATIVE_TEXT=True)
    def test_digital_and_blank_pdf_pages_use_rendered_images_only(self):
        pdf_path = Path(self.directory.name) / 'digital.pdf'
        with self.pdf() as document:
            document[0].set_rotation(90)
            document.new_page()
            document.save(pdf_path)
        extractor = SuryaExtractor.__new__(SuryaExtractor)
        extractor.output_img_dir = Path(self.directory.name)
        extractor.strict = True
        extractor.on_page_records = Mock()
        extractor.on_page_image = Mock()
        images = []

        def extract_image(**kwargs):
            image = kwargs['page_img']
            images.append(image.copy())
            self.assertEqual(image.mode, 'RGB')
            self.assertTrue(kwargs['page_img_path'].is_file())
            self.assertEqual(extractor.on_page_image.call_count, kwargs['page_num'])
            return [dict(text='OCR result', page=kwargs['page_num'],
                         chunk_id=kwargs['start_chunk_id'])], kwargs['start_chunk_id'] + 1

        with patch.object(fitz.Page, 'get_text', side_effect=AssertionError('Native text is forbidden')), \
             patch.object(SuryaExtractor, '_extract_chunks_from_page_image', side_effect=extract_image) as ocr:
            records = extractor._extract_pdf_pure_surya_vision(pdf_path, 123)
        self.assertEqual(ocr.call_count, 2)
        self.assertEqual([r['chunk_id'] for r in records], [1, 2])
        self.assertEqual([r['text'] for r in records], ['OCR result', 'OCR result'])
        self.assertTrue(all(r['text_source'] == 'surya_ocr' and r['layout_source'] == 'surya' for r in records))
        self.assertGreater(images[0].width, images[0].height)
        self.assertLess(images[1].width, images[1].height)
        self.assertEqual(extractor.on_page_records.call_count, 2)
        self.assertEqual(extractor.on_page_image.call_count, 2)

    def test_pdf_ocr_failure_propagates_in_strict_pipeline(self):
        pdf_path = Path(self.directory.name) / 'failure.pdf'
        with self.pdf() as document:
            document.save(pdf_path)
        extractor = SuryaExtractor.__new__(SuryaExtractor)
        extractor.output_img_dir = Path(self.directory.name)
        extractor.strict = True
        original_open = fitz.open
        opened = []
        def open_document(*args, **kwargs):
            document = original_open(*args, **kwargs)
            opened.append(document)
            return document
        with patch.object(fitz, 'open', side_effect=open_document), \
             patch.object(SuryaExtractor, '_extract_chunks_from_page_image', side_effect=RuntimeError('OCR unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'OCR unavailable'):
                extractor._extract_pdf_pure_surya_vision(pdf_path, 123)
        self.assertTrue(opened[0].is_closed)

    def test_completed_page_survives_new_extractor_and_changed_pixels_miss(self):
        response = Mock()
        response.json.return_value = dict(model='surya-ocr-v2', layout=[], ocr_lines=[])
        with patch('ai_extract.services.contract_pipeline_service.requests.post', return_value=response) as post:
            image = Image.new('RGB', (20, 20))
            remote_page(SimpleNamespace(), image)
            hit = remote_page(SimpleNamespace(), image)
            self.assertTrue(hit['cache_hit'])
            self.assertEqual(post.call_count, 1)
            image.putpixel((0, 0), (255, 0, 0))
            remote_page(SimpleNamespace(), image)
            self.assertEqual(post.call_count, 2)

    def test_failed_page_is_not_cached(self):
        import requests
        response = Mock()
        response.json.return_value = dict(model='surya-ocr-v2', layout=[], ocr_lines=[])
        with patch('ai_extract.services.contract_pipeline_service.requests.post', side_effect=[requests.ReadTimeout(), response]) as post:
            with self.assertRaises(RuntimeError):
                remote_page(SimpleNamespace(), Image.new('RGB', (20, 20)))
            remote_page(SimpleNamespace(), Image.new('RGB', (20, 20)))
            self.assertEqual(post.call_count, 2)

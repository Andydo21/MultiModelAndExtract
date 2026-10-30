"""Streaming cancellation must never become cached successful OCR text."""
from types import SimpleNamespace
import base64
import io
from copy import copy
from threading import Event
from unittest.mock import MagicMock, Mock, patch
from django.test import SimpleTestCase
from PIL import Image
from ai_extract.services.surya_inference_service import ManagedSuryaBackend, SuryaStreamManager, repeating_output, surya_page, frame_text_crop


class ContextTestCase(SimpleTestCase):
    def enterContext(self,manager):
        # unittest added this helper after the project's Python 3.10 runtime.
        value=manager.__enter__()
        self.addCleanup(manager.__exit__,None,None,None)
        return value


class SuryaStreamTests(ContextTestCase):
    def setUp(self):
        self.enterContext(patch.dict('os.environ',SURYA_PAGE_GENERATION_TIMEOUT='180',SURYA_REQUEST_GENERATION_TIMEOUT='80'))
        self.item = SimpleNamespace(image=Image.new('RGB', (4, 4)), prompt=None,
            prompt_type='high_accuracy_bbox', metadata={'page': 3}, max_tokens=12288,
            temperature=None, top_p=None, request_logprobs=False, guided_json=None, guided_regex=None)
        self.stream = MagicMock()
        self.stream.__enter__.return_value = self.stream
        self.client = Mock()
        self.client.chat.completions.create.return_value = self.stream
        self.openai = Mock(return_value=self.client)
        self.repeat = self.enterContext(patch('ai_extract.services.surya_inference_service.repeating_output', return_value=False))
        self.enterContext(patch.dict('sys.modules', {
            'openai': SimpleNamespace(OpenAI=self.openai),
            'surya.inference.prompts': SimpleNamespace(PROMPT_MAPPING={'high_accuracy_bbox': 'OCR'}),
            'surya.inference.schema': SimpleNamespace(BatchOutputItem=SimpleNamespace),
            'surya.inference.util': SimpleNamespace(detect_repeat_token=self.repeat, scale_to_fit=lambda image:image),
            'surya.settings': SimpleNamespace(settings=SimpleNamespace(SURYA_INFERENCE_LOGPROBS=False)),
        }))
        backend = Mock()
        backend.backend.start.return_value = SimpleNamespace(base_url='http://local/v1', model_name='surya')
        backend.backend.capacity.return_value = 1
        self.manager = SuryaStreamManager(backend)

    def chunk(self, text='', finish=None, usage=None):
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(
            delta=SimpleNamespace(content=text), finish_reason=finish, logprobs=None)])

    def test_success_keeps_source_text_and_actual_usage(self):
        self.stream.__iter__.return_value = iter([self.chunk('<p>AX-203</p>'),
            self.chunk(finish='stop'), SimpleNamespace(choices=[], usage=SimpleNamespace(completion_tokens=8))])
        result = self.manager.generate([self.item])[0]
        self.assertFalse(result.error)
        self.assertEqual(result.raw, '<p>AX-203</p>')
        self.assertEqual(result.token_count, 8)
        self.openai.assert_called_once_with(api_key='EMPTY', base_url='http://local/v1', max_retries=0)
        self.assertTrue(self.client.chat.completions.create.call_args.kwargs['stream'])

    def test_zero_deadlines_do_not_discard_slow_completed_source(self):
        self.manager.started=-10000
        self.stream.__iter__.return_value=iter([self.chunk('<p>Full source</p>',finish='stop')])
        with patch.dict('os.environ',SURYA_PAGE_GENERATION_TIMEOUT='0',SURYA_REQUEST_GENERATION_TIMEOUT='0'):
            result=self.manager.generate([self.item])[0]
        self.assertFalse(result.error)
        self.assertEqual(result.raw,'<p>Full source</p>')

    def test_repetition_closes_stream_early_without_same_page_retry(self):
        self.repeat.return_value = True
        consumed = []
        def chunks():
            consumed.append(1)
            yield self.chunk('repeat ' * 160)
            consumed.append(2)
            yield self.chunk('must not read another 12000 tokens')
        self.stream.__iter__.return_value = chunks()
        result = self.manager.generate([self.item])[0]
        self.assertTrue(result.error)
        self.assertEqual(result.raw, '')
        self.assertEqual(consumed, [1])
        self.assertEqual(self.manager.calls[0]['status'], 'repetition_cancelled')
        self.stream.__exit__.assert_called_once()
        self.client.chat.completions.create.assert_called_once()

    def test_truncated_completion_is_not_successful_page_text(self):
        self.stream.__iter__.return_value = iter([self.chunk('unfinished', finish='length')])
        result = self.manager.generate([self.item])[0]
        self.assertTrue(result.error)
        self.assertEqual(result.raw, '')

    def test_exhausted_page_budget_makes_no_request(self):
        self.manager.started = -float('inf')
        result = self.manager.generate([self.item])[0]
        self.assertTrue(result.error)
        self.client.chat.completions.create.assert_not_called()

    def test_read_error_is_not_implicitly_retried(self):
        self.client.chat.completions.create.side_effect = TimeoutError('read failed')
        result = self.manager.generate([self.item])[0]
        self.assertTrue(result.error)
        self.client.chat.completions.create.assert_called_once()

    def test_block_budget_allows_markup_and_code_without_repeating_crop(self):
        self.item.prompt_type = 'block'
        self.item.prompt = 'Read code'
        self.item.max_tokens = 820
        self.stream.__iter__.return_value = iter([self.chunk('code', finish='stop')])
        self.manager.generate([self.item])
        self.assertEqual(self.client.chat.completions.create.call_args.kwargs['max_tokens'], 1640)
        self.client.chat.completions.create.assert_called_once()

    def test_list_crop_has_token_headroom_and_larger_bounded_retry(self):
        self.item.prompt_type='block';self.item.prompt='OCR';self.item.max_tokens=128
        second=MagicMock();second.__enter__.return_value=second
        self.stream.__iter__.return_value=iter([self.chunk('unfinished',finish='length')])
        second.__iter__.return_value=iter([self.chunk('<ul><li>Shape search</li><li>Color search</li></ul>',finish='stop')])
        self.client.chat.completions.create.side_effect=[self.stream,second]
        result=self.manager.generate([self.item])[0]
        self.assertFalse(result.error)
        self.assertIn('Color search',result.raw)
        self.assertEqual([call.kwargs['max_tokens'] for call in self.client.chat.completions.create.call_args_list],[1024,2048])

    def test_timeout_retries_only_failed_block_preserving_successful_siblings(self):
        APITimeoutError = type('APITimeoutError', (Exception,), {})
        first, second = copy(self.item), copy(self.item)
        first.prompt_type = second.prompt_type = 'block'
        first.prompt, second.prompt = 'first', 'second'
        first.metadata, second.metadata = {'block_idx': 0}, {'block_idx': 1}
        self.stream.__iter__.side_effect = [iter([self.chunk('first', finish='stop')]),
                                            iter([self.chunk('second', finish='stop')])]
        self.client.chat.completions.create.side_effect = [self.stream, APITimeoutError(), self.stream]
        results = self.manager.generate([first, second])
        self.assertEqual([result.raw for result in results], ['first', 'second'])
        self.assertTrue(all(not result.error for result in results))
        requests = self.client.chat.completions.create.call_args_list
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[1].kwargs, requests[2].kwargs)
        self.assertEqual([call['block_index'] for call in self.manager.calls], [0, 1, 1])
        self.assertEqual(self.manager.calls[-1]['retry_reason'], 'APITimeoutError')
        self.assertEqual(self.manager.calls[-1]['retries'], 1)

    def test_second_timeout_discards_partial_text_and_stops_retrying(self):
        APITimeoutError = type('APITimeoutError', (Exception,), {})
        self.item.prompt_type, self.item.prompt = 'block', 'OCR block'
        def interrupted():
            yield self.chunk('partial text')
            raise APITimeoutError()
        self.stream.__iter__.side_effect = [interrupted(), interrupted()]
        result = self.manager.generate([self.item])[0]
        self.assertTrue(result.error)
        self.assertEqual(result.raw, '')
        self.assertEqual(self.client.chat.completions.create.call_count, 2)
        self.assertEqual(self.stream.__exit__.call_count, 2)

    def test_transport_retry_does_not_reset_page_deadline(self):
        APITimeoutError = type('APITimeoutError', (Exception,), {})
        self.item.prompt_type, self.item.prompt = 'block', 'OCR block'
        def timeout(**options):
            self.manager.started = -float('inf')
            raise APITimeoutError()
        self.client.chat.completions.create.side_effect = timeout
        result = self.manager.generate([self.item])[0]
        self.assertTrue(result.error)
        self.client.chat.completions.create.assert_called_once()

    def test_connection_error_retries_crop_once(self):
        APIConnectionError = type('APIConnectionError', (Exception,), {})
        self.item.prompt_type, self.item.prompt = 'block', 'OCR block'
        self.stream.__iter__.return_value = iter([self.chunk('source text', finish='stop')])
        self.client.chat.completions.create.side_effect = [APIConnectionError(), self.stream]
        result = self.manager.generate([self.item])[0]
        self.assertFalse(result.error)
        self.assertEqual(result.raw, 'source text')
        self.assertEqual(self.manager.calls[-1]['retry_reason'], 'APIConnectionError')

    def test_whitespace_loop_retries_only_failed_crop_with_source_pixels_intact(self):
        self.item.prompt_type = 'block'
        self.item.prompt = 'OCR block'
        self.stream.__iter__.return_value = iter([self.chunk('<pre>v0'+' '*300)])
        second = MagicMock()
        second.__enter__.return_value = second
        second.__iter__.return_value = iter([self.chunk('<p>v0 := ld state[748]</p>', finish='stop')])
        self.client.chat.completions.create.side_effect = [self.stream, second]
        result = self.manager.generate([self.item])[0]
        self.assertFalse(result.error)
        self.assertEqual(len(self.manager.calls), 2)
        self.assertEqual(self.manager.calls[0]['status'], 'whitespace_loop_cancelled')
        options = self.client.chat.completions.create.call_args.kwargs
        encoded = options['messages'][0]['content'][0]['image_url']['url'].split(',')[1]
        padded = Image.open(io.BytesIO(base64.b64decode(encoded)))
        self.assertGreater(padded.width, self.item.image.width)
        self.assertEqual(padded.getpixel((0,0)), self.item.image.getpixel((0,0)))
        self.assertEqual(options['temperature'], 0)
        self.assertEqual(result.metadata, self.item.metadata)

    def test_failed_crop_rescue_is_bounded_to_one_retry(self):
        self.item.prompt_type = 'block'
        self.item.prompt = 'OCR block'
        self.stream.__iter__.side_effect = [iter([self.chunk(' '*300)]), iter([self.chunk(' '*300)])]
        result = self.manager.generate([self.item])[0]
        self.assertTrue(result.error)
        self.assertEqual(self.client.chat.completions.create.call_count, 2)

    def test_tiny_crop_is_framed_without_discarding_source_pixels(self):
        crop=Image.new('RGB',(40,8),'blue')
        framed=frame_text_crop(crop)
        self.assertGreaterEqual(framed.width,256)
        self.assertGreaterEqual(framed.height,128)
        self.assertEqual(framed.getpixel((framed.width//2,framed.height//2)),(0,0,255))
        self.assertEqual(framed.getpixel((0,0)),(255,255,255))
        self.assertEqual(crop.size,(40,8))
        normal=Image.new('RGB',(400,200))
        self.assertIs(frame_text_crop(normal),normal)

    def test_visual_repetition_is_not_regenerated_into_successful_evidence(self):
        self.manager.visual_phase=True
        self.item.prompt_type='block';self.item.prompt='OCR';self.item.max_tokens=128
        self.repeat.return_value=True
        self.stream.__iter__.return_value=iter([self.chunk('bad repetition '*80)])
        result=self.manager.generate([self.item])[0]
        self.assertTrue(result.error)
        self.assertEqual(result.raw,'')
        self.assertEqual(self.client.chat.completions.create.call_count,1)
        self.assertEqual(self.client.chat.completions.create.call_args.kwargs['temperature'],0)

    def test_visual_retry_cancels_repetition_at_smaller_budget(self):
        self.manager.visual_phase=True
        self.item.prompt_type='block';self.item.prompt='OCR';self.item.max_tokens=128
        self.repeat.return_value=True
        self.stream.__iter__.return_value=iter([self.chunk('bad repetition '*30)])
        result=self.manager.generate([self.item],retry=True,retry_reason='repetition_cancelled',rescue=True)[0]
        self.assertTrue(result.error)
        self.assertLess(self.manager.calls[0]['output_chars'],1000)
        self.client.chat.completions.create.assert_called_once()

    def test_configured_sampling_cannot_make_ocr_creative(self):
        self.item.temperature=.8;self.item.top_p=.95
        self.stream.__iter__.return_value=iter([self.chunk('source',finish='stop')])
        self.manager.generate([self.item])
        options=self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(options['temperature'],0)
        self.assertEqual(options['top_p'],.1)


    def test_fast_mode_reads_layout_then_blocks_without_full_page_attempt(self):
        layout = [SimpleNamespace(error=False)]
        layout_predictor = Mock(return_value=layout)
        factory = Mock(return_value=layout_predictor)
        predictor = Mock(manager=self.manager, return_value=[SimpleNamespace(blocks=[])])
        with patch.dict('sys.modules', {'surya.layout': SimpleNamespace(LayoutPredictor=factory)}), \
             patch.dict('os.environ', SURYA_OCR_MODE='blocks'):
            result = surya_page(self.item.image, predictor)
        factory.assert_called_once_with(self.manager)
        predictor.assert_called_once_with([self.item.image], layout_results=layout, full_page=False)
        self.assertEqual(result['metrics']['ocr_mode'], 'blocks')

    def test_body_ocr_uses_sharp_pixels_and_restores_preview_coordinates(self):
        class Block:
            error=False;reading_order=0;skipped=False;label='ListGroup';html='<ul><li>One</li><li>Two</li></ul>'
            polygon=[[20,40],[160,40],[160,120],[20,120]]
            @property
            def bbox(self):
                return [self.polygon[0][0],self.polygon[0][1],self.polygon[2][0],self.polygon[2][1]]
        box=SimpleNamespace(polygon=[[10,20],[80,20],[80,60],[10,60]])
        layout=[SimpleNamespace(error=False,bboxes=[box],image_bbox=[0,0,100,100])]
        predictor=Mock(manager=self.manager,return_value=[SimpleNamespace(blocks=[Block()])])
        sharp=Image.new('RGB',(200,200))
        with patch.dict('sys.modules',{'surya.layout':SimpleNamespace(LayoutPredictor=Mock(return_value=Mock(return_value=layout)))}), \
             patch.dict('os.environ',SURYA_OCR_MODE='blocks'):
            page=surya_page(Image.new('RGB',(100,100)),predictor,sharp)
        self.assertIs(predictor.call_args.args[0][0],sharp)
        self.assertEqual(predictor.call_args.kwargs['layout_results'][0].bboxes[0].polygon,[[20,40],[160,40],[160,120],[20,120]])
        self.assertEqual(box.polygon,[[10,20],[80,20],[80,60],[10,60]])
        self.assertEqual(page['ocr_lines'][0]['bbox'],[10,20,80,60])
        self.assertEqual(page['ocr_lines'][0]['text'],'One\nTwo')

    def test_failed_layout_cannot_look_like_successful_blank_page(self):
        predictor = Mock(manager=self.manager)
        factory = Mock(return_value=Mock(return_value=[SimpleNamespace(error=True)]))
        with patch.dict('sys.modules', {'surya.layout': SimpleNamespace(LayoutPredictor=factory)}), \
             patch.dict('os.environ', SURYA_OCR_MODE='blocks'):
            with self.assertRaises(RuntimeError):
                surya_page(self.item.image, predictor)
        predictor.assert_not_called()

    def test_failed_page_reports_final_block_failure_not_recovered_attempt(self):
        self.manager.calls = [
            dict(prompt='block', block_index=0, status='APITimeoutError', retries=0),
            dict(prompt='block', block_index=0, status='completed', retries=1),
            dict(prompt='block', block_index=2, status='APITimeoutError', retries=1),
        ]
        layout = [SimpleNamespace(error=False)]
        factory = Mock(return_value=Mock(return_value=layout))
        predictor = Mock(manager=self.manager, return_value=[SimpleNamespace(
            blocks=[SimpleNamespace(error=True, reading_order=2,bbox=[0,0,4,4],label='Text',skipped=False)])])
        with patch.object(self.manager, 'begin_page'), \
             patch.dict('sys.modules', {'surya.layout': SimpleNamespace(LayoutPredictor=factory)}), \
             patch.dict('os.environ', SURYA_OCR_MODE='blocks'):
            result=surya_page(self.item.image, predictor)
        self.assertEqual(result['metrics']['failed_regions'],1)
        self.assertEqual(result['ocr_lines'],[])
        self.assertFalse(result['cacheable'])

    def test_two_crops_keep_source_order_when_later_crop_finishes_first(self):
        later_done = Event()
        first, second = copy(self.item), copy(self.item)
        first.prompt, second.prompt = 'first', 'second'
        first.metadata, second.metadata = {'block_idx': 0}, {'block_idx': 1}
        self.manager.manager.backend.capacity.return_value = 2
        def completion(**options):
            stream = MagicMock()
            stream.__enter__.return_value = stream
            def chunks():
                if options['messages'][0]['content'][1]['text'] == 'first':
                    if not later_done.wait(2):
                        raise AssertionError('Crops did not overlap')
                    yield self.chunk('first', finish='stop')
                else:
                    later_done.set()
                    yield self.chunk('second', finish='stop')
            stream.__iter__.return_value = chunks()
            return stream
        self.client.chat.completions.create.side_effect = completion
        with patch.dict('os.environ', SURYA_CROP_CONCURRENCY='2'):
            results = self.manager.generate([first, second])
        self.assertEqual([result.raw for result in results], ['first', 'second'])
        self.assertEqual([result.metadata['block_idx'] for result in results], [0, 1])
        self.assertTrue(all(not result.error for result in results))


class RepetitionGuardTests(SimpleTestCase):
    def test_code_indentation_and_separator_are_not_decoder_loops(self):
        self.assertFalse(repeating_output('mov ecx, ebx(4)'+' '*300+'// load primes'+'-'*300))

    def test_repeated_content_is_a_decoder_loop(self):
        self.assertTrue(repeating_output('<div bbox="1,2,3,4">same hallucinated content</div>'*8))


class ManagedBackendTests(ContextTestCase):
    def setUp(self):
        self.enterContext(patch.dict('os.environ', SURYA_INFERENCE_URL='http://127.0.0.1:8011/v1',
            LLAMA_CPP_BINARY='binary', SURYA_GGUF_LOCAL_MODEL_PATH='model', SURYA_GGUF_LOCAL_MMPROJ_PATH='projection'))
        self.backend = Mock()
        self.managed = ManagedSuryaBackend(self.backend)

    def test_healthy_backend_does_not_spawn_another_process(self):
        with patch.object(self.managed, 'ready', return_value=True), \
             patch('ai_extract.services.surya_inference_service.subprocess.Popen') as spawn:
            self.assertIs(self.managed.start(), self.backend.start.return_value)
        spawn.assert_not_called()

    def test_dead_owned_backend_is_replaced_using_local_files(self):
        old, new = Mock(pid=1), Mock(pid=2)
        old.poll.return_value = 1
        new.poll.return_value = None
        self.managed.process = old
        with patch.object(self.managed, 'ready', side_effect=[False, False, True]), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.open', MagicMock()), \
             patch('socket.socket') as socket, \
             patch('ai_extract.services.surya_inference_service.subprocess.Popen', return_value=new) as spawn:
            socket.return_value.__enter__.return_value.connect_ex.return_value = 1
            self.managed.start()
        old.terminate.assert_not_called()
        self.assertIs(self.managed.process, new)
        self.assertIn('--n-gpu-layers', spawn.call_args.args[0])
        self.assertEqual(self.backend.stop.call_count, 2)

    def test_release_never_terminates_an_external_process(self):
        self.managed.suspend()
        self.backend.stop.assert_not_called()

    def test_recovery_fails_bounded_when_binary_exits_at_startup(self):
        process = Mock(pid=2)
        process.poll.return_value = 9
        with patch.object(self.managed, 'ready', return_value=False), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.open', MagicMock()), \
             patch('socket.socket') as socket, \
             patch('ai_extract.services.surya_inference_service.subprocess.Popen', return_value=process):
            socket.return_value.__enter__.return_value.connect_ex.return_value = 1
            with self.assertRaises(RuntimeError):
                self.managed.start()
        self.assertIsNone(self.managed.process)

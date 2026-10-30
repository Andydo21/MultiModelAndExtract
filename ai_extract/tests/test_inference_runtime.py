"""Model lifecycle checks without downloading or loading any checkpoint."""
from types import SimpleNamespace
from unittest.mock import Mock, patch
from django.test import SimpleTestCase
from ai_extract.services.inference_runtime_service import TextModelRuntime, gpu_stage, model_inputs, run_batches


class InferenceRuntimeTests(SimpleTestCase):
    def test_idle_cleanup_waits_for_work_and_restarts_after_reuse(self):
        import threading
        from ai_extract.services import inference_runtime_service as service
        fired = threading.Event()
        with patch.object(service, '_idle_cleanup', None), \
             patch.object(service, '_idle_timer', None), \
             patch.object(service, '_idle_generation', 0), \
             patch.object(service, '_idle_seconds', 120.0):
            service.configure_idle_model_cleanup(fired.set, seconds=0.1)
            with service.gpu_stage():
                with service.gpu_stage():
                    pass
                self.assertFalse(fired.wait(0.2))
            with service.gpu_stage():
                self.assertFalse(fired.is_set())
            self.assertTrue(fired.wait(2))

    def test_gateway_console_reaches_existing_docker_container(self):
        import io
        from ai_extract.services.inference_runtime_service import DockerLogStream
        fallback=io.StringIO();process=Mock();process.poll.return_value=None
        process.stdin=io.StringIO()
        with patch('subprocess.Popen',return_value=process) as launch:
            stream=DockerLogStream('riskdl_web',fallback)
            stream.write('entity model completed\n')
            stream.flush()
        self.assertEqual(process.stdin.getvalue(),'entity model completed\n')
        self.assertEqual(launch.call_args.args[0],['docker','exec','-i','riskdl_web','sh','-c','cat > /proc/1/fd/1'])
        self.assertEqual(fallback.getvalue(),'')
        stream.close()

    def test_docker_restart_does_not_crash_gateway_logging_and_reconnects(self):
        import io
        from ai_extract.services.inference_runtime_service import DockerLogStream
        old=Mock();old.poll.return_value=None;old.stdin.write.side_effect=BrokenPipeError('restarted')
        fresh=Mock();fresh.poll.return_value=None;fresh.stdin=io.StringIO();fallback=io.StringIO()
        with patch('subprocess.Popen',side_effect=[old,fresh]):
            stream=DockerLogStream('riskdl_web',fallback)
            stream.write('during restart\n');stream.write('reconnected\n')
        self.assertEqual(fallback.getvalue(),'during restart\n')
        self.assertEqual(fresh.stdin.getvalue(),'reconnected\n')
        stream.close()

    def test_paging_error_uses_streamed_checkpoint_without_cpu_inference(self):
        runtime=TextModelRuntime()
        torch,transformers,network=self.libraries()
        transformers.AutoModel.from_pretrained.side_effect=OSError('paging file too small (1455)')
        with patch.dict('sys.modules',torch=torch,transformers=transformers), \
             patch.dict('os.environ',RAG_TEXT_DEVICE='cuda',RAG_STREAM_CHECKPOINTS='0'), \
             patch('ai_extract.services.inference_runtime_service.stream_checkpoint',return_value=network) as load:
            runtime.get('BAAI/bge-m3')
        self.assertEqual(load.call_args.args[3],'cuda')
        network.to.assert_called_with('cuda')

    def test_streamed_mode_drops_inactive_model_without_copying_weights_to_cpu(self):
        runtime=TextModelRuntime();network=Mock(device='cuda:0')
        runtime.models['old']=(Mock(),network)
        with patch.dict('os.environ',RAG_STREAM_CHECKPOINTS='1'), \
             patch.dict('sys.modules',torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=Mock()))):
            runtime.park()
        self.assertEqual(runtime.models,{})
        network.to.assert_not_called()

    def test_busy_gpu_rejects_without_adding_another_model_job(self):
        lock = Mock()
        lock.acquire.return_value = False
        with patch('ai_extract.services.inference_runtime_service.GPU_LOCK', lock):
            with self.assertRaises(TimeoutError):
                with gpu_stage():
                    self.fail('Busy GPU must not start another inference')
        lock.release.assert_not_called()

    def test_parked_weights_return_to_cuda_without_checkpoint_reload(self):
        runtime = TextModelRuntime()
        torch, transformers, network = self.libraries()
        network.device = 'cuda:0'
        network.to.side_effect = lambda device: setattr(network, 'device', device) or network
        with patch.dict('sys.modules', torch=torch, transformers=transformers), patch.dict('os.environ', RAG_TEXT_DEVICE='cuda'):
            pair = runtime.get('BAAI/bge-m3')
            runtime.park()
            self.assertEqual(network.device, 'cpu')
            self.assertIs(runtime.get('BAAI/bge-m3'), pair)
        self.assertEqual(network.device, 'cuda')
        transformers.AutoModel.from_pretrained.assert_called_once()
        torch.cuda.empty_cache.assert_called_once()

    def test_text_model_switch_frees_inactive_vram_and_restores_cached_weights(self):
        runtime = TextModelRuntime()
        torch, transformers, _ = self.libraries()
        models = []
        for _ in range(2):
            network = Mock(device='cuda:0')
            network.eval.return_value = network
            network.to.side_effect = lambda device, model=network: setattr(model, 'device', device) or model
            models.append(network)
        transformers.AutoModel.from_pretrained.side_effect = models
        with patch.dict('sys.modules', torch=torch, transformers=transformers), patch.dict('os.environ', RAG_TEXT_DEVICE='cuda'):
            runtime.get('BAAI/bge-m3')
            runtime.get('BAAI/bge-reranker-v2-m3', True)
            self.assertEqual(models[0].device, 'cpu')
            self.assertEqual(models[1].device, 'cuda')
            runtime.get('BAAI/bge-m3')
        self.assertEqual(models[0].device, 'cuda')
        self.assertEqual(models[1].device, 'cpu')
        self.assertEqual(transformers.AutoModel.from_pretrained.call_count, 2)
        for call in transformers.AutoModel.from_pretrained.call_args_list:
            self.assertEqual(call.kwargs['device_map'], {'': 'cuda'})

    def libraries(self, free=4 * 1024 ** 3):
        torch = SimpleNamespace(float16='fp16', float32='fp32',
            cuda=SimpleNamespace(is_available=lambda: True,
                                 mem_get_info=Mock(return_value=(free, free)), empty_cache=Mock()))
        network = Mock()
        network.eval.return_value = network
        network.to.return_value = network
        network.device = SimpleNamespace(type='cuda')
        factory = Mock()
        factory.from_pretrained.return_value = network
        transformers = SimpleNamespace(AutoTokenizer=Mock(), AutoModel=factory,
                                        AutoModelForSequenceClassification=factory)
        return torch, transformers, network

    def test_both_checkpoints_stay_cached_and_load_only_offline(self):
        runtime = TextModelRuntime()
        torch, transformers, network = self.libraries()
        with patch.dict('sys.modules', torch=torch, transformers=transformers), patch.dict('os.environ', RAG_TEXT_DEVICE='auto'):
            first = runtime.get('BAAI/bge-m3')
            runtime.get('BAAI/bge-reranker-v2-m3', True)
            self.assertIs(runtime.get('BAAI/bge-m3'), first)
        self.assertEqual(transformers.AutoModel.from_pretrained.call_count, 2)
        self.assertEqual(len(runtime.models), 2)
        network.to.assert_called_with('cuda')
        for call in transformers.AutoModel.from_pretrained.call_args_list:
            self.assertTrue(call.kwargs['local_files_only'])
            self.assertTrue(call.kwargs['use_safetensors'])
            self.assertEqual(call.kwargs['torch_dtype'], 'fp16')
        self.assertEqual(runtime.loads['BAAI/bge-m3']['count'], 1)

    def test_low_reported_vram_does_not_switch_inference_to_cpu(self):
        runtime = TextModelRuntime()
        torch, transformers, network = self.libraries(free=100)
        with patch.dict('sys.modules', torch=torch, transformers=transformers), patch.dict('os.environ', RAG_TEXT_DEVICE='cuda'):
            runtime.get('BAAI/bge-m3')
            runtime.get('BAAI/bge-m3')
        self.assertEqual(len(runtime.models), 1)
        self.assertEqual(transformers.AutoModel.from_pretrained.call_count, 1)
        self.assertTrue(all(call.args[0] == 'cuda' for call in network.to.call_args_list))
        torch.cuda.mem_get_info.assert_not_called()
        torch.cuda.empty_cache.assert_not_called()

    def test_cpu_is_used_only_when_explicitly_selected(self):
        runtime = TextModelRuntime()
        torch, transformers, network = self.libraries(free=100)
        with patch.dict('sys.modules', torch=torch, transformers=transformers), patch.dict('os.environ', RAG_TEXT_DEVICE='cpu'):
            runtime.get('BAAI/bge-m3')
        network.to.assert_called_with('cpu')
        self.assertEqual(transformers.AutoModel.from_pretrained.call_args.kwargs['torch_dtype'], 'fp32')

    def test_explicit_cuda_never_silently_uses_cpu(self):
        runtime = TextModelRuntime()
        torch, transformers, network = self.libraries()
        torch.cuda.is_available = lambda: False
        with patch.dict('sys.modules', torch=torch, transformers=transformers), patch.dict('os.environ', RAG_TEXT_DEVICE='cuda'):
            with self.assertRaises(RuntimeError):
                runtime.get('BAAI/bge-m3')
        transformers.AutoModel.from_pretrained.assert_not_called()

    def test_oom_retries_smaller_gpu_batches_without_losing_order(self):
        class OutOfMemoryError(RuntimeError):
            pass
        torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OutOfMemoryError, empty_cache=Mock()))
        attempts = []
        def operation(chunk):
            attempts.append(len(chunk))
            if len(chunk) > 2:
                raise OutOfMemoryError()
            return [value * 2 for value in chunk]
        self.assertEqual(run_batches(list(range(9)), 8, operation, torch), [value * 2 for value in range(9)])
        self.assertEqual(attempts[:3], [8, 4, 2])
        self.assertEqual(torch.cuda.empty_cache.call_count, 2)

    def test_oom_on_single_item_fails_without_cpu_fallback(self):
        class OutOfMemoryError(RuntimeError):
            pass
        torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OutOfMemoryError, empty_cache=Mock()))
        with self.assertRaises(OutOfMemoryError):
            run_batches([1], 1, Mock(side_effect=OutOfMemoryError()), torch)

    def test_token_tensors_follow_network_device(self):
        tensors = {'input_ids': Mock(), 'attention_mask': Mock()}
        network = SimpleNamespace(device='cuda:0')
        result = model_inputs(tensors, network)
        for name, tensor in tensors.items():
            tensor.to.assert_called_once_with('cuda:0')
            self.assertIs(result[name], tensor.to.return_value)

"""Offline text-model cache shared by embedding and reranking HTTP routes."""
import logging
from threading import Event, Thread, Timer, Lock
import os
import time
import json
import struct
from pathlib import Path
from contextlib import contextmanager
from threading import RLock

GPU_LOCK = RLock()
_idle_lock = Lock()
_idle_timer = None
_idle_generation = 0
_idle_cleanup = None
_idle_seconds = 120.0


def configure_idle_model_cleanup(callback, seconds=120):
    """Release gateway-owned models after a period without GPU work."""
    global _idle_cleanup, _idle_seconds
    if seconds <= 0:
        raise ValueError('Idle cleanup timeout must be positive')
    with _idle_lock:
        _idle_cleanup, _idle_seconds = callback, float(seconds)


def _invalidate_idle_timer():
    global _idle_generation, _idle_timer
    with _idle_lock:
        _idle_generation += 1
        if _idle_timer is not None:
            _idle_timer.cancel()
            _idle_timer = None


def _schedule_idle_cleanup():
    global _idle_timer, _idle_generation
    with _idle_lock:
        if _idle_cleanup is None:
            return
        if _idle_timer is not None:
            _idle_timer.cancel()
        _idle_generation += 1
        generation = _idle_generation
        def cleanup():
            with GPU_LOCK:
                with _idle_lock:
                    if generation != _idle_generation:
                        return
                    callback = _idle_cleanup
                try:
                    callback()
                except Exception:
                    logging.getLogger(__name__).exception('Idle model cleanup failed')
        timer = Timer(_idle_seconds, cleanup)
        timer.daemon = True
        _idle_timer = timer
        timer.start()


class DockerLogStream:
    """Send the host GPU gateway's console to an existing Docker container."""
    def __init__(self,container,fallback):
        self.container,self.fallback=container,fallback
        self.process=None;self.lock=RLock()

    def write(self,text):
        if not text:return 0
        import subprocess
        with self.lock:
            try:
                if self.process is None or self.process.poll() is not None:
                    self.process=subprocess.Popen(['docker','exec','-i',self.container,'sh','-c','cat > /proc/1/fd/1'],
                        stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                        text=True,encoding='utf-8',bufsize=1,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                self.process.stdin.write(text);self.process.stdin.flush()
            except (OSError,ValueError):
                self.close()
                self.fallback.write(text);self.fallback.flush()
        return len(text)

    def flush(self):
        with self.lock:
            try:
                if self.process is not None:self.process.stdin.flush()
            except (OSError,ValueError):self.close()
            self.fallback.flush()

    def close(self):
        with self.lock:
            process,self.process=self.process,None
            if process is not None:
                try:process.stdin.close()
                except (OSError,ValueError):pass

    def isatty(self):return False
    def __getattr__(self,name):return getattr(self.fallback,name)


def configure_gateway_logging():
    """Keep inference on the host GPU while Docker owns its operational console."""
    import sys,atexit
    container=os.getenv('A3_DOCKER_LOG_CONTAINER','')
    if not container or isinstance(sys.stderr,DockerLogStream):return
    stream=DockerLogStream(container,sys.stderr)
    sys.stdout=sys.stderr=stream
    for name in ('','uvicorn','uvicorn.error','uvicorn.access','ai_extract.pipeline'):
        for handler in logging.getLogger(name).handlers:
            if isinstance(handler,logging.StreamHandler) and not isinstance(handler,logging.FileHandler):
                handler.setStream(stream)
    atexit.register(stream.close)


def stream_checkpoint(factory, location, options, device, torch):
    """Load existing tensors in small reads when Windows cannot map the checkpoint.

    No conversion file, download or CPU inference: FP16 weights go straight
    into the allocated CUDA model, without mapping the entire FP32 file.
    """
    from transformers import AutoConfig
    from transformers.utils.hub import cached_file
    checkpoint=Path(location)/'model.safetensors'
    if not checkpoint.is_file():
        checkpoint=Path(cached_file(location,'model.safetensors',**options))
    config=AutoConfig.from_pretrained(location,**options)
    # Meta construction avoids even a temporary FP32 CPU embedding allocation.
    with torch.device('meta'):
        network=factory.from_config(config)
    network.half() if device=='cuda' else network.float()
    network.to_empty(device=device)
    targets=network.state_dict()
    dtypes={'F32':torch.float32,'F16':torch.float16,'BF16':torch.bfloat16,
            'I64':torch.int64,'I32':torch.int32,'I16':torch.int16,'I8':torch.int8,'U8':torch.uint8,'BOOL':torch.bool}
    with checkpoint.open('rb') as handle,torch.no_grad():
        header_size=struct.unpack('<Q',handle.read(8))[0]
        if not 0<header_size<=64*1024*1024:raise ValueError('Invalid safetensors header size')
        header=json.loads(handle.read(header_size));data_start=8+header_size
        missing=set(targets)-set(header)
        if missing:raise ValueError('Checkpoint missing weights: '+', '.join(sorted(missing)[:5]))
        loaded_bytes=0
        for name,target in targets.items():
            info=header[name];start,end=info['data_offsets'];shape=info['shape']
            if list(target.shape)!=shape or info['dtype'] not in dtypes:
                raise ValueError('Checkpoint tensor mismatch: '+name)
            dtype=dtypes[info['dtype']]
            item_size=torch.empty((),dtype=dtype).element_size()
            if end-start!=target.numel()*item_size or start<0 or data_start+end>checkpoint.stat().st_size:
                raise ValueError('Invalid checkpoint offsets: '+name)
            handle.seek(data_start+start)
            flat=target.view(-1);offset=0
            while offset<flat.numel():
                count=min(flat.numel()-offset,8*1024*1024//item_size)
                payload=bytearray(handle.read(count*item_size))
                if len(payload)!=count*item_size:raise ValueError('Truncated checkpoint tensor: '+name)
                source=torch.frombuffer(payload,dtype=dtype)
                flat[offset:offset+count].copy_(source)
                offset+=count;loaded_bytes+=len(payload)
                del source,payload
        for name,value in network.named_buffers():
            if name in targets:continue
            if name.endswith('position_ids'):
                value.copy_(torch.arange(value.numel(),device=device,dtype=value.dtype).reshape(value.shape))
            elif name.endswith('token_type_ids'):value.zero_()
            else:raise ValueError('Unsupported non-checkpoint buffer: '+name)
    print(json.dumps(dict(event='text_model_load',backend='streamed_checkpoint',device=device,
                          checkpoint_bytes=loaded_bytes,tensors=len(targets))),flush=True)
    return network


@contextmanager
def gpu_stage(timeout=5):
    """Reject overlapping GPU work instead of accumulating timed-out requests."""
    if not GPU_LOCK.acquire(timeout=timeout):
        raise TimeoutError('GPU stage busy; retry after the current page')
    _invalidate_idle_timer()
    try:
        yield
    finally:
        try:
            _schedule_idle_cleanup()
        finally:
            GPU_LOCK.release()


class TextModelRuntime:
    """Cache checkpoints in RAM; only the active text model occupies CUDA."""

    def __init__(self):
        self.models = {}
        self.loads = {}

    def _device(self, torch):
        requested = os.getenv('RAG_TEXT_DEVICE', 'cuda').lower()
        if requested not in {'auto', 'cpu', 'cuda'}:
            raise ValueError('RAG_TEXT_DEVICE must be auto, cpu or cuda')
        if requested == 'cpu':
            return 'cpu'
        if not torch.cuda.is_available():
            if requested == 'cuda':
                raise RuntimeError('CUDA requested but unavailable')
            return 'cpu'
        return 'cuda'

    def get(self, name, classification=False):
        self.park(except_name=name)
        if name in self.models:
            import torch
            network = self.models[name][1]
            device = self._device(torch)
            if str(network.device).split(':')[0] != device:
                network.to(device)
            return self.models[name]
        import torch
        from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
        device = self._device(torch)
        factory = AutoModelForSequenceClassification if classification else AutoModel
        location = (os.getenv('BGE_MODEL_PATH', '') if not classification else '') or name
        options = dict(local_files_only=True, token=False)
        cache = os.getenv('BGE_LOCAL_CACHE', '')
        if cache:
            options['cache_dir'] = cache
        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(location, **options)
        paging_failure=os.getenv('RAG_STREAM_CHECKPOINTS','0')=='1'
        if not paging_failure:
            try:
                network = factory.from_pretrained(location, use_safetensors=True,
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                    device_map={'': device}, **options)
            except OSError as error:
                if '1455' not in str(error) and 'paging file' not in str(error).lower():raise
                paging_failure=True
        if paging_failure:
            import gc
            gc.collect();torch.cuda.empty_cache()
            print(json.dumps(dict(event='text_model_load',model=name,status='retry_streamed_checkpoint')),flush=True)
            network=stream_checkpoint(factory,location,options,device,torch)
        network = network.eval().to(device)
        self.models[name] = (tokenizer, network)
        previous = self.loads.get(name, {})
        self.loads[name] = dict(count=previous.get('count', 0) + 1,
            load_ms=round((time.perf_counter() - started) * 1000), device=device)
        return self.models[name]

    def park(self, except_name=None):
        """Keep weights in host memory during OCR/Docling; inference stays on CUDA."""
        if os.getenv('RAG_STREAM_CHECKPOINTS','0')=='1':
            # Avoid copying inactive CUDA weights back into the exhausted Windows
            # paging pool. The next stage reloads existing weights in small reads.
            removed=False
            for name in list(self.models):
                if name!=except_name:
                    del self.models[name];removed=True
            if removed:
                import gc,torch
                gc.collect();torch.cuda.empty_cache()
            return
        moved = False
        for name, (_, network) in self.models.items():
            if name == except_name:
                continue
            if str(network.device).startswith('cuda'):
                network.to('cpu')
                moved = True
        if moved:
            import torch
            torch.cuda.empty_cache()

    def unload_all(self):
        if not self.models:
            return
        self.models.clear()
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def health(self):
        return dict(device_policy=os.getenv('RAG_TEXT_DEVICE', 'cuda'),
                    residency_policy='active_model_only_streamed_reload' if os.getenv('RAG_STREAM_CHECKPOINTS','0')=='1'
                        else 'active_model_on_cuda_inactive_weights_in_ram',
                    loaded={name: str(model.device) for name, (_, model) in self.models.copy().items()},
                    loads=dict(self.loads))


def model_inputs(batch, network):
    """Tokenizer tensors and weights must use the same device."""
    return {key: tensor.to(network.device) for key, tensor in batch.items()}


def rerank_passages(runtime, query, passages):
    """Shared BGE scoring for the HTTP gateway and optional local backend."""
    return rerank_pairs(runtime, [[query, text] for text in passages])


def rerank_pairs(runtime, pairs):
    """Mixed questions share a bounded GPU batch with OOM backoff."""
    import torch
    with gpu_stage(), torch.inference_mode():
        tokenizer, network = runtime.get('BAAI/bge-reranker-v2-m3', True)
        def score(chunk):
            inputs = tokenizer(chunk, padding=True,
                               truncation=True, max_length=1024, return_tensors='pt')
            return network(**model_inputs(inputs, network)).logits.view(-1).cpu().float().tolist()
        return run_batches(pairs, 4, score, torch)


def run_batches(items, batch_size, operation, torch):
    """Retry CUDA OOM with smaller batches, preserving every item and its order."""
    result, offset = [], 0
    while offset < len(items):
        chunk = items[offset:offset + batch_size]
        try:
            values = operation(chunk)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            torch.cuda.empty_cache()
            continue
        result.extend(values)
        offset += len(chunk)
    return result


logger = logging.getLogger('ai_extract.pipeline')

if not logger.handlers:
    try:
        stream=open('/proc/1/fd/1','a',encoding='utf-8',buffering=1) if Path('/.dockerenv').is_file() and os.getenv('A3_DOCKER_LOG_CONTAINER') else None
    except OSError:
        stream=None
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)

logger.setLevel(logging.INFO)

logger.propagate = False

class PipelineProgress:
    def __init__(self, run_id):
        value = str(run_id).removeprefix('contract_run_')
        self.run_id = int(value) if value.isdigit() else value

    def emit(self, stage, event, **fields):
        record = dict(run=self.run_id, stage=stage, event=event, **fields)
        level = logging.ERROR if event == 'failed' or fields.get('status') == 'FAILED' else logging.INFO
        if event == 'completed_with_index_errors':
            level = logging.WARNING
        if fields.get('status') in {'partial','unavailable'} or event=='completed_with_source_warnings':
            level=logging.WARNING
        logger.log(level, 'A3 %s', json.dumps(record, ensure_ascii=False))

    @contextmanager
    def stage(self, name, *, heartbeat_seconds=15, **fields):
        started = time.perf_counter()
        stopped = Event()
        result = dict(fields)
        self.emit(name, 'started', **fields)
        def heartbeat():
            while not stopped.wait(heartbeat_seconds):
                self.emit(name, 'waiting', elapsed_s=round(time.perf_counter()-started, 1), **fields)
        worker = Thread(target=heartbeat, daemon=True, name='a3-progress')
        worker.start()
        try:
            yield result
        except BaseException as error:
            stopped.set()
            worker.join()
            self.emit(name, 'failed', elapsed_s=round(time.perf_counter()-started, 2),
                      error_type=type(error).__name__, **result)
            raise
        else:
            stopped.set()
            worker.join()
            self.emit(name, 'completed', elapsed_s=round(time.perf_counter()-started, 2), **result)

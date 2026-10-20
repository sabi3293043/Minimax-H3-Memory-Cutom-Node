"""Per-callback CPU frame sharing for compatible KJ 2D Tiny VAE consumers.

Hooks are context-local in effect and reference-counted in lifetime. Outside a
memory sampler's callback they call the untouched decoder. No core/node files
are changed and no input tensors, model weights or GPU RGB are cached.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import functools
import hashlib
import inspect
from pathlib import Path
import sys
import threading

_CURRENT = ContextVar('minimax_h3_shared_decode',default=None)
_LOCK = threading.RLock()
_HOOKS = {}
_PREVIEW_HOOKS = {}


def tensor_digest(tensor):
    """Exact input/weight identity, including inference tensors without versions."""
    import torch
    host = tensor.detach().cpu().contiguous()
    # A channels-last 1x1 kernel can be "contiguous" while its singleton
    # final dimension has stride >1. Flatten before reinterpreting its bytes.
    return hashlib.sha256(host.reshape(-1).view(torch.uint8).numpy().tobytes()).digest()


def current_session():
    return _CURRENT.get()


class SharedTinyVAE:
    def __init__(self, *, max_bytes=384*1024**2):
        self.max_bytes = max_bytes
        self.frames = OrderedDict()
        self.bytes = self.peak_bytes = 0
        self.active = False
        self.retain_last_callback = False
        self.classes = set()
        self.preview_classes = set()
        self.suppress_default_preview = False
        self.decode_calls = self.decoded_frames = self.reused_frames = self.evicted_frames = 0
        self._token = None

    def __enter__(self):
        self._token = _CURRENT.set(self)
        # KJ Preview Override may have constructed its decoder before our outer
        # wrapper ran. Its class still participates in the same callback scope.
        for module in list(sys.modules.values()):
            namespace = getattr(module,'__dict__',{})
            filename = namespace.get('__file__','') or ''
            if Path(filename).name == 'tiny_vae.py':
                cls = namespace.get('TinyVAEDecoder')
                if isinstance(cls,type):
                    self.register(cls)
        return self

    def register(self, cls):
        if cls in self.classes or getattr(cls,'decodes_prefix',True):
            return
        original = getattr(cls,'decode_video',None)
        if original is None:
            return
        with _LOCK:
            if cls in _HOOKS:
                _HOOKS[cls][2] += 1
            else:
                @functools.wraps(original)
                def decode(decoder, latent, frame_indices=None):
                    session = _CURRENT.get()
                    if session is None or not session.active or getattr(decoder,'latent_channels',None)!=24:
                        return original(decoder,latent,frame_indices=frame_indices)
                    return session.decode(original,decoder,latent,frame_indices)
                _HOOKS[cls] = [original,decode,1]
                cls.decode_video = decode
            self.classes.add(cls)

    def suppress_standard_preview(self):
        """Our player replaces the sampler thumbnail; don't run a second VAE for it."""
        self.suppress_default_preview = True
        module=sys.modules.get('latent_preview')
        base=getattr(module,'LatentPreviewer',None)
        pending=[base] if isinstance(base,type) else []
        while pending:
            cls=pending.pop();pending.extend(cls.__subclasses__())
            if cls in self.preview_classes or 'decode_latent_to_preview_image' not in cls.__dict__:
                continue
            with _LOCK:
                if cls in _PREVIEW_HOOKS:
                    _PREVIEW_HOOKS[cls][2]+=1
                else:
                    original=cls.decode_latent_to_preview_image
                    def make_hook(original):
                        @functools.wraps(original)
                        def image(previewer,*args,**kwargs):
                            session=_CURRENT.get()
                            if session is not None and session.active and session.suppress_default_preview:
                                return None
                            return original(previewer,*args,**kwargs)
                        return image
                    hook=make_hook(original)
                    _PREVIEW_HOOKS[cls]=[original,hook,1]
                    cls.decode_latent_to_preview_image=hook
                self.preview_classes.add(cls)

    @contextmanager
    def step(self):
        # A nested callback belongs to this same step, not another cache epoch.
        if self.active:
            yield
            return
        self.clear()
        self.active = True
        try:
            yield
        finally:
            self.active = False
            if not self.retain_last_callback:
                self.clear()

    @contextmanager
    def completed_step(self):
        """Consume the bounded final callback cache during optional post-inspection.

        Exact input/weight keys still apply; a different completed latent must
        decode normally. No old-step input or GPU tensors are retained.
        """
        previous = self.active
        self.active = True
        try:
            yield
        finally:
            self.active = previous
            self.clear()

    def clear(self):
        self.frames.clear()
        self.bytes = 0

    def decode(self, original, decoder, latent, frame_indices):
        import torch
        if latent.ndim!=5 or latent.shape[0]!=1 or not hasattr(decoder,'model'):
            return original(decoder,latent,frame_indices=frame_indices)
        indices = list(range(latent.shape[2])) if frame_indices is None else list(frame_indices)
        if not indices or any(not isinstance(i,int) or not 0<=i<latent.shape[2] for i in indices):
            return original(decoder,latent,frame_indices=frame_indices)
        # Fingerprint actual weights, not filenames: different decoders/LoRAs,
        # dtypes, devices or modified weights must never share guessed results.
        weights = hashlib.sha256()
        for name,value in decoder.model.state_dict().items():
            weights.update(str((name,tuple(value.shape),tuple(value.stride()),str(value.dtype),str(value.device))).encode())
            weights.update(tensor_digest(value))
        key = (str(Path(inspect.getfile(type(decoder))).resolve()),type(decoder).__qualname__,
               str(getattr(decoder,'dtype',None)),getattr(decoder,'upscale_ratio',None),weights.digest(),
               tuple(latent.shape),tuple(latent.stride()),str(latent.dtype),str(latent.device),
               torch.is_autocast_enabled(latent.device.type),torch.get_autocast_dtype(latent.device.type),
               torch.backends.cudnn.allow_tf32,tensor_digest(latent))
        hits = {i:self.frames[(key,i)] for i in indices if (key,i) in self.frames}
        missing = list(dict.fromkeys(i for i in indices if (key,i) not in self.frames))
        self.reused_frames += len(indices)-sum(i in missing for i in indices)
        fresh = {}
        if missing:
            output = original(decoder,latent,frame_indices=missing)
            self.decode_calls += 1
            self.decoded_frames += len(missing)
            if (not isinstance(output,torch.Tensor) or output.ndim!=4 or output.shape[0]!=len(missing)
                    or output.dtype!=torch.uint8 or output.device.type!='cpu'):
                # Only the known uint8, independent-frame KJ contract is shared.
                if missing==indices:
                    return output
                return original(decoder,latent,frame_indices=indices)
            fresh = dict(zip(missing,output))
            for index,frame in fresh.items():
                size = frame.numel()*frame.element_size()
                if size>self.max_bytes:
                    continue
                while self.frames and self.bytes+size>self.max_bytes:
                    _,old = self.frames.popitem(last=False)
                    self.bytes -= old.numel()*old.element_size()
                    self.evicted_frames += 1
                self.frames[(key,index)] = frame.clone()
                self.bytes += size
                self.peak_bytes = max(self.peak_bytes,self.bytes)
        # Every consumer receives its own CPU tensor, so preview/third-party
        # in-place edits cannot corrupt the memory reference or cached RGB.
        selected = []
        for index in indices:
            cached = hits.get(index)
            if cached is None:
                cached = self.frames.get((key,index))
            if cached is None:
                cached = fresh.get(index)
            if cached is None:
                # A small budget may have evicted an earlier hit while filling
                # misses. Preserve correctness rather than return another frame.
                cached = original(decoder,latent,frame_indices=[index])[0]
                self.decode_calls += 1
                self.decoded_frames += 1
            selected.append(cached)
        return torch.stack(selected)

    def metrics(self):
        return dict(decode_calls=self.decode_calls,decoded_frames=self.decoded_frames,
            reused_frames=self.reused_frames,evicted_frames=self.evicted_frames,
            peak_cpu_bytes=self.peak_bytes,retained_cpu_bytes=self.bytes)

    def __exit__(self,*args):
        self.active = False
        self.clear()
        with _LOCK:
            for cls in self.classes:
                original,hook,count = _HOOKS[cls]
                _HOOKS[cls][2] = count-1
                if count==1:
                    if cls.decode_video is hook:
                        cls.decode_video = original
                    del _HOOKS[cls]
            for cls in self.preview_classes:
                original,hook,count=_PREVIEW_HOOKS[cls]
                _PREVIEW_HOOKS[cls][2]=count-1
                if count==1:
                    if cls.decode_latent_to_preview_image is hook:
                        cls.decode_latent_to_preview_image=original
                    del _PREVIEW_HOOKS[cls]
        _CURRENT.reset(self._token)


@contextmanager
def install_shared_callback(guider,session):
    """Native sampler wrapper brackets the complete callback chain in either order."""
    from comfy.patcher_extension import WrappersMP,copy_nested_dicts,add_wrapper_with_key
    original = guider.model_options
    options = copy_nested_dicts(original)
    def wrap(executor,model_wrap,sigmas,extra_args,callback,noise,latent_image=None,
             denoise_mask=None,disable_pbar=False):
        def shared_callback(*args,**kwargs):
            with session.step():
                if callback is not None:
                    return callback(*args,**kwargs)
        return executor(model_wrap,sigmas,extra_args,shared_callback,noise,latent_image,denoise_mask,disable_pbar)
    add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE,'minimax_shared_tiny_vae',wrap,options,is_model_options=True)
    guider.model_options = options
    try:
        yield
    finally:
        guider.model_options = original

"""Bounded CPU-only observation playback, without a second Tiny VAE decode."""
import base64
import io
import logging
import queue
import threading
import uuid

LOG = logging.getLogger(__name__)


def external_preview_wrappers(guider):
    if guider is None:
        return ()
    options = getattr(guider,'model_options',{}).get('transformer_options',{}).get('wrappers',{})
    patcher = getattr(guider,'model_patcher',None)
    sources = (options,getattr(patcher,'wrappers',{}))
    return tuple(wrapper for wrappers in sources
                 for wrapper in wrappers.get('outer_sample',{}).get('kj_preview_override',()))


def has_external_preview(guider):
    return bool(external_preview_wrappers(guider))


def external_preview_uses_decoder(guider,decoder):
    name=getattr(decoder,'_minimax_tiny_vae_name',None)
    if not name:
        return False
    import folder_paths
    from pathlib import Path
    selected=folder_paths.get_full_path('vae_approx',name)
    if not selected:
        return False
    for wrapper in external_preview_wrappers(guider):
        other=getattr(wrapper,'tiny_vae',None)
        path=folder_paths.get_full_path('vae_approx',other) if other and other!='none' else None
        if path and Path(path).resolve()==Path(selected).resolve():
            return True
    return False


def send(node_id, **payload):
    if node_id is None:
        return
    try:
        from server import PromptServer
        if PromptServer.instance is not None:
            PromptServer.instance.send_sync('minimax.memory_preview',dict(node=node_id,**payload))
    except Exception:
        LOG.debug('Memory preview delivery failed',exc_info=True)


def select_indices(count,limit):
    if count<=limit:
        return tuple(range(count))
    return tuple(round(i*(count-1)/(limit-1)) for i in range(limit)) if limit>1 else (0,)


class MemoryPreview:
    def __init__(self,node_id,*,enabled,frames=16,max_resolution=512,fps=8.):
        self.node_id = node_id
        self.enabled = bool(enabled and node_id is not None)
        self.limit,self.max_resolution,self.fps = frames,max_resolution,fps
        self.pending = queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self.thread = None

    def __enter__(self):
        if self.enabled:
            self.thread = threading.Thread(target=self._run,name='minimax-memory-preview',daemon=True)
            self.thread.start()
        return self

    def submit(self,frames,*,step,total,indices=None):
        if not self.enabled or self.closed.is_set():
            return
        # A bounded copy owned by the encoder; never keep sampler/GPU tensors.
        picks = select_indices(len(frames),self.limit)
        import torch
        selected = torch.stack([frames[i].detach().cpu() for i in picks])
        ids = tuple(indices[i] for i in picks) if indices is not None else picks
        item = (selected,step,total,ids)
        try:
            self.pending.put_nowait(item)
        except queue.Full:
            try:self.pending.get_nowait()
            except queue.Empty:pass
            try:self.pending.put_nowait(item)
            except queue.Full:pass

    def _run(self):
        while not self.closed.is_set():
            item = self.pending.get()
            if item is None:
                return
            try:
                frames,step,total,indices = item
                payload = self.encode(frames,self.max_resolution)
                if not self.closed.is_set():
                    send(self.node_id,frames=payload,step=step+1,total=total,indices=indices,
                         fps=self.fps,width=frames.shape[-1],height=frames.shape[-2],mode='auto')
            except Exception:
                LOG.debug('Memory preview encoding failed; sampling preserved',exc_info=True)

    @staticmethod
    def encode(frames,max_resolution):
        import torch
        from PIL import Image
        result = []
        for frame in frames:
            if frame.dtype!=torch.uint8:
                frame = frame.float().clamp(0,1).mul(255).byte()
            image = Image.fromarray(frame.permute(1,2,0).numpy())
            image.thumbnail((max_resolution,max_resolution))
            stream = io.BytesIO()
            image.save(stream,format='JPEG',quality=80)
            result.append(base64.b64encode(stream.getvalue()).decode('ascii'))
        return result

    def __exit__(self,*args):
        if self.thread is not None:
            try:self.pending.put(None,timeout=2)
            except queue.Full:pass
            self.thread.join(timeout=3)
        self.closed.set()
        while True:
            try:self.pending.get_nowait()
            except queue.Empty:break
        try:self.pending.put_nowait(None)
        except queue.Full:pass


class RepairComparison:
    """Two bounded CPU/JPEG views of already-decoded second-pass predictions.

    The complete pair is sent at completion so losing the initial message does
    not leave an unmatched after view. Nothing is written to disk or decoded.
    Preview failures must never change the sampled result.
    """
    def __init__(self,node_id,*,enabled,frames=16,max_resolution=512,fps=8.):
        self.node_id = node_id
        self.enabled = bool(enabled and node_id is not None)
        self.limit,self.max_resolution,self.fps = frames,max_resolution,fps
        self.run_id = uuid.uuid4().hex
        self.before = None
        self.indices = ()
        self.count = self.width = self.height = 0

    def start(self,frames):
        if not self.enabled:
            return
        try:
            import torch
            self.count = len(frames)
            self.indices = select_indices(self.count,self.limit)
            selected = torch.stack([frames[i].detach().cpu() for i in self.indices])
            self.before = MemoryPreview.encode(selected,self.max_resolution)
            self.height,self.width = frames.shape[-2:]
            self._send(phase='inspecting',outcome='pending',after=None,
                caption='Before: completed first pass · inspecting for repair · silent Tiny VAE preview')
        except Exception:
            self.before = None
            LOG.warning('Before/after preview unavailable; sampling preserved',exc_info=True)

    def finish(self,frames=None,*,outcome,caption):
        if not self.enabled or self.before is None:
            return
        try:
            import torch
            after = None
            if frames is not None:
                if len(frames)!=self.count or tuple(frames.shape[-2:])!=(self.height,self.width):
                    raise ValueError('Before/after preview frames must have matching temporal/spatial geometry')
                selected = torch.stack([frames[i].detach().cpu() for i in self.indices])
                after = MemoryPreview.encode(selected,self.max_resolution)
            self._send(phase='complete',outcome=outcome,after=after,caption=caption)
        except Exception:
            LOG.warning('Before/after preview unavailable; sampled result preserved',exc_info=True)
            self._send(phase='complete',outcome=outcome,after=None,
                caption=caption+' · comparison preview unavailable',unavailable=True)

    def _send(self,*,phase,outcome,after,caption,unavailable=False):
        send(self.node_id,mode='auto',comparison=dict(run_id=self.run_id,phase=phase,
            outcome=outcome,before=self.before,after=after,unavailable=unavailable),indices=self.indices,
            caption=caption,fps=self.fps,width=self.width,height=self.height)

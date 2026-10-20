"""Bounded per-MODEL comparison data, read only after a matching sampler result.

The MODEL node cannot return images produced later by sampling. Its extra socket
carries this handle; a downstream LATENT dependency orders the IMAGE collector.
No global frame cache, model/VAE calls or comparison files are used.
"""
import hashlib
import json
import threading


SOCKET = 'MINIMAX_H3_COMPARISON'


def latent_signature(streams):
    from .shared_tiny_vae import tensor_digest
    signature = hashlib.sha256()
    for value in streams:
        signature.update(json.dumps([list(value.shape),str(value.dtype)]).encode())
        signature.update(tensor_digest(value))
    return signature.hexdigest()


def bounded_images(frames, indices, resolution):
    import numpy as np
    import torch
    from PIL import Image
    result = []
    for index in indices:
        value = frames[index].detach().cpu()
        if value.dtype != torch.uint8:
            value = value.float().clamp(0,1).mul(255).byte()
        image = Image.fromarray(value.permute(1,2,0).numpy())
        image.thumbnail((resolution,resolution))
        result.append(torch.from_numpy(np.array(image,copy=True)))
    return torch.stack(result)


class ComparisonCapture:
    def __init__(self,*,enabled=False,frames=16,resolution=512):
        self.enabled = enabled
        self.limit,self.resolution = frames,resolution
        self._lock = threading.RLock()
        self.reset('No completed second-pass comparison. Run the connected sampler first.')

    def reset(self,reason):
        with self._lock:
            self.before = self.after = self.signature = None
            self.indices = ()
            self.outcome = 'pending'
            self.reason = reason

    def start(self,frames):
        if not self.enabled:
            return
        from .native_preview import select_indices
        with self._lock:
            self.indices = select_indices(len(frames),self.limit)
            self.before = bounded_images(frames,self.indices,self.resolution)

    def finish(self,after,*,outcome,result,latent_shapes,caption):
        if not self.enabled:
            return
        import comfy.utils
        with self._lock:
            if self.before is None:
                raise ValueError('First-pass comparison frames are unavailable')
            images = bounded_images(after,self.indices,self.resolution) if after is not None else None
            if images is not None and images.shape != self.before.shape:
                raise ValueError('Before/after comparison geometry changed')
            signature = latent_signature(comfy.utils.unpack_latents(result,latent_shapes))
            self.after,self.signature = images,signature
            self.outcome,self.reason = outcome,caption

    def read(self,samples):
        if not self.enabled:
            raise ValueError('Enable Before/After comparison on the connected memory node, with Second-pass repair on.')
        value = samples.get('samples') if isinstance(samples,dict) else None
        if value is None:
            raise ValueError('Connect the sampler output LATENT to Comparison Outputs.samples')
        streams = value.unbind() if getattr(value,'is_nested',False) else (value,)
        with self._lock:
            if self.signature is None or self.before is None:
                raise ValueError(self.reason)
            if latent_signature(streams) != self.signature:
                raise ValueError('Comparison does not match these samples. Connect the same sampler\'s output '
                    '(not denoised_output, input latents, a different branch or split video/audio). '
                    'Use one memory node per sampler comparison.')
            before = self.before.float().div(255)
            after = self.after.float().div(255) if self.after is not None else before.clone()
            retained = after if self.outcome=='accepted' else before
            note = self.reason
            if self.after is None:
                note += ' · no attempted after image; after_attempt equals before'
            return before,after,retained,note


class MiniMaxH3ComparisonOutputs:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'comparison':(SOCKET,),
            'samples':('LATENT',{'lazy':True,'tooltip':'Connect the same sampler\'s output, before splitting video/audio. '
                'This dependency makes frames available after sampling. No latent is decoded here.'}),
        }}

    RETURN_TYPES = ('IMAGE','IMAGE','IMAGE','STRING')
    RETURN_NAMES = ('before','after_attempt','retained','status')
    FUNCTION = 'collect'
    CATEGORY = 'MiniMax H3/Memory'

    @classmethod
    def IS_CHANGED(cls,**kwargs):
        # The MODEL's handle is filled later by sampling, including cached MODEL
        # reuse. Always read/verify it rather than caching a previous frame pair.
        return float('nan')

    def collect(self,comparison,samples):
        if not isinstance(comparison,ComparisonCapture):
            raise ValueError('Connect the memory node\'s comparison socket')
        return comparison.read(samples)

    def check_lazy_status(self,comparison,samples=None):
        if not isinstance(comparison,ComparisonCapture) or not comparison.enabled:
            raise ValueError('Enable Before/After comparison and Second-pass repair on the connected memory node.')
        return ['samples'] if samples is None else []

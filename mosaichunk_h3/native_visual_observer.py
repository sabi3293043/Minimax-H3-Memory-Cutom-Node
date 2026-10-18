"""Non-invasive native H3 sampler observer; no denoising guidance yet.

The original sampler, Spectrum, Sage and other model hooks are not replaced.
Each invocation owns a disk session, even when MODEL graphs are shared across
runs. Visual image capture is explicitly experimental and opt-in.
"""
from __future__ import annotations

import importlib.util
import io
import logging
from pathlib import Path
import sys

from .visual_memory_disk import VisualMemoryDisk

LOG = logging.getLogger(__name__)
KEY = "mosaichunk_h3_native_visual_observer"


def choose_distinct_frames(decoded, limit: int):
    """Greedy visual-diversity shortlist, not a trained identity or occlusion model."""
    import torch
    import torch.nn.functional as F

    if decoded.ndim != 4 or decoded.shape[0] < 1 or limit < 1:
        return []
    features = F.adaptive_avg_pool2d(decoded.float(), (4, 4)).flatten(1)
    features = F.normalize(features, dim=-1)
    chosen = [0]
    closest = (features - features[0]).square().sum(-1)
    while len(chosen) < min(limit, len(features)):
        next_id = int(closest.argmax().item())
        if float(closest[next_id]) < 0.0005:
            break
        chosen.append(next_id)
        closest = torch.minimum(closest, (features - features[next_id]).square().sum(-1))
        closest[chosen] = -1
    return sorted(chosen)


def _unpack_video_x0(x0, latent_shapes):
    """Use KJ Preview Override's video normalization for H3 packed AV latents."""
    if not latent_shapes or not hasattr(x0, "ndim"):
        return None
    target = latent_shapes[0]
    if x0.ndim == 3 and len(target) >= 3:
        import math
        count = math.prod(int(d) for d in target[1:])
        if x0.shape[-1] < count:
            return None
        x0 = x0[:, :, :count].reshape([x0.shape[0]] + list(target)[1:])
    return x0 if x0.ndim == 5 and x0.shape[1] == 24 else None


def _load_tae_decoder(name):
    """Reuse the installed KJ tiny-VAE loader instead of copying its decoder."""
    import folder_paths
    home = (Path(folder_paths.__file__).resolve().parent / "custom_nodes"
            / "ComfyUI-KJNodes" / "nodes" / "tiny_vae.py")
    if not home.is_file():
        raise RuntimeError("ComfyUI-KJNodes/nodes/tiny_vae.py is required for TAEH3 observation")
    module_name = "_mosaichunk_h3_kj_tiny_vae"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, home)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    decoder = module.load_tiny_vae_decoder(name)
    if decoder is None or decoder.latent_channels != 24:
        raise ValueError("Select a 24-channel MiniMax H3 TAEH3 decoder in models/vae_approx")
    return decoder


class NativeVisualObserver:
    def __init__(self, *, tiny_vae_name, max_frames, min_progress, disk_gib, temp_root=None):
        self.tiny_vae_name = tiny_vae_name
        self.max_frames = int(max_frames)
        self.min_progress = float(min_progress)
        self.disk_gib = float(disk_gib)
        self.temp_root = temp_root

    def __call__(self, executor, noise, latent_image, sampler, sigmas,
                 denoise_mask=None, callback=None, disable_pbar=False,
                 seed=None, latent_shapes=None):
        decoder = _load_tae_decoder(self.tiny_vae_name)
        with VisualMemoryDisk(self.temp_root, max_bytes=int(self.disk_gib * 1024**3)) as store:
            def observing_callback(step, x0, x, total):
                # Never change the sampler's x0 or x, or skip its original callback.
                if total and step / total >= self.min_progress:
                    try:
                        video = _unpack_video_x0(x0, latent_shapes)
                        if video is not None:
                            # H3's temporal decoder needs the whole sequence to
                            # cover late frames; this may be costly on long clips.
                            frames = decoder.decode_video(video[:1], frame_indices=None)
                            selected = choose_distinct_frames(frames.cpu(), self.max_frames)
                            from PIL import Image
                            for idx in selected:
                                pixels = frames[idx].cpu().movedim(0, -1).numpy()
                                if pixels.dtype != "uint8":
                                    import numpy as np
                                    pixels = (pixels.clip(0, 1) * 255).astype(np.uint8)
                                buffer = io.BytesIO()
                                Image.fromarray(pixels).save(buffer, "WEBP", quality=80)
                                store.save(step=step, frame=idx, data=buffer.getvalue(), fmt="webp")
                    except Exception as exc:
                        LOG.warning("Native H3 visual observation skipped step %s: %s", step, exc)
                if callback is not None:
                    callback(step, x0, x, total)
            return executor(noise, latent_image, sampler, sigmas, denoise_mask,
                            observing_callback, disable_pbar, seed, latent_shapes=latent_shapes)


class MosaiChunkH3NativeVisualObserver:
    """Explicit opt-in experiment. Observes H3; does not guide generation."""
    CATEGORY = "MiniMax H3/MosaiChunk"
    DESCRIPTION = (
        "EXPERIMENTAL native full-video observation and temporary disk image capture. "
        "Uses KJ TAEH3; does not modify native sampling or provide consistency guidance yet. "
        "Can be chained with Model Preview Override; decoding currently runs separately."
    )
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {
            "model": ("MODEL",),
            "tiny_vae_name": (folder_paths.get_filename_list("vae_approx"),
                              {"default": "taeh3.safetensors"}),
            "max_frames": ("INT", {"default": 4, "min": 1, "max": 64}),
            "min_progress": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            "disk_gib": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 100.0, "step": 0.1}),
        }}

    def apply(self, model, tiny_vae_name, max_frames=4, min_progress=0.5, disk_gib=2.0):
        from comfy.patcher_extension import WrappersMP
        if model.get_wrappers(WrappersMP.OUTER_SAMPLE, KEY):
            raise ValueError("Apply Native H3 Visual Observer only once")
        result = model.clone()
        result.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, KEY, NativeVisualObserver(
            tiny_vae_name=tiny_vae_name, max_frames=max_frames,
            min_progress=min_progress, disk_gib=disk_gib))
        return (result,)

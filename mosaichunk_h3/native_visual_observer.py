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
from .visual_memory_index import AdaptiveVisualIndex
from .memory_feedback import retrieve_step_references

TINY_VAE_URL = "https://github.com/madebyollin/taehv/blob/main/safetensors/taeh3.safetensors"

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
    closest[0] = -1
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



def choose_tiny_vae(name, filenames):
    """Honor a user choice; auto prefers MiniMax H3 tiny decoders."""
    names = list(filenames)
    if name and name != "auto":
        if name not in names:
            raise FileNotFoundError(f"Tiny VAE {name!r} is not in models/vae_approx")
        return name
    for preferred in ("taeh3.safetensors", "taeh3_decoder.safetensors"):
        matching = [entry for entry in names
                    if entry.replace("\\", "/").rsplit("/", 1)[-1].lower() == preferred]
        if matching:
            return matching[0]
    raise FileNotFoundError(
        "Install a MiniMax H3 tiny VAE in models/vae_approx and choose it. " + TINY_VAE_URL
    )


def _load_tae_decoder(name):
    """Reuse the installed KJ tiny-VAE loader instead of copying its decoder."""
    import folder_paths
    name = choose_tiny_vae(name, folder_paths.get_filename_list("vae_approx"))
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


def _to_webp(frame):
    """Encode one RGB frame without creating a full-video float32 copy."""
    import torch
    from PIL import Image
    f = frame.detach().to(device="cpu")
    if f.ndim != 3 or f.shape[0] != 3:
        raise ValueError("TAEH3 frame must be [3,H,W]")
    if f.dtype != torch.uint8:
        f = f.float().clamp_(0, 1).mul_(255).to(torch.uint8)
    pixels = f.permute(1, 2, 0).contiguous().numpy()
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="WEBP", quality=80)
    return output.getvalue()


class NativeVisualObserver:
    """Lifecycle-bound observer; does not alter denoising predictions."""

    def __init__(self, *, tiny_vae_name, max_frames, min_progress, disk_gib,
                 max_memory_frames=24, temp_root=None, feedback_hook=None):
        if not 1 <= int(max_frames) <= 64:
            raise ValueError("max_frames must be between 1 and 64")
        if not 1 <= int(max_memory_frames) <= 256:
            raise ValueError("max_memory_frames must be between 1 and 256")
        if not 0 <= float(min_progress) <= 1:
            raise ValueError("min_progress must be between zero and one")
        if float(disk_gib) <= 0:
            raise ValueError("disk_gib must be positive")
        self.tiny_vae_name = tiny_vae_name
        self.max_frames = int(max_frames)
        self.max_memory_frames = int(max_memory_frames)
        self.min_progress = float(min_progress)
        self.disk_gib = float(disk_gib)
        self.temp_root = temp_root
        self.feedback_hook = feedback_hook

    def __call__(self, executor, noise, latent_image, sampler, sigmas,
                 denoise_mask=None, callback=None, disable_pbar=False,
                 seed=None, latent_shapes=None):
        decoder = _load_tae_decoder(self.tiny_vae_name)
        index = AdaptiveVisualIndex(capacity=self.max_memory_frames,
                                    captures_per_step=self.max_frames)
        with VisualMemoryDisk(self.temp_root, max_bytes=int(self.disk_gib * 1024**3)) as store:
            budget_full = False

            def observing_callback(step, x0, x, total):
                nonlocal budget_full
                # Observational failures must not change H3's sampling process
                # or stop native progress and third-party preview callbacks.
                if not budget_full and total and (step + 1) / total >= self.min_progress:
                    try:
                        video = _unpack_video_x0(x0, latent_shapes)
                        if video is not None:
                            # Temporal TAEH3 maintains state; decode all frames
                            # to see late frames. This can be slow on long clips.
                            frames = decoder.decode_video(video[:1], frame_indices=None)
                            # Calculate descriptors once, and retrieve historical
                            # suggestions BEFORE updating the memory index.
                            descriptors = index.descriptors(frames)
                            references = retrieve_step_references(
                                descriptors, index.entries, step=step,
                                max_references=self.max_frames,
                            )
                            # The feedback hook is diagnostic-only. It cannot
                            # mutate the sampler's x0/x or model predictions;
                            # actual guidance requires a validated integration.
                            if references and self.feedback_hook is not None:
                                try:
                                    self.feedback_hook(step, tuple(references), store.read)
                                except Exception:
                                    LOG.warning(
                                        "Native visual feedback hook failed; sampling unaffected",
                                        exc_info=True,
                                    )
                            for frame_id, descriptor, confidence in index.candidates(
                                frames, step=step, descriptors=descriptors
                            ):
                                record = store.save(
                                    step=step, frame=frame_id,
                                    data=_to_webp(frames[frame_id]), fmt="webp"
                                )
                                accepted = index.put(step, frame_id, descriptor,
                                                     confidence, record, on_evict=store.delete)
                                if not accepted:
                                    store.delete(record)
                    except MemoryError as exc:
                        budget_full = True
                        LOG.warning("Native H3 visual memory reached its disk limit: %s", exc)
                    except Exception as exc:
                        LOG.warning("Native H3 visual observation skipped step %s: %s", step, exc)
                if callback is not None:
                    callback(step, x0, x, total)

            return executor(noise, latent_image, sampler, sigmas, denoise_mask,
                            observing_callback, disable_pbar, seed, latent_shapes=latent_shapes)


class MosaiChunkH3NativeVisualObserver:
    """Opt-in native full-video visual observation, not a correction engine."""

    CATEGORY = "MiniMax H3/MosaiChunk"
    DESCRIPTION = (
        "EXPERIMENTAL native H3 visual-memory capture. Pick your 24-channel Tiny "
        "VAE from models/vae_approx. Saves content-selected preview frames in "
        "temporary disk storage, and cleans them up after sampling. "
        "This does not yet modify denoising. Recommended decoder: " + TINY_VAE_URL
    )
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        choices = ["auto"] + folder_paths.get_filename_list("vae_approx")
        return {"required": {
            "model": ("MODEL",),
            "tiny_vae_name": (choices, {
                "default": "auto",
                "tooltip": (
                    "Select your own installed H3 Tiny VAE from models/vae_approx. "
                    "Auto prefers taeh3.safetensors, then taeh3_decoder.safetensors. "
                    "Recommended decoder: " + TINY_VAE_URL
                ),
            }),
            "max_frames": ("INT", {"default": 4, "min": 1, "max": 64,
                "tooltip": "Maximum content-selected frames per observed denoising step; no fixed frame intervals."}),
            "max_memory_frames": ("INT", {"default": 24, "min": 1, "max": 256,
                "tooltip": "Maximum retained RGB memory images. Redundant older images are evicted from disk."}),
            "min_progress": ("FLOAT", {"default": 0.5, "min": 0.0,
                "max": 1.0, "step": 0.05,
                "tooltip": "Begin capturing at this fraction of the native denoising process."}),
            "disk_gib": ("FLOAT", {"default": 2.0, "min": 0.1,
                "max": 100.0, "step": 0.1}),
        }}

    def apply(self, model, tiny_vae_name, max_frames=4, min_progress=0.5,
              disk_gib=2.0, max_memory_frames=24):
        from comfy.patcher_extension import WrappersMP
        if model.get_wrappers(WrappersMP.OUTER_SAMPLE, "mosaichunk_h3_controller"):
            raise ValueError("Native H3 Visual Observer cannot follow the RAVEN/MosaiChunk chunk controller")
        if model.get_wrappers(WrappersMP.OUTER_SAMPLE, KEY):
            raise ValueError("Apply Native H3 Visual Observer only once")
        result = model.clone()
        result.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, KEY, NativeVisualObserver(
            tiny_vae_name=tiny_vae_name, max_frames=max_frames,
            min_progress=min_progress, disk_gib=disk_gib,
            max_memory_frames=max_memory_frames
        ))
        return (result,)

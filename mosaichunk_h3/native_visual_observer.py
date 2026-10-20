"""Native H3 Tiny VAE observer with opt-in experimental appearance guidance.

The original sampler, Spectrum, Sage and other model hooks are not replaced.
Each invocation owns a disk session, even when MODEL graphs are shared across
runs. Visual image capture is explicitly experimental and opt-in.
"""
from __future__ import annotations

import importlib.util
import io
import logging
import math
from pathlib import Path
import sys
import time
from contextlib import contextmanager, nullcontext, ExitStack

from .visual_memory_disk import VisualMemoryDisk
from .visual_memory_index import AdaptiveVisualIndex
from .memory_feedback import retrieve_step_references
from .native_status import notify_native
from .observation_policy import ObservationSettings, ObservationPolicy, resize_observation

TINY_VAE_URL = "https://github.com/madebyollin/taehv/blob/main/safetensors/taeh3.safetensors"

LOG = logging.getLogger(__name__)
KEY = "mosaichunk_h3_native_visual_observer"


@contextmanager
def preserve_observation_rng():
    import torch
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        yield


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
    """Lifecycle-bound observer; optional correction remains execution-local."""

    def __init__(self, *, tiny_vae_name, max_frames, min_progress, disk_gib,
                 max_memory_frames=24, temp_root=None, feedback_hook=None,
                 guidance=None, max_observation_megapixels=64.,
                 observation_interval=1, node_id=None, observation_settings=None,
                 store_observation_images=True, acceleration=None):
        if not 1 <= int(max_frames) <= 64:
            raise ValueError("max_frames must be between 1 and 64")
        if not 1 <= int(max_memory_frames) <= 256:
            raise ValueError("max_memory_frames must be between 1 and 256")
        if not 0 <= float(min_progress) <= 1:
            raise ValueError("min_progress must be between zero and one")
        if not math.isfinite(float(disk_gib)) or float(disk_gib) <= 0:
            raise ValueError("disk_gib must be finite and positive")
        if not 1 <= int(observation_interval) <= 100:
            raise ValueError("observation_interval must be between 1 and 100")
        if not math.isfinite(float(max_observation_megapixels)) or float(max_observation_megapixels) <= 0:
            raise ValueError("max_observation_megapixels must be finite and positive")
        self.tiny_vae_name = tiny_vae_name
        self.max_frames = int(max_frames)
        self.max_memory_frames = int(max_memory_frames)
        self.min_progress = float(min_progress)
        self.disk_gib = float(disk_gib)
        self.temp_root = temp_root
        self.feedback_hook = feedback_hook
        self.guidance = guidance
        self.max_observation_megapixels = float(max_observation_megapixels)
        self.observation_interval = int(observation_interval)
        self.node_id = node_id
        self.observation_settings = (observation_settings or ObservationSettings()).validate(
            min_progress=self.min_progress, interval=self.observation_interval)
        self.store_observation_images = bool(store_observation_images)
        self.acceleration = acceleration

    def __call__(self, executor, noise, latent_image, sampler, sigmas,
                 denoise_mask=None, callback=None, disable_pbar=False,
                 seed=None, latent_shapes=None):
        try:
            return self._call(executor, noise, latent_image, sampler, sigmas,
                denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
        except BaseException as exc:
            notify_native(self.node_id, f'Stopped: {type(exc).__name__}: {str(exc)[:160]}',
                          log=True, state='stopped')
            raise

    def _call(self, executor, noise, latent_image, sampler, sigmas,
              denoise_mask=None, callback=None, disable_pbar=False,
              seed=None, latent_shapes=None):
        guided = self.guidance is not None and self.guidance.enabled
        accelerated = self.acceleration is not None and self.acceleration.mode != 'off'
        if not guided and not accelerated:
            return self._sample(executor, noise, latent_image, sampler, sigmas,
                denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
        from .native_guidance import GuidanceSession, install_guidance
        from .generation_acceleration import install_acceleration
        guider = getattr(executor, 'class_obj', None)
        if guider is None or not hasattr(guider, 'model_options'):
            raise ValueError('Native correction/acceleration requires ComfyUI BasicGuider/CFGGuider')
        if ((guided or (accelerated and self.acceleration.cache)) and
                (float(getattr(guider, 'cfg', 1.)) != 1. or denoise_mask is not None)):
            raise ValueError('Experimental native correction currently requires CFG 1 and no denoise mask')
        if not latent_shapes or len(latent_shapes) != 2:
            raise ValueError('Native correction requires H3 video/audio latents')
        session = GuidanceSession(self.guidance, latent_shapes[0]) if guided else None
        with ExitStack() as stack:
            acceleration = (stack.enter_context(install_acceleration(guider, self.acceleration, session))
                            if accelerated else None)
            if guided:
                stack.enter_context(install_guidance(guider, session))
            try:
                return self._sample(executor, noise, latent_image, sampler, sigmas,
                    denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes,
                    guidance=session, acceleration=acceleration)
            finally:
                if session is not None:
                    LOG.warning('Native Tiny VAE correction: %d observations, %d frame-pair proposals, '
                            '%d attention corrections affecting %d rows. Quality is unvalidated.',
                                session.observations, session.matched_proposals,
                                session.corrections, session.corrected_rows)

    def _sample(self, executor, noise, latent_image, sampler, sigmas,
                denoise_mask=None, callback=None, disable_pbar=False,
                seed=None, latent_shapes=None, guidance=None, acceleration=None):
        import torch
        with preserve_observation_rng():
            decoder = _load_tae_decoder(self.tiny_vae_name)
        disabled_reason = ''
        if latent_shapes and len(latent_shapes[0]) == 5:
            from .native_guidance import decoded_to_latent, frame_numbers
            shape = latent_shapes[0]
            count = (sum(1 if i % 5 == 0 else 4 for i in range(shape[2]))
                     if getattr(decoder, 'decodes_prefix', False) else shape[2])
            pixels = count * shape[3] * shape[4] * getattr(decoder, 'upscale_ratio', 16) ** 2
            if pixels > self.max_observation_megapixels * 1e6:
                message = ('Tiny VAE observation exceeds max_observation_megapixels; '
                           'reduce resolution/frames or raise the explicit observation budget')
                if guidance is not None:
                    raise ValueError(message)
                disabled_reason = message
            if guidance is not None and guidance.config.mode == 'manual_frames':
                pixel_count = sum(1 if i % 5 == 0 else 4 for i in range(shape[2]))
                for frame in (guidance.config.source_frame, *frame_numbers(guidance.config.target_frames)):
                    decoded_to_latent(frame, pixel_count, shape[2])
        index = AdaptiveVisualIndex(capacity=self.max_memory_frames,
                                    captures_per_step=self.max_frames)
        policy = ObservationPolicy(self.observation_settings, min_progress=self.min_progress,
                                   interval=self.observation_interval)
        storage = (VisualMemoryDisk(self.temp_root, max_bytes=int(self.disk_gib * 1024**3))
                   if self.store_observation_images else nullcontext(None))
        with storage as store:
            budget_full = bool(disabled_reason)
            observations = 0
            skipped_observations = 0
            observation_seconds = 0.
            peak_cache_bytes = 0
            preview_size = None
            observation_reason = ''
            mode = guidance.config.mode if guidance is not None else 'observe_only'
            label = ('Observation only (guidance inactive)' if guidance is None else
                     f'Guidance enabled: {mode}')

            def delete_record(record):
                if store is not None:
                    store.delete(record)

            def read_record(record):
                if store is None:
                    raise FileNotFoundError('Observation image storage is disabled; only descriptors are retained')
                return store.read(record)

            def report(state, *, log=False):
                corrections = guidance.corrections if guidance is not None else 0
                frame_pairs = guidance.matched_proposals if guidance is not None else 0
                text = (f'{state}: {label}; {observations} observations, '
                        f'{frame_pairs} frame-pair proposals, {corrections} corrections; '
                        f'observation {observation_seconds:.1f}s; '
                        f'image cache peak {peak_cache_bytes / 1024**2:.1f} MiB')
                if disabled_reason:
                    text += f'; observation disabled: {disabled_reason}'
                text += f'; {skipped_observations} observation skips'
                if preview_size is not None:
                    text += f'; preview {preview_size[1]}x{preview_size[0]}'
                if store is None:
                    text += '; image storage off'
                if acceleration is not None:
                    text += '; ' + acceleration.summary()
                notify_native(self.node_id, text, log=log, state=state.lower(),
                    guidance_mode=mode, observations=observations, corrections=corrections,
                    frame_pair_proposals=frame_pairs,
                    corrected_rows=guidance.corrected_rows if guidance is not None else 0,
                    observation_seconds=observation_seconds, peak_cache_bytes=peak_cache_bytes,
                    skipped_observations=skipped_observations, preview_size=preview_size,
                    observation_reason=observation_reason, store_observation_images=store is not None,
                    acceleration=acceleration.metrics() if acceleration is not None else None,
                    disabled_reason=disabled_reason)

            report('Sampling', log=True)

            def observing_callback(step, x0, x, total):
                nonlocal budget_full, disabled_reason, observations, observation_seconds, peak_cache_bytes
                nonlocal skipped_observations, preview_size, observation_reason
                if guidance is not None:
                    guidance.progress = (step + 1) / max(1, total)
                if acceleration is not None:
                    acceleration.advance((step + 1) / max(1, total))
                # Observational failures must not change H3's sampling process
                # or stop native progress and third-party preview callbacks.
                first_step = max(0, math.ceil(self.min_progress * total) - 1)
                eligible = total > 0 and step >= first_step
                # No subsequent model evaluation can consume the final callback's plan.
                if guidance is not None and total and (step + 1) / total >= guidance.config.end_progress:
                    eligible = False
                    guidance.pairs = ()
                if not budget_full and eligible:
                    started = time.perf_counter()
                    try:
                        video = _unpack_video_x0(x0, latent_shapes)
                        if video is not None:
                            decision = policy.decide(video, step=step, total=total)
                            if acceleration is not None:
                                acceleration.observe_prediction(video, progress=(step + 1) / max(1, total),
                                                                fingerprint=decision.fingerprint)
                            observation_reason = decision.reason
                            if decision.observe:
                                if policy.last_size is not None and decision.size != policy.last_size:
                                    # Decoder outputs at different scales are not calibrated
                                    # to one descriptor distribution. Rebuild references.
                                    index.reset(on_evict=delete_record)
                                    if guidance is not None:
                                        guidance.pairs = ()
                                with preserve_observation_rng():
                                    preview_video = resize_observation(video[:1], decision.size)
                                    frames = decoder.decode_video(preview_video, frame_indices=None)
                                observations += 1
                                preview_size = tuple(frames.shape[-2:])
                                descriptors = index.descriptors(frames)
                                retrieval_options = {}
                                if guidance is not None and guidance.config.mode == 'retrieved_frames':
                                    from .native_guidance import decoded_to_latent
                                    retrieval_options = dict(forward_only=True, frame_groups=tuple(
                                        decoded_to_latent(frame, len(frames), guidance.latent_t)
                                        for frame in range(len(frames))))
                                references = retrieve_step_references(
                                    descriptors, index.entries, step=step,
                                    max_references=self.max_frames, **retrieval_options,
                                )
                                if guidance is not None:
                                    guidance.observe(frames, references, step=step, total=total)
                                if references and self.feedback_hook is not None:
                                    try:
                                        self.feedback_hook(step, tuple(references), read_record)
                                    except Exception:
                                        LOG.warning('Native visual feedback hook failed; sampling unaffected', exc_info=True)
                                for frame_id, descriptor, confidence in index.candidates(
                                    frames, step=step, descriptors=descriptors
                                ):
                                    record = None
                                    if store is not None:
                                        record = store.save(step=step, frame=frame_id,
                                            data=_to_webp(frames[frame_id]), fmt='webp')
                                        peak_cache_bytes = max(peak_cache_bytes, store.used_bytes)
                                    accepted = index.put(step, frame_id, descriptor,
                                        confidence, record, on_evict=delete_record)
                                    if not accepted:
                                        delete_record(record)
                                policy.observed(decision, step=step)
                            else:
                                skipped_observations += 1
                    except (MemoryError, torch.OutOfMemoryError) as exc:
                        if acceleration is not None:
                            acceleration.prediction_unavailable()
                        budget_full = True
                        disabled_reason = f'{type(exc).__name__}: {str(exc)[:160]}'
                        if guidance is not None:
                            guidance.pairs = ()
                        LOG.warning("Native H3 observation disabled for this run after a memory limit: %s", exc)
                    except Exception as exc:
                        if acceleration is not None:
                            acceleration.prediction_unavailable()
                        if guidance is not None:
                            # Never retain stale frame matches after an observation error.
                            guidance.pairs = ()
                        LOG.warning("Native H3 visual observation skipped step %s: %s", step, exc)
                    finally:
                        observation_seconds += time.perf_counter() - started
                        report('Sampling')
                if callback is not None:
                    callback(step, x0, x, total)

            try:
                result = executor(noise, latent_image, sampler, sigmas, denoise_mask,
                                  observing_callback, disable_pbar, seed, latent_shapes=latent_shapes)
            finally:
                # The context manager also closes on BaseException/cancellation.
                if store is not None:
                    store.close()
            label_cleanup = ('images disabled' if store is None else
                             'cleanup deferred' if store.directory.exists() else 'cache removed')
            report(f'Complete ({label_cleanup})', log=True)
            return result


class MosaiChunkH3NativeVisualObserver:
    """Tiny VAE observation and explicitly enabled experimental correction."""

    CATEGORY = "MiniMax H3/MosaiChunk"
    DESCRIPTION = (
        "EXPERIMENTAL native H3 visual-memory capture. Pick your 24-channel Tiny "
        "VAE from models/vae_approx. Optionally saves content-selected preview frames in "
        "temporary disk storage, and cleans them up after sampling. "
        "Manual/retrieved frame guidance adds experimental appearance attention "
        "during native denoising. Default is retrieved_frames at strength 0.10; "
        "use CFG 1. Active guidance disables Spectrum forecasting. Quality is unvalidated. "
        "Recommended decoder: " + TINY_VAE_URL
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
            "max_frames": ("INT", {"default": 8, "min": 1, "max": 64,
                "tooltip": "Maximum content-selected frames per observed denoising step; no fixed frame intervals."}),
            "max_memory_frames": ("INT", {"default": 64, "min": 1, "max": 256,
                "tooltip": "Maximum retained observation descriptors, plus images if storage is enabled. Redundant older records are evicted."}),
            "min_progress": ("FLOAT", {"default": 0.25, "min": 0.0,
                "max": 1.0, "step": 0.05,
                "tooltip": "Begin capturing at this fraction of the native denoising process."}),
            "disk_gib": ("FLOAT", {"default": 2.0, "min": 0.1,
                "max": 100.0, "step": 0.1,
                "tooltip": "Temporary image-cache limit; unused when store_observation_images is disabled."}),
        }, "optional": {
            "guidance_mode": (["retrieved_frames", "manual_frames", "observe_only"], {
                "default": "retrieved_frames",
                "tooltip": "Automatic frame-pair proposals from earlier Tiny VAE observations. Experimental; may find no confident matches. Requires CFG 1. observe_only records images without correction."}),
            "guidance_strength": ("FLOAT", {"default": .10, "min": 0., "max": .3, "step": .01}),
            "guidance_source_frame": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "guidance_target_frames": ("STRING", {"default": "", "tooltip": "Manual mode: comma-separated final-video frame indices, starting at zero. Choose frames where the subject reappears."}),
            "guidance_layers": ("STRING", {"default": "8,16,24", "tooltip": "Experimental H3 layer selection, not a calibrated quality preset."}),
            "guidance_similarity": ("FLOAT", {"default": .85, "min": 0., "max": 1., "step": .01}),
            "max_anchor_tokens": ("INT", {"default": 256, "min": 1, "max": 512}),
            "max_observation_megapixels": ("FLOAT", {"default": 64., "min": 1., "max": 1024., "step": 1.}),
            "observation_interval": ("INT", {"default": 1, "min": 1, "max": 100,
                "tooltip": "Observe every N eligible sampler callbacks. 1 preserves existing behavior; 2 or 3 reduces Tiny VAE work but refreshes memory less often."}),
            "observation_schedule": (["adaptive", "fixed"], {"default": "fixed",
                "tooltip": "Adaptive: refresh when coarse latent predictions change, with warmup and a maximum refresh interval. Experimental; not an object tracker."}),
            "progressive_resolution": ("BOOLEAN", {"default": False,
                "tooltip": "Decode smaller observation latents early, then medium and full size. Final video/model attention remain full resolution. Retrieval references rebuild at each size change."}),
            "observation_start_scale": ("FLOAT", {"default": .5, "min": .25, "max": 1., "step": .05}),
            "observation_full_progress": ("FLOAT", {"default": .75, "min": .05, "max": .9, "step": .05,
                "tooltip": "Reach full observation resolution at this fraction of denoising; must exceed min_progress."}),
            "adaptive_change_threshold": ("FLOAT", {"default": .04, "min": .001, "max": 1., "step": .005}),
            "adaptive_max_interval": ("INT", {"default": 3, "min": 1, "max": 100,
                "tooltip": "Maximum callbacks between adaptive observations; must be at least observation_interval. Initial observations at each size seed retrieval."}),
            "store_observation_images": ("BOOLEAN", {"default": False,
                "tooltip": "Disable to retain bounded RGB descriptors in RAM without encoding images or creating an image-cache directory. Retrieval decisions are unchanged; diagnostic image reads are unavailable."}),
            "generation_acceleration": (["off", "sol", "selective_cache", "sol_and_cache"], {
                "default": "off", "tooltip": "Optional experimental generation acceleration, independent of Spectrum. Try sol first, then sol_and_cache. Memory layers remain real/dense. Approximate; compare picture, motion and audio. Turn other generation caches and Spectrum off."}),
            "sol_tau": ("FLOAT", {"default": 1., "min": 0., "max": 2., "step": .05,
                "tooltip": "Higher is sparser/faster with more approximation. Uses installed ComfyUI Sol kernels; unsupported calls retain existing dense/Sage attention."}),
            "acceleration_start": ("FLOAT", {"default": .25, "min": 0., "max": .95, "step": .05}),
            "acceleration_end": ("FLOAT", {"default": .85, "min": .05, "max": 1., "step": .05}),
            "cache_threshold": ("FLOAT", {"default": .05, "min": .001, "max": .3, "step": .005,
                "tooltip": "Conservative sampled residual-change threshold. Lower refreshes more; higher reuses more approximate work. Each video latent frame and the audio/conditioning segments get a separate vote."}),
            "cache_max_reuse": ("INT", {"default": 1, "min": 1, "max": 3,
                "tooltip": "Maximum consecutive reuses before refreshing a section. Probe and memory blocks always execute."}),
            "cache_vram_gib": ("FLOAT", {"default": .5, "min": .1, "max": 8., "step": .1,
                "tooltip": "Retained GPU-cache budget, additional to model/activations. GPU residuals keep 1 GiB headroom; RAM storage uses bounded staging with 256 MiB headroom. These are admission guards, not a total VRAM cap."}),
            "cache_storage": (["gpu", "ram", "auto"], {"default": "gpu",
                "tooltip": "GPU preserves the original cache. RAM stores residuals in bounded system RAM with small GPU transfers. Auto prefers GPU and falls back to RAM when a full section does not fit. No disk writes."}),
            "cache_ram_gib": ("FLOAT", {"default": 2., "min": .1, "max": 16., "step": .1}),
            "cache_transfer_mib": ("INT", {"default": 16, "min": 1, "max": 64,
                "tooltip": "Maximum staging chunk for RAM-cache transfers. Cache storage retains the original dtype; no additional quantization."}),
            "adaptive_sol": ("BOOLEAN", {"default": False,
                "tooltip": "Experimental prediction-drift heuristic: slightly more sparse after stable predictions, denser after large changes and near the end. Uses latent thumbnails; not an object/motion detector. Tau changes refresh residual caches."}),
            "sol_stable_boost": ("FLOAT", {"default": .15, "min": 0., "max": .3, "step": .05}),
            "sol_drift_threshold": ("FLOAT", {"default": .04, "min": .001, "max": 1., "step": .005}),
        }, "hidden": {"unique_id": "UNIQUE_ID"}}

    def apply(self, model, tiny_vae_name, max_frames=8, min_progress=0.25,
              disk_gib=2.0, max_memory_frames=64, guidance_mode='retrieved_frames',
              guidance_strength=.10, guidance_source_frame=0, guidance_target_frames='',
              guidance_layers='8,16,24', guidance_similarity=.85, max_anchor_tokens=256,
              max_observation_megapixels=64., observation_interval=1, unique_id=None,
              observation_schedule='fixed', progressive_resolution=False,
              observation_start_scale=.5, observation_full_progress=.75,
              adaptive_change_threshold=.04, adaptive_max_interval=3,
              store_observation_images=False, generation_acceleration='off', sol_tau=1.,
              acceleration_start=.25, acceleration_end=.85, cache_threshold=.05,
              cache_max_reuse=1, cache_vram_gib=.5, cache_storage='gpu', cache_ram_gib=2.,
              cache_transfer_mib=16, adaptive_sol=False, sol_stable_boost=.15, sol_drift_threshold=.04):
        from comfy.patcher_extension import WrappersMP
        from .native_guidance import GuidanceConfig
        from .generation_acceleration import AccelerationConfig
        acceleration = AccelerationConfig(mode=generation_acceleration, sol_tau=float(sol_tau),
            start=float(acceleration_start), end=float(acceleration_end),
            cache_threshold=float(cache_threshold), cache_max_reuse=int(cache_max_reuse),
            cache_gib=float(cache_vram_gib), cache_storage=cache_storage, cache_ram_gib=float(cache_ram_gib),
            transfer_mib=int(cache_transfer_mib), adaptive_sol=bool(adaptive_sol),
            sol_stable_boost=float(sol_stable_boost), sol_drift_threshold=float(sol_drift_threshold)).validate()
        guidance = GuidanceConfig(mode=guidance_mode, strength=float(guidance_strength),
            source_frame=int(guidance_source_frame), target_frames=guidance_target_frames,
            layers=guidance_layers, similarity=float(guidance_similarity),
            max_anchor_tokens=int(max_anchor_tokens)).validate()
        if model.get_wrappers(WrappersMP.OUTER_SAMPLE, "mosaichunk_h3_controller"):
            raise ValueError("Native H3 Visual Observer cannot follow the RAVEN/MosaiChunk chunk controller")
        if model.get_wrappers(WrappersMP.OUTER_SAMPLE, KEY):
            raise ValueError("Apply Native H3 Visual Observer only once")
        result = model.clone()
        result.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, KEY, NativeVisualObserver(
            tiny_vae_name=tiny_vae_name, max_frames=max_frames,
            min_progress=min_progress, disk_gib=disk_gib,
            max_memory_frames=max_memory_frames, guidance=guidance,
            max_observation_megapixels=max_observation_megapixels,
            observation_interval=observation_interval, node_id=unique_id,
            observation_settings=ObservationSettings(schedule=observation_schedule,
                progressive_resolution=bool(progressive_resolution),
                start_scale=float(observation_start_scale), full_progress=float(observation_full_progress),
                change_threshold=float(adaptive_change_threshold), max_interval=int(adaptive_max_interval)),
            store_observation_images=store_observation_images, acceleration=acceleration
        ))
        if guidance.enabled:
            # Own the execution scope before Spectrum starts a runtime, regardless
            # of whether the user connected this node before or after Spectrum.
            outer = result.wrappers[WrappersMP.OUTER_SAMPLE]
            result.wrappers[WrappersMP.OUTER_SAMPLE] = {KEY: outer[KEY], **{k: v for k, v in outer.items() if k != KEY}}
        return (result,)

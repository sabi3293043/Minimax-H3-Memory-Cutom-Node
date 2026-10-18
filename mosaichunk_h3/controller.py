"""Execution-local RAVEN controller hosted by ordinary ComfyUI samplers.

The outgoing MODEL stays native until sampling. Unsupported image/reference
modes therefore use the original model, LoRAs, guider, sampler and offloader.
Only text-only FL2VA runs prepare an independent RAVEN model.
"""

import logging

import torch

from .conditioning import prepare_model
from .consistency import SamplerConfig, sample_streaming
from .contracts import parse_conditioning, parse_latent, resolve_transformer_options
from .pruned import tensor_sha256

LOG = logging.getLogger(__name__)
KEY = "mosaichunk_h3_controller"
FL2VA_TABLE = "ac8727cdec52137c73878d004de5bd2a0e19227e8311e29ab3b68f328310e34e"
MODES = ("auto", "text_to_video", "first_last_frame", "reference", "off")


def select_mode(controls, conds, root):
    mode = controls.get("generation_mode", "auto")
    if mode not in MODES:
        raise ValueError(f"Unknown H3 controller mode: {mode}")
    if mode in ("off", "first_last_frame", "reference"):
        return mode
    # Check every conditioning branch. Explicit text_to_video must not discard
    # image/reference rows either. Tensors must never be coerced to bool here.
    for entries in conds.values():
        for entry in entries or ():
            if entry.get("minimax_refs") is not None:
                return "reference"
            if entry.get("minimax_keyframes") is not None:
                return "first_last_frame"
    table = getattr(root, "adaln_t_table", None)
    if table is None or tensor_sha256(table) != FL2VA_TABLE:
        return "unsupported_checkpoint"
    return "text_to_video"


def notify(node_id, message):
    LOG.warning("[MosaiChunk H3] %s", message)
    # A status widget is optional; a headless ComfyUI still gets the log notice.
    try:
        from server import PromptServer
        if PromptServer.instance is not None:
            PromptServer.instance.send_sync("mosaichunk.controller_status",
                                            {"node": node_id, "message": message})
    except Exception:
        LOG.debug("Could not deliver optional memory status widget event", exc_info=True)


def strip_controller(model):
    """Remove only our wrapper, keeping all unrelated hooks for validation."""
    from comfy.patcher_extension import WrappersMP, copy_nested_dicts
    model = model.clone()
    model.remove_wrappers_with_key(WrappersMP.OUTER_SAMPLE, KEY)
    model.wrappers = {kind: groups for kind, groups in model.wrappers.items() if groups}
    model.model_options = copy_nested_dicts(model.model_options)
    to = model.model_options.get("transformer_options", {})
    wrappers = to.get("wrappers", {})
    groups = wrappers.get(WrappersMP.OUTER_SAMPLE, {})
    groups.pop(KEY, None)
    if not groups:
        wrappers.pop(WrappersMP.OUTER_SAMPLE, None)
    if not wrappers:
        to.pop("wrappers", None)
    return model


class MemoryController:
    def __init__(self, conditioning_path, adapter_path, controls, cache_factory_builder, node_id=None):
        self.conditioning_path = conditioning_path
        self.adapter_path = adapter_path
        self.controls = dict(controls)
        self.cache_factory_builder = cache_factory_builder
        self.node_id = node_id

    def __call__(self, executor, noise, latent_image, sampler, sigmas,
                 denoise_mask=None, callback=None, disable_pbar=False, seed=None, latent_shapes=None):
        guider = executor.class_obj
        mode = select_mode(self.controls, guider.original_conds,
                           guider.model_patcher.get_model_object("diffusion_model"))
        if mode != "text_to_video":
            notify(self.node_id, f"Memory inactive ({mode}); using normal H3 generation and sampler.")
            return executor(noise, latent_image, sampler, sigmas, denoise_mask,
                            callback, disable_pbar, seed, latent_shapes=latent_shapes)
        # RAVEN is a consistency model. The host provides step count, seed and
        # initial noise; its numerical solver/sigma values are explicitly replaced
        # by the trained fresh-noise recipe, rather than approximated as an ODE.
        if not torch.isfinite(sigmas).all() or len(sigmas) < 2:
            raise ValueError("MosaiChunk memory requires a finite schedule with at least one step")
        if abs(float(sigmas[0]) - 1.) > 1e-5 or float(sigmas[-1]) != 0.:
            raise ValueError("MosaiChunk memory requires full denoise (sigma 1 to 0); partial denoise is unsupported")
        if denoise_mask is not None:
            raise ValueError("MosaiChunk T2V memory does not support a latent noise mask")
        if float(guider.cfg) != 1.:
            raise ValueError("MosaiChunk RAVEN memory requires CFG 1 (use BasicGuider or CFGGuider with cfg=1)")
        if type(guider).__name__ not in ("CFGGuider", "Guider_Basic"):
            raise ValueError("MosaiChunk memory supports BasicGuider/CFGGuider only; custom guider behavior cannot be discarded")

        import comfy.model_management as mm
        import comfy.utils
        from comfy.nested_tensor import NestedTensor

        if latent_shapes is None or len(latent_shapes) != 2:
            raise ValueError("MosaiChunk memory requires the native empty H3 video/audio latent pair")
        latent = {"samples": NestedTensor(comfy.utils.unpack_latents(latent_image, latent_shapes))}
        positive = []
        for entry in guider.original_conds.get("positive", ()):
            extras = {k: v for k, v in entry.items() if k not in ("cross_attn", "model_conds", "uuid")}
            if entry.get("model_conds"):
                raise ValueError("Preprocessed model conditions are unsupported in MosaiChunk T2V memory")
            positive.append([entry.get("cross_attn"), extras])
        # Validate before reopening the local model or reading the 5 GB adapter.
        parse_conditioning(positive)
        parse_latent(latent)
        if not self.conditioning_path or not self.adapter_path:
            raise ValueError("Select the converted FL2VA conditioning file and RAVEN adapter for T2V memory")
        clean = strip_controller(guider.model_patcher)
        resolve_transformer_options(clean)
        steps = len(sigmas) - 1
        controls = self.controls
        notify(self.node_id, f"Memory active: T2V, RAVEN consistency, {steps} steps per chunk. "
                             "Using sampler seed/noise/step count; solver and sigma grid replaced by RAVEN recipe.")
        prepared = prepare_model(clean, self.conditioning_path, self.adapter_path, controls)
        config = SamplerConfig(steps=steps, seed=int(seed or 0),
                               video_shift=controls["video_shift"], audio_shift=controls["audio_shift"],
                               sink=1, window=2 + controls["far_memory_chunks"],
                               kv_cache_storage=controls["kv_cache_storage"])
        request = parse_latent(latent, warn_experimental=False)
        layout = request.layout(parse_conditioning(positive).text_len, warn_experimental=False)
        from .nodes import estimate_rollout_budget, dit_dimensions
        budget = estimate_rollout_budget(layout=layout, text_len=layout.text_len,
            sink=config.sink, window=config.window, dims=dit_dimensions(prepared),
            kv_cache_storage=config.kv_cache_storage)
        callback_workspace = 4 * latent_image.numel() * 4 if callback is not None else 0
        total = layout.num_chunks * steps
        pbar = comfy.utils.ProgressBar(total)

        def on_step(chunk, step, vx0, ax0, vxt, axt, completed_v, completed_a):
            index = chunk.index * steps + step
            mm.throw_exception_if_processing_interrupted()
            if callback is not None:
                # CFGGuider's AV callback expects full-clip packed shapes, not
                # just this chunk. Completed chunks stay intact for previews.
                def assemble(v, a):
                    out_v, out_a = completed_v.clone(), completed_a.clone()
                    out_v[:, :, chunk.video_start:chunk.video_stop] = v
                    out_a[:, :, :, chunk.audio_start:chunk.audio_stop] = a
                    return comfy.utils.pack_latents((out_v, out_a))[0]
                # SamplerCustom(Advanced) processes its captured x0 with the
                # incoming BaseModel.process_latent_out later. Encode that
                # callback into the incoming model's carry convention, which
                # can depend on a previous native run's latent_shapes. The
                # final output below stays in native AV latent space.
                x0 = assemble(vx0, ax0)
                x0 = NestedTensor(comfy.utils.unpack_latents(x0, latent_shapes))
                x0 = guider.model_patcher.model.process_latent_in(x0)
                callback(index, comfy.utils.pack_latents(x0.unbind())[0], assemble(vxt, axt), total)
            if not disable_pbar:
                pbar.update_absolute(index + 1, total)

        def load_models(models, **kwargs):
            # Reserve chunk attention workspace using the same native estimate
            # as the old sampler; leave residency decisions to ComfyUI.
            mm.load_models_gpu(models, memory_required=budget.total_bytes + callback_workspace, force_full_load=False)
            prepared.pre_run()

        try:
            with mm.cuda_device_context(prepared.load_device), torch.inference_mode():
                result = sample_streaming(
                    model=prepared, positive=positive, latent=latent, config=config,
                    initial_noise=tuple(comfy.utils.unpack_latents(noise, latent_shapes)),
                    on_step=on_step, load_models=load_models,
                    cancel_check=mm.throw_exception_if_processing_interrupted,
                    cache_factory=self.cache_factory_builder(prepared),
                )
            notify(self.node_id, f"Memory complete: {result.num_chunks} chunks; history released.")
            return comfy.utils.pack_latents(result.latent["samples"].unbind())[0].float()
        finally:
            prepared.cleanup()


def attach_controller(model, conditioning_path, adapter_path, controls, cache_factory_builder, node_id=None):
    from comfy.patcher_extension import WrappersMP
    for key, choices in (("generation_mode", MODES), ("memory_mode", ("mosaichunk", "matched_window")),
                         ("offloading", ("inherit", "static")), ("kv_cache_storage", ("cpu", "cpu_pinned")),
                         ("far_memory_chunks", (1, 2))):
        if controls.get(key) not in choices:
            raise ValueError(f"Invalid MosaiChunk control {key}: {controls.get(key)!r}")
    if KEY in model.wrappers.get(WrappersMP.OUTER_SAMPLE, {}):
        raise ValueError("Apply the MosaiChunk H3 controller only once per MODEL chain")
    result = model.clone()
    result.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, KEY,
        MemoryController(conditioning_path, adapter_path, controls, cache_factory_builder, node_id))
    return result

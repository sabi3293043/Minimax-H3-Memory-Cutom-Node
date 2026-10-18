"""Execution-local RAVEN controller hosted by ordinary ComfyUI samplers.

The outgoing MODEL stays native until sampling. Active modes prepare an
independent RAVEN model and retain native image/audio conditioning as a
permanent prefix. Off mode delegates to the original sampling path.
"""

import logging

import torch

from .conditioning import prepare_model
from .consistency import SamplerConfig, sample_streaming
from .contracts import parse_conditioning, parse_latent, resolve_transformer_options
from .pruned import tensor_sha256
from .image_context import split_conditioning, ConditionedLayout

LOG = logging.getLogger(__name__)
KEY = "mosaichunk_h3_controller"
FL2VA_TABLE = "ac8727cdec52137c73878d004de5bd2a0e19227e8311e29ab3b68f328310e34e"
MODES = ("auto", "text_to_video", "first_last_frame", "reference", "off")


def select_mode(controls, conds, root):
    mode = controls.get("generation_mode", "auto")
    if mode not in MODES:
        raise ValueError(f"Unknown H3 controller mode: {mode}")
    if mode == "off":
        return mode
    # Inspect the positive branch used at CFG 1. Explicit text_to_video must not discard
    # image/reference rows either. Tensors must never be coerced to bool here.
    for entries in (conds.get('positive', ()),):
        for entry in entries or ():
            refs = entry.get("minimax_refs")
            frames = entry.get("minimax_keyframes")
            if refs is not None and (not isinstance(refs, (tuple, list)) or len(refs)):
                return "reference"
            if frames is not None and (not isinstance(frames, (tuple, list)) or len(frames)):
                return "first_last_frame"
    if mode in ('first_last_frame', 'reference'):
        raise ValueError(f"{mode} requires encoded image/keyframe/reference conditioning on the positive input")
    table = getattr(root, "adaln_t_table", None)
    if table is None:
        return "unsupported_checkpoint"
    return "text_to_video"


def conditioning_baseline(name='auto'):
    import folder_paths
    from .conditioning_transfer import read_baseline_table
    if name != 'auto':
        return folder_paths.get_full_path_or_raise('diffusion_models', name)
    # This known original checkpoint is a delta baseline, never the output
    # generator. Do not guess that an arbitrary similarly-named finetune is it.
    candidates = ('minimax_h3_fl2va_pruned_int8_convrot.safetensors',
                  'minimax_h3_fl2va_pruned_w4a8.safetensors')
    for basename in candidates:
        for filename in folder_paths.get_filename_list('diffusion_models'):
            if filename.replace('\\', '/').rsplit('/', 1)[-1].lower() == basename:
                path = folder_paths.get_full_path_or_raise('diffusion_models', filename)
                if tensor_sha256(read_baseline_table(path)) == FL2VA_TABLE:
                    return path
    raise ValueError('Select your original FL2VA pruned checkpoint in conditioning_base_unet; no automatic model download is performed')


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


def is_preview_wrapper(wrapper):
    """Known callback-only adapter; never infer compatibility from its key alone."""
    cls = type(wrapper)
    return (cls.__name__ == '_PreviewOverrideWrapper'
            and cls.__module__.rsplit('.', 1)[-1] == 'preview_override_node')


def strip_controller(model, *, handled_previews=False):
    """Remove hooks executed outside the chunk loop, preserving all others."""
    from comfy.patcher_extension import WrappersMP, copy_nested_dicts
    model = model.clone()
    model.remove_wrappers_with_key(WrappersMP.OUTER_SAMPLE, KEY)
    model.wrappers = {kind: groups for kind, groups in model.wrappers.items() if groups}
    model.model_options = copy_nested_dicts(model.model_options)
    to = model.model_options.get("transformer_options", {})
    wrappers = to.get("wrappers", {})
    groups = wrappers.get(WrappersMP.OUTER_SAMPLE, {})
    groups.pop(KEY, None)
    if handled_previews:
        for registry in (model.wrappers, wrappers):
            outer = registry.get(WrappersMP.OUTER_SAMPLE, {})
            for key, hooks in list(outer.items()):
                remaining = [hook for hook in hooks if not is_preview_wrapper(hook)]
                if remaining:
                    outer[key] = remaining
                else:
                    del outer[key]
            if not outer:
                registry.pop(WrappersMP.OUTER_SAMPLE, None)
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
        if mode in ('off', 'unsupported_checkpoint'):
            notify(self.node_id, f"Memory inactive ({mode}); using normal H3 generation and sampler.")
            return executor(noise, latent_image, sampler, sigmas, denoise_mask,
                            callback, disable_pbar, seed, latent_shapes=latent_shapes)
        # Native OUTER_SAMPLE hooks may occur on either side of this node.
        # Earlier preview hooks already wrap us. Execute later ones around the
        # chunk loop instead of skipping them or calling the native ODE sampler.
        clean = strip_controller(guider.model_patcher, handled_previews=True)
        resolve_transformer_options(clean)
        from comfy.patcher_extension import WrapperExecutor
        pending = executor.wrappers[executor.idx + 1:]
        if any(not is_preview_wrapper(hook) for hook in pending):
            raise ValueError('MosaiChunk memory cannot honour an unknown outer_sample wrapper')

        def terminal(*args, **kwargs):
            return self.sample_memory(guider, mode, clean, *args, **kwargs)

        return WrapperExecutor.new_class_executor(terminal, guider, pending).execute(
            noise, latent_image, sampler, sigmas, denoise_mask, callback,
            disable_pbar, seed, latent_shapes=latent_shapes)

    def sample_memory(self, guider, mode, clean, noise, latent_image, sampler, sigmas,
                      denoise_mask=None, callback=None, disable_pbar=False,
                      seed=None, latent_shapes=None):
        # RAVEN is a consistency model. The host provides step count, seed and
        # initial noise; its numerical solver/sigma values are explicitly replaced
        # by the trained fresh-noise recipe, rather than approximated as an ODE.
        if not torch.isfinite(sigmas).all() or len(sigmas) < 2:
            raise ValueError("MosaiChunk memory requires a finite schedule with at least one step")
        if abs(float(sigmas[0]) - 1.) > 1e-5 or float(sigmas[-1]) != 0.:
            raise ValueError("MosaiChunk memory requires full denoise (sigma 1 to 0); partial denoise is unsupported")
        if denoise_mask is not None:
            raise ValueError("MosaiChunk H3 memory does not support a latent noise mask")
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
                raise ValueError("Preprocessed model conditions are unsupported in MosaiChunk H3 memory")
            positive.append([entry.get("cross_attn"), extras])
        # Validate before reopening the local model or reading the 5 GB adapter.
        request = parse_latent(latent)
        positive, image_context = split_conditioning(positive, request)
        parse_conditioning(positive)
        if not self.conditioning_path or not self.adapter_path:
            raise ValueError("Select the converted FL2VA conditioning file and RAVEN adapter for H3 memory")
        steps = len(sigmas) - 1
        controls = dict(self.controls)
        # Preserve the incoming variant's modulation even for text-only use
        # of REF2VA/all-in-one checkpoints. On the original FL2VA base this
        # delta addition reproduces the converted conditioning itself.
        controls['conditioning_transfer'] = True
        controls['conditioning_base_path'] = conditioning_baseline(controls.get('conditioning_base_unet', 'auto'))
        notify(self.node_id, f"Preparing memory: {mode}, RAVEN consistency, {steps} steps per chunk. "
                             "Using sampler seed/noise/step count; solver and sigma grid replaced by RAVEN recipe.")
        config = SamplerConfig(steps=steps, seed=int(seed or 0),
                               video_shift=controls["video_shift"], audio_shift=controls["audio_shift"],
                               sink=1, window=2 + controls["far_memory_chunks"],
                               kv_cache_storage=controls["kv_cache_storage"])
        if image_context.active:
            image_context.validate_schedule(config)
        prepared = prepare_model(clean, self.conditioning_path, self.adapter_path, controls)
        request = parse_latent(latent, warn_experimental=False)
        layout = request.layout(parse_conditioning(positive).text_len, warn_experimental=False)
        if image_context.active:
            layout = ConditionedLayout(layout, image_context)
        from .nodes import estimate_rollout_budget, dit_dimensions
        budget = estimate_rollout_budget(layout=layout, text_len=layout.prefix_rows,
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
            notify(self.node_id, f"Memory active: {mode}; {layout.prefix_rows} permanent context rows, {steps} RAVEN steps per chunk."
                                 + (' Experimental image-conditioned router/adapter transfer.' if image_context.active else ''))

        try:
            with mm.cuda_device_context(prepared.load_device), torch.inference_mode():
                result = sample_streaming(
                    model=prepared, positive=positive, latent=latent, config=config,
                    initial_noise=tuple(comfy.utils.unpack_latents(noise, latent_shapes)),
                    image_context=image_context,
                    on_step=on_step, load_models=load_models,
                    cancel_check=mm.throw_exception_if_processing_interrupted,
                    cache_factory=self.cache_factory_builder(prepared),
                )
            notify(self.node_id, f"Memory complete: {mode}, {result.num_chunks} chunks; "
                                 f"{result.kv_cache.get('retrievals', 0)} historical retrievals; history released.")
            return comfy.utils.pack_latents(result.latent["samples"].unbind())[0].float()
        except Exception as exc:
            notify(self.node_id, f"Memory stopped: {type(exc).__name__}: {str(exc)[:200]}")
            raise
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

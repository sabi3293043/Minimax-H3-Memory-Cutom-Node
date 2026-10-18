"""Prepare an existing ComfyUI MODEL for RAVEN without replacing its loader."""

import torch

from . import lora
from .pruned import attach_pruned_raven, read_conditioning


def prepare_model(model, conditioning_path, adapter_path, controls, *, clone=True, disable_dynamic=False):
    import comfy.model_management
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from .causal_model import RavenCausalMiniMaxH3Model, MosaiChunkStreamingMiniMaxH3Model, RavenCausalDiTBlock, RavenCausalAttention

    root = model.get_model_object("diffusion_model")
    if not isinstance(root, MiniMaxH3Model) or not root.use_adaln_curves:
        raise ValueError("MosaiChunk conditioning requires a native ComfyUI pruned FL2VA H3 MODEL")
    if isinstance(root, RavenCausalMiniMaxH3Model):
        raise ValueError("This MODEL already has RAVEN conditioning; apply this node once")
    if any(block.attn.to_gate_compress is not None for block in root.blocks):
        raise ValueError("FastVideo/VSA finetunes are not the FL2VA base required by RAVEN")
    converted = read_conditioning(conditioning_path, root.adaln_t_table, adapter_path)
    factory = model.cached_patcher_init
    if factory is None:
        raise ValueError("Use ComfyUI's UNET Loader: this MODEL has no reload factory for an independent conditioned copy")
    # Keep the incoming native offloading policy unless static was explicitly
    # requested. Reopening the same local file isolates the original branch.
    static = disable_dynamic or controls.get("offloading", "inherit") == "static"
    prepared = model.clone(disable_dynamic=static, force_deepcopy=True) if clone else model
    root = prepared.model.diffusion_model
    root.__class__ = MosaiChunkStreamingMiniMaxH3Model
    root.adaln_t_table = converted["adaln_t_table"]
    for index, block in enumerate(root.blocks):
        block.__class__ = RavenCausalDiTBlock
        block.attn.__class__ = RavenCausalAttention
        block.attn.layer_idx = index
        block.attn._attention_site = ("dit", index)
    for prefix in [f"blocks.{i}" for i in range(len(root.blocks))] + ["final_layer"]:
        name = prefix + ".adaln_proj.linear"
        linear = root.get_submodule(name)
        linear.weight = torch.nn.Parameter(converted[name + ".weight"], requires_grad=False)
        linear.bias = torch.nn.Parameter(converted[name + ".bias"], requires_grad=False)
    manifest = lora.manifest_from_file(adapter_path)
    attachment, manifest = attach_pruned_raven(prepared.model, manifest)
    prepared.model.raven_lora_attachment = attachment
    prepared.model.raven_lora_manifest = manifest
    prepared.model_options["mosaichunk_controls"] = dict(controls)
    prepared.model_options["mosaichunk_dynamic_ready"] = bool(prepared.is_dynamic())
    # Cloning measured the base before new residuals were attached. Quantized
    # state_dict now includes those factors, so this accounts for every byte.
    prepared.size = comfy.model_management.module_size(prepared.model)
    prepared.cached_patcher_init = (rebuild_model, (factory, conditioning_path, adapter_path, dict(controls)))
    return prepared


def rebuild_model(factory, conditioning_path, adapter_path, controls, disable_dynamic=False):
    static = disable_dynamic or controls.get("offloading", "inherit") == "static"
    model = factory[0](*factory[1], disable_dynamic=static)
    if len(factory) > 2:
        model = model[factory[2]]
    return prepare_model(model, conditioning_path, adapter_path, controls, clone=False, disable_dynamic=static)

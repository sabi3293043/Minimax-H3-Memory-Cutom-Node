"""RAVEN conditioning conversion and quantized, curve-form H3 attachment."""

import dataclasses
import hashlib

import torch
import torch.nn.functional as F
from safetensors import safe_open

from . import lora, runtime_linear


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def fit_time_curve(embeddings, rank=8):
    """Factor the adapted smooth time curve, retaining its affine offset."""
    x = embeddings.double()
    mean = x.mean(0)
    _, _, vh = torch.linalg.svd(x - mean, full_matrices=False)
    basis = vh[:rank]
    coords = (x - mean) @ basis.T
    residual = torch.linalg.vector_norm(coords @ basis + mean - x) / torch.linalg.vector_norm(x)
    return coords.float(), basis.float(), mean.float(), float(residual)


def fit_existing_time_curve(embeddings, table):
    """Retain the original coordinate convention used by pruned-model LoRAs."""
    table = table.to(device=embeddings.device, dtype=torch.float64)
    design = torch.cat((table, torch.ones_like(table[:, :1])), dim=1)
    solution = torch.linalg.lstsq(design, embeddings.double()).solution
    residual = torch.linalg.vector_norm(design @ solution - embeddings.double()) / torch.linalg.vector_norm(embeddings.double())
    return table.float(), solution[:-1].float(), solution[-1].float(), float(residual)


def fold_projection(weight, bias, a, b, basis, mean, scale=1.0):
    """Fold W + scale*B*A onto an affine time basis without forming B*A."""
    w = weight.float()
    p = w @ basis.T + scale * (b.float() @ (a.float() @ basis.T))
    offset = bias.float() + w @ mean + scale * (b.float() @ (a.float() @ mean))
    return p, offset


def read_conditioning(path, base_table, adapter_path):
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        if meta.get("format") != "mosaichunk-raven-conditioning-v1":
            raise ValueError("Not a converted MosaiChunk RAVEN conditioning file")
        if meta.get("base_table_sha256") != tensor_sha256(base_table):
            raise ValueError("Conditioning file does not match this checkpoint's time table; use the FL2VA base used for conversion")
        # The published adapter is fixed; a header/tensor inventory check alone
        # would also accept a different finetune with identical shapes.
        digest = hashlib.sha256()
        with open(adapter_path, "rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if meta.get("adapter_sha256") != digest.hexdigest():
            raise ValueError("Conditioning was built for a different RAVEN adapter")
        keys = {"adaln_t_table"}
        for prefix in [f"blocks.{i}" for i in range(50)] + ["final_layer"]:
            keys.update(f"{prefix}.adaln_proj.linear.{suffix}" for suffix in ("weight", "bias"))
        if set(f.keys()) != keys:
            raise ValueError("Conditioning file must contain all 51 projections and the time table")
        tensors = {key: f.get_tensor(key) for key in keys}
    rank = tensors["adaln_t_table"].shape[1]
    if tensor_sha256(tensors["adaln_t_table"]) != tensor_sha256(base_table):
        raise ValueError("Converted conditioning must preserve the base time coordinates for existing LoRA compatibility")
    if tensors["adaln_t_table"].shape[0] != 1025:
        raise ValueError("Converted time table must have 1025 rows")
    for prefix in [f"blocks.{i}" for i in range(50)] + ["final_layer"]:
        rows = 10752 if prefix == "final_layer" else 96768
        for suffix, shape in (("weight", (rows, rank)), ("bias", (rows,))):
            x = tensors[f"{prefix}.adaln_proj.linear.{suffix}"]
            if tuple(x.shape) != shape or x.dtype != torch.float32 or not torch.isfinite(x).all():
                raise ValueError(f"Invalid converted projection: {prefix}.{suffix}")
    if not torch.isfinite(tensors["adaln_t_table"]).all():
        raise ValueError("Converted time table contains nonfinite values")
    return tensors


def attach_pruned_raven(model, manifest, row_chunk=None):
    """Attach every non-conditioning target; conditioning is already baked."""
    from comfy.ldm.minimax.model import MLP

    class RavenHookMLP(MLP):
        def forward(self, x):
            # Call fc2 itself so its residual hook cannot be skipped by the
            # fused INT8 linear_input_act fast path. Quantized fc2 is preserved.
            gate, value = self.fc1(x).chunk(2, dim=-1)
            return self.fc2(F.silu(gate).mul_(value))

    root = model.diffusion_model
    for module in root.modules():
        if isinstance(module, MLP):
            module.__class__ = RavenHookMLP
    modules = {p: m for p, m in manifest.modules.items()
               if m.entry.category not in (lora.CATEGORY_ADALN, lora.CATEGORY_TIME)}
    filtered = dataclasses.replace(manifest, modules=modules,
                                  counts=lora.category_counts(m.entry for m in modules.values()))
    weights = lora.load_lora_weights(manifest.source, filtered)
    plan = []
    for path, item in modules.items():
        module = root.get_submodule(path)
        if (module.out_features, module.in_features) != item.entry.weight_shape:
            raise ValueError(f"RAVEN target shape does not match: {path}")
        a, b = weights[path]
        plan.append(runtime_linear.ResidualPlan(path=path, module=module, a=a, b=b,
                    alpha=manifest.alpha, rank=item.rank, strength=1.0, base_key=item.base_key))
    attachment = runtime_linear.attach_residuals(plan, row_chunk=row_chunk, allow_quantized=True)
    return attachment, filtered

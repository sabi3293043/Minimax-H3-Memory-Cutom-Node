"""Transfer the converted RAVEN modulation delta onto a pruned H3 variant.

Reads only the local baseline's tiny time table and 51 projections. Never
loads its backbone, downloads weights or overwrites a generator checkpoint.
"""

import logging
import torch
from safetensors import safe_open

from .pruned import read_conditioning, tensor_sha256

LOG = logging.getLogger(__name__)


def _prefix(file):
    for prefix in ('', 'model.diffusion_model.', 'diffusion_model.'):
        if prefix + 'adaln_t_table' in file.keys():
            return prefix
    raise ValueError("Conditioning baseline must be a pruned H3 checkpoint with an AdaLN time table")


def read_baseline_table(path):
    with safe_open(path, framework='pt', device='cpu') as file:
        return file.get_tensor(_prefix(file) + 'adaln_t_table')


def read_transfer_source(conditioning_path, adapter_path, baseline_path):
    if not baseline_path:
        raise ValueError("Image/reference memory needs the local original FL2VA pruned checkpoint as conditioning_base_unet")
    table = read_baseline_table(baseline_path)
    return read_conditioning(conditioning_path, table, adapter_path), table


def transfer_conditioning(converted, source_table, baseline_path, target, *, max_relative_error=.01):
    source = source_table.detach().cpu().double()
    target_table = target.adaln_t_table.detach().cpu().double()
    if source.shape[0] != target_table.shape[0] or not torch.isfinite(target_table).all():
        raise ValueError("Reference checkpoint has an incompatible AdaLN time grid")
    design = torch.cat((source, torch.ones_like(source[:, :1])), dim=1)
    target_design = torch.cat((target_table, torch.ones_like(target_table[:, :1])), dim=1)
    if torch.linalg.matrix_rank(target_design) != target_design.shape[1]:
        raise ValueError("Target checkpoint time basis is rank deficient")
    if tensor_sha256(source_table) == tensor_sha256(target.adaln_t_table):
        mapping = torch.eye(design.shape[1], dtype=torch.float64)
    else:
        mapping = torch.linalg.lstsq(target_design, design).solution
    residual = target_design @ mapping - design
    error_gram, norm_gram = residual.T @ residual, design.T @ design
    output = {'adaln_t_table': target.adaln_t_table.detach().cpu().clone()}
    worst = 0.
    with safe_open(baseline_path, framework='pt', device='cpu') as file:
        prefix = _prefix(file)
        for name in [f'blocks.{i}.adaln_proj.linear' for i in range(len(target.blocks))] + ['final_layer.adaln_proj.linear']:
            layer = target.get_submodule(name)
            original_w = file.get_tensor(prefix + name + '.weight').double()
            original_b = file.get_tensor(prefix + name + '.bias').double()
            adapted_w = converted[name + '.weight'].double()
            adapted_b = converted[name + '.bias'].double()
            if adapted_w.shape != original_w.shape or adapted_b.shape != original_b.shape:
                raise ValueError(f"Conditioning baseline projection shape mismatch: {name}")
            delta = torch.cat((adapted_w - original_w, (adapted_b - original_b)[:, None]), dim=1)
            num = float(((delta @ error_gram) * delta).sum().clamp_min(0))
            den = float(((delta @ norm_gram) * delta).sum().clamp_min(0))
            error = (num / max(den, 1e-30)) ** .5
            worst = max(worst, error)
            if error > max_relative_error:
                raise ValueError(f"RAVEN update cannot be represented accurately in this checkpoint's time basis: {name}, relative error {error:.3%}")
            mapped = delta @ mapping.T
            if tuple(layer.weight.shape) != (len(mapped), target_table.shape[1]):
                raise ValueError(f"Target modulation shape mismatch: {name}")
            output[name + '.weight'] = (layer.weight.detach().cpu().double() + mapped[:, :-1]).float()
            output[name + '.bias'] = (layer.bias.detach().cpu().double() + mapped[:, -1]).float()
            if not torch.isfinite(output[name + '.weight']).all() or not torch.isfinite(output[name + '.bias']).all():
                raise ValueError(f"Nonfinite transferred modulation: {name}")
    LOG.warning("[MosaiChunk H3] RAVEN update transfer: preserved target time table and modulation; worst update fit error %.6g", worst)
    return output, worst

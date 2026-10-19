"""Experimental Tiny-VAE-selected, current-evaluation appearance attention.

Only target-video rows are changed. No weights, sampler, RoPE, old KV, or audio
are replaced. Matching is heuristic, not a trained identity/occlusion detector.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import logging
import math

import torch
import torch.nn.functional as F

LOG = logging.getLogger(__name__)
MODES = ('observe_only', 'manual_frames', 'retrieved_frames')


def frame_numbers(text):
    if not text.strip():
        return ()
    try:
        result = tuple(sorted(set(int(value.strip()) for value in text.split(','))))
    except ValueError as exc:
        raise ValueError('guidance_target_frames must be comma-separated frame indices') from exc
    if any(value < 0 for value in result):
        raise ValueError('Guidance frame indices must be nonnegative')
    return result


@dataclass(frozen=True)
class GuidanceConfig:
    mode: str = 'observe_only'
    strength: float = 0.10
    layers: str = '8,16,24'
    source_frame: int = 0
    target_frames: str = ''
    max_anchor_tokens: int = 128
    similarity: float = 0.85
    margin: float = 0.02
    end_progress: float = 0.95

    def validate(self):
        if self.mode not in MODES:
            raise ValueError('Unknown native guidance mode')
        if not math.isfinite(self.strength) or not 0 <= self.strength <= 0.3:
            raise ValueError('Guidance strength must be finite and between 0 and 0.3')
        if self.source_frame < 0 or not 1 <= self.max_anchor_tokens <= 512:
            raise ValueError('Invalid source frame or anchor token budget')
        if not 0 <= self.similarity <= 1 or not 0 <= self.margin <= 1 or not 0 < self.end_progress <= 1:
            raise ValueError('Invalid guidance thresholds')
        layers = frame_numbers(self.layers)
        if not layers:
            raise ValueError('Select at least one guidance layer')
        targets = frame_numbers(self.target_frames)
        if self.mode == 'manual_frames' and (not targets or self.source_frame in targets):
            raise ValueError('manual_frames requires target frames distinct from guidance_source_frame')
        return self

    @property
    def enabled(self):
        return self.mode != 'observe_only' and self.strength > 0


def decoded_to_latent(frame, decoded_count, latent_count):
    if not 0 <= frame < decoded_count:
        raise ValueError(f'Guidance frame {frame} is outside the {decoded_count}-frame Tiny VAE preview')
    if decoded_count == latent_count:
        return int(frame)
    # Temporal TAEH3: each five-token group emits 1+4+4+4+4 frames.
    spans = [1 if index % 5 == 0 else 4 for index in range(latent_count)]
    if sum(spans) != decoded_count:
        raise ValueError('Tiny VAE temporal output does not match native H3 frame mapping')
    stop = 0
    for index, span in enumerate(spans):
        stop += span
        if frame < stop:
            return index
    raise ValueError('Unable to map Tiny VAE frame to H3 latent')


class GuidanceSession:
    def __init__(self, config, latent_shape):
        self.config = config.validate()
        if len(latent_shape) != 5 or latent_shape[0] != 1 or latent_shape[1] != 24:
            raise ValueError('Native guidance requires batch-one H3 video latents')
        self.latent_t = int(latent_shape[2])
        self.frame_rows = (int(latent_shape[3]) // 2) * (int(latent_shape[4]) // 2)
        self.layers = frozenset(frame_numbers(config.layers))
        self.pairs = ()
        self.progress = 0.
        self.corrections = self.corrected_rows = self.observations = 0
        self.matched_proposals = 0

    def observe(self, frames, references, *, step, total):
        """Publish frame IDs only. Previous-step tensors never enter attention."""
        self.observations += 1
        self.progress = (step + 1) / max(1, total)
        count = int(frames.shape[0])
        if self.config.mode == 'manual_frames':
            proposals = [(self.config.source_frame, frame) for frame in frame_numbers(self.config.target_frames)]
            # User controls always refer to final-video pixel frames, including
            # when a 2D tiny VAE emits only one observation per latent token.
            count = sum(1 if index % 5 == 0 else 4 for index in range(self.latent_t))
        else:
            # Read-only retrieval is an untrusted proposal; current-call feature
            # similarity and uniqueness must also pass before any row is changed.
            proposals = [(ref.source_frame, ref.query_frame) for ref in references
                         if ref.source_step < step and ref.confidence >= .5
                         and ref.source_frame < ref.query_frame]
        pairs = []
        for source, target in proposals:
            pair = (decoded_to_latent(source, count, self.latent_t),
                    decoded_to_latent(target, count, self.latent_t))
            if pair[0] != pair[1] and pair not in pairs:
                pairs.append(pair)
        # Do not alter a frame that is itself an anchor in the same evaluation.
        sources = {source for source, _ in pairs}
        self.pairs = tuple(pair for pair in pairs if pair[1] not in sources)[:8]
        self.matched_proposals += len(self.pairs)

    def attention(self, backend, q, k, v, heads, *, transformer_options=None, **kwargs):
        options = transformer_options or {}
        # Always execute the original backend once with all original arguments.
        output = backend(q, k, v, heads, transformer_options=transformer_options, **kwargs)
        layout = options.get('minimax_h3_layout')
        if (not self.config.enabled or not self.pairs or self.progress >= self.config.end_progress
                or options.get('block_index') not in self.layers or layout is None):
            return output
        if (kwargs.get('mask') is not None or not kwargs.get('skip_reshape', False)
                or kwargs.get('skip_output_reshape', False)
                or q.ndim != 4 or q.shape[0] != 1 or q.shape != k.shape or k.shape != v.shape):
            raise ValueError('Native guidance requires native H3 unmasked batch-one self-attention')
        # These dimensions also prevent accidental application to the text refiner.
        spans = [entry for entry in layout.segments if entry[2] == 'video']
        if len(spans) != 1:
            raise ValueError('Native guidance needs exactly one target-video segment')
        start, stop, _ = spans[0]
        if stop - start != self.latent_t * self.frame_rows or output.shape != (1, q.shape[2], heads * q.shape[3]):
            raise ValueError('H3 packed layout does not match guidance video geometry')
        out = None
        for source_frame, target_frame in self.pairs:
            source = start + source_frame * self.frame_rows
            target = start + target_frame * self.frame_rows
            anchor_ids = torch.linspace(source, source + self.frame_rows - 1,
                min(self.frame_rows, self.config.max_anchor_tokens), device=q.device).round().long().unique()
            # Values carry content without RoPE. Use a bounded deterministic
            # descriptor slice, not a full N x N attention/similarity matrix.
            anchor_v = v.index_select(2, anchor_ids)
            anchor_features = anchor_v.transpose(1, 2).flatten(2)[0].float()
            stride = max(1, anchor_features.shape[1] // 128)
            anchor_features = F.normalize(anchor_features[:, ::stride][:, :128], dim=-1)
            for offset in range(0, self.frame_rows, 256):
                end = min(offset + 256, self.frame_rows)
                ids = torch.arange(target + offset, target + end, device=q.device)
                features = v.index_select(2, ids).transpose(1, 2).flatten(2)[0].float()
                features = F.normalize(features[:, ::stride][:, :128], dim=-1)
                scores = features @ anchor_features.T
                best = scores.topk(min(2, scores.shape[1]), dim=1).values
                valid = best[:, 0] >= self.config.similarity
                if best.shape[1] == 2:
                    valid &= best[:, 0] - best[:, 1] >= self.config.margin
                match = scores.argmax(dim=1)
                reverse = scores.argmax(dim=0)
                valid &= reverse[match] == torch.arange(len(features), device=q.device)
                if not valid.any():
                    continue
                ids = ids[valid]
                # Source K retains the native H3 RoPE/position; no keys are copied
                # from a different denoising step or assigned invented positions.
                extra_kwargs = dict(kwargs, mask=None, skip_reshape=True, skip_output_reshape=False)
                anchor_output = backend(q.index_select(2, ids), k.index_select(2, anchor_ids),
                    anchor_v, heads, transformer_options=transformer_options, **extra_kwargs)
                normal = output.index_select(1, ids)
                delta = (anchor_output.float() - normal.float()) * self.config.strength
                # Bound each token's residual relative to the normal attention.
                limit = normal.float().norm(dim=-1, keepdim=True) * .1
                delta *= (limit / delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.)
                if not torch.isfinite(delta).all():
                    raise ValueError('Nonfinite native appearance correction')
                if out is None:
                    out = output.clone()
                out[:, ids] = (normal.float() + delta).to(output.dtype)
                self.corrected_rows += int(ids.numel())
                self.corrections += 1
        return output if out is None else out

    def close(self):
        self.pairs = ()


@contextmanager
def install_guidance(guider, session):
    """Execution-local options; existing Sage/attention overrides are composed."""
    from comfy.patcher_extension import copy_nested_dicts
    original = guider.model_options
    options = copy_nested_dicts(original)
    transformer = options.setdefault('transformer_options', {})
    root = guider.model_patcher.get_model_object('diffusion_model')
    if any(index >= len(root.blocks) for index in session.layers):
        raise ValueError('A selected guidance layer is outside this H3 model')
    replacements = transformer.setdefault('patches_replace', {}).setdefault('dit', {})

    def block_adapter(index, previous_block):
        def replacement(args, extra):
            local = dict(args['transformer_options'])
            previous = local.get('optimized_attention_override')
            preferred = getattr(getattr(root.blocks[index].attn, 'comfy_attention', None), 'function', None)

            def override(func, q, k, v, heads, **kwargs):
                def backend(q, k, v, heads, **call_kwargs):
                    if previous is not None:
                        return previous(func, q, k, v, heads, **call_kwargs)
                    return (preferred or func)(q, k, v, heads, **call_kwargs)
                return session.attention(backend, q, k, v, heads, **kwargs)

            local['optimized_attention_override'] = override
            forwarded = dict(args, transformer_options=local)
            if previous_block is not None:
                return previous_block(forwarded, extra)
            return extra['original_block'](forwarded)
        return replacement

    for index in session.layers:
        key = ('double_block', index)
        replacements[key] = block_adapter(index, replacements.get(key))
    binding = options.get('spectrum_h3_binding')
    if binding is not None:
        # Forecasting bypasses attention, so never claim that active correction
        # is compatible with skipped transformer calls. Use an independent run
        # with actual evaluations and no replay/rollback of changed predictions.
        runtime = binding.runtime
        config = replace(runtime.config, force_actual=True, offline_smoothing_replay=False,
            anchor_residual_feedback=False, selective_rollback_correction=False,
            model_aware_mode='off', model_aware_trust_shrinkage=False,
            model_aware_replay_generic_correction=False)
        options['spectrum_h3_binding'] = type(binding)(type(runtime)(config))
        LOG.warning('Native appearance guidance: Spectrum uses actual evaluations; forecasting/replay disabled for this execution')
    guider.model_options = options
    try:
        yield
    finally:
        guider.model_options = original
        session.close()

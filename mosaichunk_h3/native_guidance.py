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
    memory_regions: str = '[]'
    protected_frames: str = ''
    confidence_guidance: bool = False
    inspect_memory: bool = False

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
        from .region_memory import parse_regions
        parse_regions(self.memory_regions)
        if len(frame_numbers(self.protected_frames)) > 8:
            raise ValueError('Protect at most eight source frames')
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
        if latent_shape[2] < 1 or any(size < 2 or size % 2 for size in latent_shape[3:]):
            raise ValueError('Native H3 memory requires positive temporal length and even spatial latent dimensions')
        self.latent_t = int(latent_shape[2])
        self.frame_rows = (int(latent_shape[3]) // 2) * (int(latent_shape[4]) // 2)
        self.layers = frozenset(frame_numbers(config.layers))
        self.pairs = ()
        self.progress = 0.
        self.corrections = self.corrected_rows = self.observations = 0
        self.matched_proposals = 0
        self.frame_h, self.frame_w = int(latent_shape[3]) // 2, int(latent_shape[4]) // 2
        self.region_plans = ()
        self.plan_version = 0
        self.decisions = []
        self.heatmap = {}
        self.evaluated_plan_version = None
        from .region_memory import parse_regions
        self.has_regions = bool(parse_regions(config.memory_regions))

    def set_region_plans(self, plans):
        if tuple(plans) != self.region_plans:
            self.plan_version += 1
        self.region_plans = tuple(plans)
        self.pairs = tuple(dict.fromkeys((p.source, p.target) for p in plans))
        self.matched_proposals += len(plans)

    def clear_plan(self):
        self.pairs = self.region_plans = ()
        self.plan_version += 1

    def diagnostics(self):
        return dict(corrected_rows=self.corrected_rows, corrections=self.corrections,
                    decisions=[dict(d) for d in self.decisions[-24:]],
                    heatmap={key:list(cells) for key,cells in self.heatmap.items()},
                    evaluated_plan_version=self.evaluated_plan_version,
                    next_plan_version=self.plan_version,
                    grid=[16, 16])

    def observe(self, frames, references, *, step, total):
        """Publish frame IDs only. Previous-step tensors never enter attention."""
        self.observations += 1
        if not self.has_regions:
            self.region_plans = ()
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
        from .region_memory import RegionPlan, grid_ids
        plans = self.region_plans or tuple(RegionPlan(s, t, 'whole frame', (0,0,1,1),
            (0,0,1,1), 1., s, t) for s,t in self.pairs)
        self.decisions = []
        self.evaluated_plan_version = self.plan_version
        # Source/target descriptors are computed once per frame/attention call.
        feature_cache = {}
        def frame_features(frame):
            if frame not in feature_cache:
                values = v[:, :, start+frame*self.frame_rows:start+(frame+1)*self.frame_rows]
                values = values.transpose(1,2).flatten(2)[0]
                stride = max(1, values.shape[1] // 128)
                feature_cache[frame] = F.normalize(values[:, ::stride][:,:128].float(), dim=-1)
            return feature_cache[frame]
        # Divide the anchor budget between regions instead of multiplying it.
        per_plan = max(1, self.config.max_anchor_tokens // max(1, len({p.name for p in plans})))
        changed_targets = {}
        for plan in plans:
            source_frame, target_frame = plan.source, plan.target
            source = start + source_frame * self.frame_rows
            target = start + target_frame * self.frame_rows
            local_source = grid_ids(plan.source_box, self.frame_h, self.frame_w, device=q.device,
                                    mask=plan.source_mask)
            local_target = grid_ids(plan.target_box, self.frame_h, self.frame_w, device=q.device,
                                    mask=plan.target_mask)
            if target_frame in changed_targets:
                # Overlapping regions cannot apply or overwrite multiple corrections.
                local_target = local_target[~torch.isin(local_target,changed_targets[target_frame])]
            if not local_source.numel() or not local_target.numel():
                continue
            selected = torch.linspace(0, len(local_source)-1, min(len(local_source),per_plan),
                                      device=q.device).round().long().unique()
            local_source = local_source[selected]
            anchor_ids = source + local_source
            # Values carry content without RoPE. Use a bounded deterministic
            # descriptor slice, not a full N x N attention/similarity matrix.
            anchor_v = v.index_select(2, anchor_ids)
            anchor_features = frame_features(source_frame).index_select(0, local_source)
            target_features = frame_features(target_frame)
            changed, tested = 0, 0
            rejected = dict(low_similarity=0, ambiguous=0, non_mutual=0)
            for local_ids in local_target.split(256):
                ids = target + local_ids
                features = target_features.index_select(0, local_ids)
                scores = features @ anchor_features.T
                best = scores.topk(min(2, scores.shape[1]), dim=1).values
                valid = best[:, 0] >= self.config.similarity
                if self.config.inspect_memory:
                    rejected['low_similarity'] += int((~valid).sum().item())
                if best.shape[1] == 2:
                    unique = best[:, 0] - best[:, 1] >= self.config.margin
                    if self.config.inspect_memory:
                        rejected['ambiguous'] += int((valid & ~unique).sum().item())
                    valid &= unique
                match = scores.argmax(dim=1)
                reverse = scores.argmax(dim=0)
                mutual = reverse[match] == torch.arange(len(features), device=q.device)
                if self.config.inspect_memory:
                    rejected['non_mutual'] += int((valid & ~mutual).sum().item())
                valid &= mutual
                tested += len(features)
                if not valid.any():
                    continue
                ids = ids[valid]
                # Source K retains the native H3 RoPE/position; no keys are copied
                # from a different denoising step or assigned invented positions.
                extra_kwargs = dict(kwargs, mask=None, skip_reshape=True, skip_output_reshape=False)
                anchor_output = backend(q.index_select(2, ids), k.index_select(2, anchor_ids),
                    anchor_v, heads, transformer_options=transformer_options, **extra_kwargs)
                normal = output.index_select(1, ids)
                weight = self.config.strength
                if self.config.confidence_guidance:
                    # Confidence only attenuates; it cannot amplify the configured strength.
                    confidence = ((best[valid,0]-self.config.similarity) /
                                  max(1e-6,1-self.config.similarity)).clamp(0,1)
                    weight = weight * confidence[None,:,None] * plan.confidence
                delta = (anchor_output.float() - normal.float()) * weight
                # Bound each token's residual relative to the normal attention.
                limit = normal.float().norm(dim=-1, keepdim=True) * .1
                delta *= (limit / delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.)
                if not torch.isfinite(delta).all():
                    raise ValueError('Nonfinite native appearance correction')
                affected = delta.norm(dim=-1)[0] > 0
                ids, delta, normal = ids[affected], delta[:,affected], normal[:,affected]
                if not ids.numel():
                    continue
                prior_ids = changed_targets.get(target_frame)
                changed_targets[target_frame] = (ids-target if prior_ids is None else
                    torch.cat((prior_ids,ids-target)))
                if out is None:
                    out = output.clone()
                out[:, ids] = (normal.float() + delta).to(output.dtype)
                self.corrected_rows += int(ids.numel())
                self.corrections += 1
                changed += int(ids.numel())
                if self.config.inspect_memory:
                    cells = ((ids-target) // self.frame_w * 16 // self.frame_h * 16
                             + (ids-target) % self.frame_w * 16 // self.frame_w).unique().cpu().tolist()
                    key = str(target_frame)
                    heat = set(self.heatmap.get(key, []))
                    self.heatmap[key] = sorted(heat.union(cells))[:256]
                    while len(self.heatmap) > 32:
                        self.heatmap.pop(next(iter(self.heatmap)))
            if self.config.inspect_memory:
                self.decisions.append(dict(region=plan.name, source=plan.source_frame,
                    target=plan.target_frame, tested=tested, corrected=changed,
                    reason='matched' if changed else 'no confident unique feature matches',
                    confidence=round(plan.confidence,3),rejected=rejected))
        return output if out is None else out

    def close(self):
        self.clear_plan()
        self.heatmap.clear()


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

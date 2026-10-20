"""Execution-local H3 acceleration. Approximate; never skips guidance blocks.

Sol uses the installed ComfyUI producer, not a copied kernel. The independent
cache stores a section's residual update, not old hidden states. Probe blocks,
guidance blocks and the first/last blocks always execute. No disk cache/weights.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import logging
import math

import torch

from .host_residual import HostResidual, HostAllocationError
from .adaptive_sol import AdaptiveSol

LOG = logging.getLogger(__name__)
MODES = ('off', 'sol', 'selective_cache', 'sol_and_cache')


@dataclass(frozen=True)
class AccelerationConfig:
    mode: str = 'off'
    sol_tau: float = 1.0
    start: float = .25
    end: float = .85
    cache_threshold: float = .05
    cache_max_reuse: int = 1
    cache_gib: float = .5
    cache_storage: str = 'gpu'
    cache_ram_gib: float = 2.
    transfer_mib: int = 16
    adaptive_sol: bool = False
    sol_stable_boost: float = .15
    sol_drift_threshold: float = .04

    def validate(self):
        if self.mode not in MODES:
            raise ValueError('Unknown generation_acceleration mode')
        if not all(math.isfinite(v) for v in (self.sol_tau, self.start, self.end,
                                              self.cache_threshold, self.cache_gib, self.cache_ram_gib,
                                              self.sol_stable_boost, self.sol_drift_threshold)):
            raise ValueError('Acceleration settings must be finite')
        if not 0 <= self.start < self.end <= 1 or not 0 <= self.sol_tau <= 2:
            raise ValueError('Invalid acceleration window or Sol threshold')
        if not 0 < self.cache_threshold <= .3 or not 1 <= self.cache_max_reuse <= 3:
            raise ValueError('Invalid residual cache threshold or reuse cap')
        if not 0 < self.cache_gib <= 8:
            raise ValueError('cache_vram_gib must be positive and at most 8')
        if self.cache_storage not in ('gpu', 'ram', 'auto') or not 0 < self.cache_ram_gib <= 16:
            raise ValueError('Invalid cache_storage or cache_ram_gib')
        if not 1 <= self.transfer_mib <= 64:
            raise ValueError('cache_transfer_mib must be in [1,64]')
        if not 0 <= self.sol_stable_boost <= .3 or not 0 < self.sol_drift_threshold <= 1:
            raise ValueError('Invalid adaptive Sol controls')
        return self

    @property
    def sol(self):
        return self.mode in ('sol', 'sol_and_cache')

    @property
    def cache(self):
        return self.mode in ('selective_cache', 'sol_and_cache')


def cache_sections(count, protected):
    """One real probe plus at least two reusable blocks per section."""
    runs, run = [], []
    for index in range(count):
        if index in protected:
            if len(run) >= 3:
                runs.append(tuple(run))
            run = []
        else:
            run.append(index)
    if len(run) >= 3:
        runs.append(tuple(run))
    return tuple(runs)


class Section:
    def __init__(self, indices):
        self.indices = indices
        self.reset()

    def reset(self):
        self.residual = self.probe = None
        self.capture = self.hit = False
        self.reuses = 0
        self.error = 0.


def probe_features(before, after, layout):
    """Bounded features grouped by conditioning stream and each video frame.

    Every video latent frame gets a vote; a quiet background cannot hide a
    moving frame or changing audio. This remains a sampled approximation.
    """
    if before.ndim != 2 or after.shape != before.shape:
        raise ValueError('Residual caching requires native packed H3 activations')
    n, width = before.shape
    if layout.seq_len != n:
        raise ValueError('Residual cache packed layout mismatch')
    frames = int(layout.signature[1])
    if not 1 <= frames <= 256:
        raise ValueError('Residual cache supports at most 256 latent frames')
    groups = []
    for a, b, kind in layout.segments:
        if not 0 <= a <= b <= n:
            raise ValueError('Invalid H3 segment')
        if a == b:
            continue
        if kind == 'video':
            if (b - a) % frames:
                raise ValueError('Video rows do not divide into latent frames')
            span = (b - a) // frames
            groups.extend((a + i * span, a + (i + 1) * span) for i in range(frames))
        else:
            groups.append((a, b))
    if not groups:
        raise ValueError('H3 cache needs a packed layout')
    # Fixed size per group permits one GPU reduction for the complete probe.
    ids = torch.tensor([[a + (b - a - 1) * j // 7 for j in range(8)]
                        for a, b in groups], device=before.device)
    stride = max(1, math.ceil(width / 128))
    return (after[ids, ::stride].float() - before[ids, ::stride].float()).detach()


class AccelerationSession:
    def __init__(self, config, root, sampling, guidance=None):
        self.config = config.validate()
        self.root = root
        self.guidance = guidance
        self.protected = {0, len(root.blocks) - 1}
        if guidance is not None:
            self.protected.update(guidance.layers)
        self.sections = [Section(run) for run in cache_sections(len(root.blocks), self.protected)]
        self.start_sigma = float(sampling.percent_to_sigma(config.start))
        self.end_sigma = float(sampling.percent_to_sigma(config.end))
        self.previous_sigma = self.signature = None
        self.active = False
        self.allowed = set()
        self.bytes = self.peak_bytes = 0
        self.ram_bytes = self.peak_ram_bytes = self.transfer_bytes = 0
        self.backends = {}
        self.real_blocks = self.reused_blocks = self.cache_hits = self.evaluations = 0
        self.sparse_calls = self.dense_calls = self.invalidations = 0
        self.reasons = set()
        self.disabled_sections = set()
        self.sol_module = self.sol_patch = None
        self.closed = False
        self.compiler_bypass_calls = 0
        self.adaptive = AdaptiveSol(config.sol_tau, config.sol_stable_boost,
                                    config.sol_drift_threshold, config.end)
        self.tau = self.tau_min = self.tau_max = config.sol_tau
        self.tau_changes = 0

    def reason(self, text):
        if text not in self.reasons:
            self.reasons.add(text)
            LOG.warning('[MosaiChunk H3 acceleration] %s', text)

    def clear_cache(self):
        for section in self.sections:
            self.drop(section)
        self.bytes = self.ram_bytes = 0

    def observe_prediction(self, video, *, progress, fingerprint=None):
        if not self.config.sol or not self.config.adaptive_sol:
            return
        from .observation_policy import latent_fingerprint
        try:
            self.adaptive.observe(fingerprint if fingerprint is not None else latent_fingerprint(video), progress)
        except (ValueError, MemoryError, torch.OutOfMemoryError) as exc:
            self.adaptive.observe(None, progress)
            self.reason(f'Adaptive Sol using conservative fallback: {type(exc).__name__}')

    def advance(self, progress):
        self.adaptive.progress = progress

    def prediction_unavailable(self):
        if self.config.adaptive_sol:
            self.adaptive.observe(None, self.adaptive.progress)

    def available_gpu(self, x):
        budget = int(self.config.cache_gib * 1024**3)
        if x.device.type != 'cuda':
            return budget, budget
        free, _ = torch.cuda.mem_get_info(x.device)
        # RAM storage needs small staging, rather than a full resident residual.
        return (min(budget, max(0, free + self.bytes - 1024**3)),
                max(0, free + self.bytes - 256 * 1024**2))

    def available_ram(self):
        budget = int(self.config.cache_ram_gib * 1024**3)
        try:
            import psutil
        except ImportError:
            return budget
        return min(budget, max(0, psutil.virtual_memory().available + self.ram_bytes - 512 * 1024**2))

    def begin(self, args):
        self.evaluations += 1
        self.active = False
        if self.config.sol and self.config.adaptive_sol:
            tau = self.adaptive.tau()
            if tau != self.tau:
                self.tau_changes += 1
                self.tau = tau
                self.tau_min, self.tau_max = min(self.tau_min, tau), max(self.tau_max, tau)
            if self.sol_patch is not None:
                self.sol_patch.tau = tau
        for section in self.sections:
            section.hit = section.capture = False
        if not self.config.cache:
            return
        opts = args['transformer_options']
        sigmas = opts.get('sigmas')
        layout = opts.get('minimax_h3_layout')
        x = args['img']
        if sigmas is None or sigmas.numel() != 1 or layout is None or x.ndim != 2:
            self.clear_cache()
            self.reason('Cache inactive: requires a single sigma and native H3 packed layout')
            return
        sigma = float(sigmas.reshape(-1)[0])
        if not math.isfinite(sigma):
            self.clear_cache()
            return
        plan = ((tuple(self.guidance.pairs), getattr(self.guidance,'plan_version',0),
                 self.guidance.progress >= self.guidance.config.end_progress)
                if self.guidance is not None else None)
        signature = (tuple(x.shape), x.dtype, x.device, tuple(layout.signature),
                     tuple(layout.segments), tuple(opts.get('uuids', ())), self.tau, plan)
        changed = signature != self.signature
        repeated_or_reversed = self.previous_sigma is not None and sigma >= self.previous_sigma
        if changed or repeated_or_reversed:
            self.clear_cache()
            self.invalidations += 1
        self.signature = signature
        self.previous_sigma = sigma
        self.active = self.end_sigma <= sigma <= self.start_sigma
        if not self.active:
            self.clear_cache()
        available, staging_available = self.available_gpu(x)
        ram_available = self.available_ram() if self.config.cache_storage != 'gpu' else 0
        groups = max(0, len(layout.segments) - 1) + int(layout.signature[1])
        feature_width = math.ceil(x.shape[1] / max(1, math.ceil(x.shape[1] / 128)))
        descriptor = groups * 8 * feature_width * 4
        cost = x.numel() * x.element_size() + descriptor
        transfer = self.config.transfer_mib * 1024**2
        host_cost = HostResidual.storage_bytes(x.shape, x.element_size(), transfer)
        staging = min(x.numel() * x.element_size(), transfer) if x.device.type == 'cuda' else 0
        self.allowed = set()
        self.backends = {}
        storage_available = int(self.config.cache_gib * 1024**3)
        # Prefer longer sections for the largest saved compute per cached byte.
        for section in sorted(self.sections, key=lambda s: -len(s.indices)):
            if section.indices[0] in self.disabled_sections:
                continue
            if self.config.cache_storage != 'ram' and cost <= min(available, storage_available):
                self.allowed.add(section.indices[0])
                self.backends[section.indices[0]] = 'gpu'
                available -= cost
                storage_available -= cost
                staging_available -= cost
            elif (self.config.cache_storage != 'gpu' and host_cost <= ram_available
                  and descriptor + staging <= staging_available
                  and descriptor <= storage_available):
                self.allowed.add(section.indices[0])
                self.backends[section.indices[0]] = 'ram'
                ram_available -= host_cost
                staging_available -= descriptor
                storage_available -= descriptor
                available = max(0, available - descriptor)
        # reserve_bytes checks the actual descriptor size again before capture.
        for section in self.sections:
            desired = self.backends.get(section.indices[0])
            if (desired is None or (section.residual is not None and
                    isinstance(section.residual, HostResidual) != (desired == 'ram'))):
                self.drop(section)
        if self.active and not self.allowed:
            self.reason('Cache inactive: no eligible sections' if not self.sections else
                        'Cache inactive: storage budget/headroom cannot hold one section residual')

    def drop(self, section):
        if section.residual is not None:
            if isinstance(section.residual, HostResidual):
                self.ram_bytes -= section.residual.nbytes
                self.transfer_bytes += section.residual.transferred
            else:
                self.bytes -= section.residual.numel() * section.residual.element_size()
        if section.probe is not None:
            self.bytes -= section.probe.numel() * section.probe.element_size()
        section.reset()

    def reserve_bytes(self, amount):
        return self.bytes + amount <= int(self.config.cache_gib * 1024**3)

    def head(self, section, args, execute):
        result = execute(args)
        if not self.active or section.indices[0] not in self.allowed:
            return result
        backend = self.backends[section.indices[0]]
        if section.residual is not None and (isinstance(section.residual, HostResidual) != (backend == 'ram')):
            self.drop(section)
        current = None
        try:
            current = probe_features(args['img'], result['img'], args['transformer_options']['minimax_h3_layout'])
            if not bool(torch.isfinite(current).all()):
                self.drop(section)
                self.reason('Cache refresh: nonfinite probe; no residual reused')
                return result
            difference = math.inf
            if section.probe is not None and current.shape == section.probe.shape:
                change = (current - section.probe).abs().mean(dim=(1, 2))
                norm = section.probe.abs().mean(dim=(1, 2)).clamp_min(1e-6)
                difference = float((change / norm).max())
            section.error += difference
            finite = math.isfinite(difference)
            if (finite and section.residual is not None and section.reuses < self.config.cache_max_reuse
                    and section.error <= self.config.cache_threshold):
                section.hit = True
                section.reuses += 1
            else:
                self.drop(section)
                amount = current.numel() * current.element_size()
                if backend == 'gpu':
                    amount += result['img'].numel() * result['img'].element_size()
                if not self.reserve_bytes(amount):
                    self.reason('Cache section skipped: explicit cache_vram_gib limit')
                    return result
                if backend == 'ram':
                    host_bytes = HostResidual.storage_bytes(result['img'].shape, result['img'].element_size(),
                                                            self.config.transfer_mib * 1024**2)
                    if self.ram_bytes + host_bytes > int(self.config.cache_ram_gib * 1024**3):
                        self.reason('Cache section skipped: explicit cache_ram_gib limit')
                        return result
                    section.residual = HostResidual(result['img'], self.config.transfer_mib * 1024**2)
                    self.ram_bytes += section.residual.nbytes
                    self.peak_ram_bytes = max(self.peak_ram_bytes, self.ram_bytes)
                else:
                    section.residual = result['img'].detach().clone()
                    self.bytes += section.residual.numel() * section.residual.element_size()
                section.capture = True
                section.error = 0.
            if section.probe is not None:
                self.bytes -= section.probe.numel() * section.probe.element_size()
            section.probe = current
            self.bytes += current.numel() * current.element_size()
            self.peak_bytes = max(self.peak_bytes, self.bytes)
        except (ValueError, torch.OutOfMemoryError, HostAllocationError) as exc:
            self.drop(section)
            self.disabled_sections.add(section.indices[0])
            self.reason(f'Cache section disabled: {type(exc).__name__}: {exc}')
        return result

    def block(self, index, args, execute):
        if index == 0:
            self.begin(args)
        if self.config.cache:
            for section in self.sections:
                if index == section.indices[0]:
                    self.real_blocks += 1
                    return self.head(section, args, execute)
                if index in section.indices[1:]:
                    if section.hit:
                        # Apply the cached update exactly once to CURRENT features.
                        if index == section.indices[1]:
                            replay_failed = False
                            try:
                                result = (section.residual.add(args['img']) if isinstance(section.residual, HostResidual)
                                          else args['img'] + section.residual)
                            except torch.OutOfMemoryError:
                                self.drop(section)
                                self.disabled_sections.add(section.indices[0])
                                self.reason('Cache section disabled: replay allocation OOM; computing real blocks')
                                replay_failed = True
                            # Leave the exception handler before evaluating the
                            # model: its traceback may retain failed CUDA buffers.
                            if replay_failed:
                                self.real_blocks += 1
                                return execute(args)
                            self.cache_hits += 1
                            self.reused_blocks += 1
                            return {'img': result}
                        self.reused_blocks += 1
                        return {'img': args['img']}
                    self.real_blocks += 1
                    result = execute(args)
                    if index == section.indices[-1] and section.capture:
                        # Reuse the reserved buffer: no second full residual cache.
                        if isinstance(section.residual, HostResidual):
                            section.residual.finish(result['img'])
                        else:
                            section.residual.neg_().add_(result['img'])
                        section.capture = False
                    return result
        self.real_blocks += 1
        return execute(args)

    def metrics(self):
        return dict(mode=self.config.mode, sparse_calls=self.sparse_calls, dense_calls=self.dense_calls,
                    real_blocks=self.real_blocks, reused_blocks=self.reused_blocks,
                    cache_hits=self.cache_hits, cache_peak_bytes=self.peak_bytes,
                    cache_peak_ram_bytes=self.peak_ram_bytes, cache_ram_bytes=self.ram_bytes,
                    cache_transfer_bytes=self.transfer_bytes + sum(s.residual.transferred for s in self.sections
                        if isinstance(s.residual, HostResidual)), cache_storage=self.config.cache_storage,
                    sol_tau=self.tau, sol_tau_min=self.tau_min, sol_tau_max=self.tau_max,
                    adaptive_sol_changes=self.tau_changes, adaptive_sol_state=self.adaptive.state,
                    allocation_compiler_bypass_calls=self.compiler_bypass_calls,
                    invalidations=self.invalidations, reasons=sorted(self.reasons))

    def summary(self):
        text = (f'acceleration {self.config.mode}: {self.sparse_calls} sparse calls, '
                f'{self.reused_blocks} reused blocks, {self.cache_hits} section hits, '
                f'cache peak {self.peak_bytes / 1024**2:.1f} MiB')
        if self.config.cache_storage != 'gpu' and self.config.cache:
            text += f' GPU / {self.peak_ram_bytes / 1024**2:.1f} MiB RAM'
        if self.config.sol and self.config.adaptive_sol:
            text += f'; adaptive Sol tau {self.tau:.2f} ({self.adaptive.state})'
        if self.compiler_bypass_calls:
            text += f'; allocation recording bypassed for {self.compiler_bypass_calls} cache evaluations'
        if self.reasons:
            text += '; ' + '; '.join(sorted(self.reasons))
        elif self.config.sol and not self.sparse_calls:
            text += '; Sol inactive so far (dense fallback/window)'
        return text

    def close(self):
        self.clear_cache()
        self.adaptive.close()
        if self.sol_patch is not None:
            self.sol_patch.reset()
        self.closed = True


@contextmanager
def install_acceleration(guider, config, guidance=None):
    """Compose options inside the native sample; restore them on every exit."""
    from comfy.patcher_extension import copy_nested_dicts
    config.validate()
    if config.mode == 'off':
        yield None
        return
    root = guider.model_patcher.get_model_object('diffusion_model')
    if not getattr(root, 'blocks', None):
        raise ValueError('Native acceleration requires MiniMax H3 transformer blocks')
    original = guider.model_options
    if original.get('spectrum_h3_binding') is not None:
        raise ValueError('Turn Spectrum off when using independent generation_acceleration')
    options = copy_nested_dicts(original)
    transformer = options.setdefault('transformer_options', {})
    if 'easycache' in transformer:
        raise ValueError('Turn other generation caches off before using native acceleration')
    replacements = transformer.setdefault('patches_replace', {}).setdefault('dit', {})
    if config.cache and any(key and key[0] == 'double_block' for key in replacements if isinstance(key, tuple)):
        raise ValueError('Selective cache cannot skip existing double_block patches; remove the other block patch')
    sampling = guider.model_patcher.get_model_object('model_sampling')
    session = AccelerationSession(config, root, sampling, guidance)
    if guidance is not None and any(('double_block', index) in replacements for index in guidance.layers):
        raise ValueError('Existing block patch on a memory layer may bypass guidance; remove that block patch before acceleration')
    try:
        if config.cache:
            from .cache_compiler import CacheCompilerBoundary
            groups = transformer.setdefault('wrappers', {})
            previous = groups.get('diffusion_model', {})
            groups['diffusion_model'] = {
                'mosaichunk_cache_allocation_boundary': [CacheCompilerBoundary(session)], **previous}
        if config.sol:
            try:
                module = importlib.import_module('comfy_extras.nodes_sparse_attention')
                for name in ('SparseAttnPatch', 'h3_eligible', 'h3_sparse_attention'):
                    if not callable(getattr(module, name, None)):
                        raise AttributeError(f'missing sparse-attention API: {name}')
                session.sol_module = module
                dense = set(session.protected)
                if config.cache:
                    dense.update(section.indices[0] for section in session.sections)
                session.sol_patch = module.SparseAttnPatch(
                    tau=config.sol_tau, topk_ratio=0., vsa=False,
                    sigma_start=session.start_sigma, sigma_end=session.end_sigma,
                    min_tokens=12288, dense_blocks=dense,
                    sink_conditioning='exact_kv_and_rows', extra_tokens=256, verbose=False)
            except (ImportError, AttributeError, TypeError) as exc:
                session.reason(f'Sol unavailable in this ComfyUI/kernel version: {type(exc).__name__}: {exc}')

        def adapter(index, previous):
            def replacement(args, extra):
                def execute(forwarded):
                    if session.sol_patch is not None and index not in session.sol_patch.dense_blocks:
                        module = session.sol_module
                        block = root.blocks[index]
                        if previous is not None:
                            session.reason(f'Sol kept block {index} dense: existing block patch')
                        elif (forwarded['transformer_options'].get('sigmas') is not None and
                              module.h3_eligible(block.attn, forwarded['img'], forwarded.get('rope_freqs'),
                                                forwarded['transformer_options'], session.sol_patch, index)):
                            def attention(h, rope_freqs=None, transformer_options=None):
                                return module.h3_sparse_attention(block.attn, h, rope_freqs,
                                    transformer_options or {}, session.sol_patch, index)
                            forwarded = dict(forwarded, attention=attention)
                            session.sparse_calls += 1
                        else:
                            session.dense_calls += 1
                    return previous(forwarded, extra) if previous is not None else extra['original_block'](forwarded)
                return session.block(index, args, execute)
            return replacement

        for index in range(len(root.blocks)):
            key = ('double_block', index)
            replacements[key] = adapter(index, replacements.get(key))
        guider.model_options = options
        yield session
    finally:
        guider.model_options = original
        if config.sol and session.sparse_calls == 0:
            session.reason('Sol used no sparse calls: check kernel/GPU eligibility, token length and activation window')
        session.close()
        LOG.warning('[MosaiChunk H3] %s', session.summary())

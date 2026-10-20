"""Execution-local scheduling and spatial sizing for Tiny VAE observation only."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ObservationSettings:
    schedule: str = 'fixed'
    progressive_resolution: bool = False
    start_scale: float = .5
    full_progress: float = .75
    change_threshold: float = .04
    max_interval: int = 3

    def validate(self, *, min_progress, interval):
        if self.schedule not in ('fixed', 'adaptive'):
            raise ValueError('Unknown observation_schedule')
        if not .25 <= self.start_scale <= 1. or not 0 < self.full_progress <= 1:
            raise ValueError('Invalid progressive observation scale/progress')
        if self.progressive_resolution and self.full_progress <= min_progress:
            raise ValueError('observation_full_progress must exceed min_progress')
        if not math.isfinite(self.change_threshold) or not 0 < self.change_threshold <= 1:
            raise ValueError('adaptive_change_threshold must be finite and in (0,1]')
        if not 1 <= self.max_interval <= 100:
            raise ValueError('adaptive_max_interval must be between 1 and 100')
        if self.schedule == 'adaptive' and self.max_interval < interval:
            raise ValueError('adaptive_max_interval must be at least observation_interval')
        return self


@dataclass
class ObservationDecision:
    observe: bool
    size: tuple
    reason: str
    fingerprint: object = None
    change: float | None = None


def latent_fingerprint(video):
    """Bounded CPU thumbnails from original latents, not decoded RGB or old KV.

    Up to 64 temporal samples; casts are bounded to four samples at a time.
    This is a heuristic for prediction drift, not an occlusion detector.
    """
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        ids = torch.linspace(0, video.shape[2] - 1, min(64, video.shape[2]),
                             device=video.device).round().long().unique()
        parts = []
        for group in ids.split(4):
            clip = video.detach().index_select(2, group)
            flat = clip.movedim(2, 1).reshape(-1, video.shape[1], *video.shape[-2:])
            parts.append(F.adaptive_avg_pool2d(flat.float(), (4, 4)).flatten(1).cpu())
        result = torch.cat(parts)
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite observation fingerprint')
        return result


def resize_observation(video, size):
    """Spatial-only area downsample. Original sampler tensors remain untouched."""
    import torch.nn.functional as F
    if tuple(video.shape[-2:]) == tuple(size):
        return video
    b, c, t, h, w = video.shape
    if not (0 < size[0] <= h and 0 < size[1] <= w):
        raise ValueError('Observation resizing must not upscale latents')
    # Work one temporal sample at a time to bound temporary full-frame casts.
    frames = [F.interpolate(video[:, :, i].detach().float(), size=size, mode='area')
              .to(dtype=video.dtype) for i in range(t)]
    import torch
    return torch.stack(frames, dim=2)


def decode_observation_views(decoder, video, *, size, reference_sizes=()):
    """Current-detail preview plus same-scale queries for retained history.

    RGB resized after decoding cannot calibrate different decoder input sizes.
    Decode each retained size once, with the current preview first. Progressive
    mode has at most three sizes. No historical latents or GPU KV are retained.
    """
    sizes = tuple(dict.fromkeys((tuple(size), *(tuple(s) for s in reference_sizes))))
    views = {}
    for dimensions in sizes:
        frames = decoder.decode_video(resize_observation(video[:1], dimensions), frame_indices=None)
        if len(sizes) > 1:
            # A temporal decoder may return GPU RGB; finish its host transfer
            # before decoding another scale rather than stacking GPU previews.
            frames = frames.cpu()
        if views and frames.shape[:2] != next(iter(views.values())).shape[:2]:
            raise ValueError('Progressive previews and reference views must share temporal/channel layout')
        views[dimensions] = frames
    return views


class ObservationPolicy:
    def __init__(self, settings, *, min_progress, interval):
        self.settings = settings.validate(min_progress=min_progress, interval=interval)
        self.min_progress = min_progress
        self.interval = interval
        self.last_step = None
        self.last_size = None
        self.last_fingerprint = None
        self.stage_observations = 0

    def dimensions(self, shape, progress):
        scale = 1.
        if self.settings.progressive_resolution:
            phase = (progress - self.min_progress) / (self.settings.full_progress - self.min_progress)
            if phase < .5:
                scale = self.settings.start_scale
            elif phase < 1:
                scale = (self.settings.start_scale + 1.) / 2
        return tuple(min(int(d), max(min(4, int(d)), int(round(d * scale)))) for d in shape[-2:])

    def decide(self, video, *, step, total):
        progress = (step + 1) / max(1, total)
        size = self.dimensions(video.shape, progress)
        first = max(0, math.ceil(self.min_progress * total) - 1)
        if self.settings.schedule == 'fixed':
            return ObservationDecision((step - first) % self.interval == 0, size, 'fixed interval')
        gap = step - self.last_step if self.last_step is not None else None
        changed_size = size != self.last_size
        if not changed_size and gap is not None and gap < self.interval:
            return ObservationDecision(False, size, 'minimum interval')
        fingerprint = latent_fingerprint(video)
        change = None
        if self.last_fingerprint is not None and self.last_fingerprint.shape == fingerprint.shape:
            delta = (fingerprint - self.last_fingerprint).square().mean(dim=1).sqrt()
            norm = self.last_fingerprint.square().mean(dim=1).sqrt().clamp_min(.1)
            change = float((delta / norm).max())
        if changed_size:
            reason = 'resolution refresh'
        elif self.stage_observations < 3:
            reason = 'retrieval warmup'
        elif gap >= self.settings.max_interval:
            reason = 'maximum interval'
        elif change is None or change >= self.settings.change_threshold:
            reason = 'prediction changed'
        else:
            return ObservationDecision(False, size, 'prediction stable', fingerprint, change)
        return ObservationDecision(True, size, reason, fingerprint, change)

    def observed(self, decision, *, step):
        if decision.size != self.last_size:
            self.stage_observations = 0
        self.last_size = decision.size
        self.last_step = step
        self.last_fingerprint = decision.fingerprint
        self.stage_observations += 1

"""Bounded adaptive RGB observation index for native H3 previews.

Not the trained MosaiChunk RAVEN KV router: this ranks preview frames and
offers approximate nearest-reference lookup. It NEVER guides the sampler.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StoredObservation:
    step: int
    frame: int
    confidence: float
    descriptor: Any
    record: Any


class AdaptiveVisualIndex:
    """Bounded image index using small CPU thumbnails and cross-step stability.

    Thumbnail similarity cannot reliably identify characters across poses.
    These are intentionally conservative exploratory retrieval heuristics.
    """

    def __init__(self, *, capacity: int = 24, captures_per_step: int = 4,
                 match_threshold: float = 0.025):
        if capacity < 1 or captures_per_step < 1:
            raise ValueError("capacity and captures_per_step must be positive")
        if not (0 < match_threshold < 1):
            raise ValueError("match_threshold must be between zero and one")
        self.capacity = int(capacity)
        self.captures_per_step = int(captures_per_step)
        self.match_threshold = float(match_threshold)
        self.entries: list[StoredObservation] = []
        self._previous = None
        self._previous_step = None

    @staticmethod
    def descriptors(frames):
        """[T,3,H,W] uint8 or float [0,1] -> [T,108] CPU thumbnail features."""
        import torch
        import torch.nn.functional as F
        if frames.ndim != 4 or frames.shape[0] == 0 or frames.shape[1] != 3:
            raise ValueError("decoded video must be nonempty [T,3,H,W]")
        with torch.no_grad():
            parts = []
            # Never allocate a full RGB video as fp32 just to extract features.
            for clip in frames.detach().split(2, dim=0):
                thumb = F.adaptive_avg_pool2d(clip.float(), (6, 6))
                if frames.dtype == torch.uint8:
                    thumb = thumb / 255.0
                parts.append(thumb.flatten(1).clamp(0, 1).to(device="cpu"))
            return torch.cat(parts, dim=0)

    def candidates(self, frames, *, step: int):
        """Choose frame IDs by content changes and similarity, not intervals.

        Returns (frame index, CPU descriptor, cross-step stability).
        """
        import torch
        if step < 0:
            raise ValueError("step must be nonnegative")
        x = self.descriptors(frames)
        n = int(x.shape[0])
        transition = torch.zeros(n)
        if n > 1:
            change = (x[1:] - x[:-1]).abs().mean(dim=1)
            transition[1:] = torch.maximum(transition[1:], change)
            transition[:-1] = torch.maximum(transition[:-1], change)
        variation = x.view(n, 3, 36).std(dim=2).mean(dim=1)
        if self._previous is not None and self._previous.shape == x.shape:
            drift = (x - self._previous).abs().mean(dim=1)
            stable = (1.0 - drift / self.match_threshold).clamp(0.0, 1.0)
        else:
            stable = torch.zeros(n)
        self._previous = x.detach().clone()
        self._previous_step = int(step)

        pool = [e.descriptor for e in self.entries]
        chosen = []
        for _ in range(min(self.captures_per_step, n)):
            scores = 0.25 * transition + 0.15 * variation + 0.1 * stable
            if pool:
                refs = torch.stack(pool, 0)
                nearest = (x[:, None] - refs[None]).abs().mean(dim=2).min(dim=1).values
                scores = scores + nearest
            else:
                scores = scores + 0.15 * torch.ones(n)
            if chosen:
                refs = x[torch.tensor(chosen)]
                diversity = (x[:, None] - refs[None]).abs().mean(dim=2).min(dim=1).values
                scores = scores + diversity
                scores[chosen] = -float("inf")
            best = int(scores.argmax())
            if pool and float(scores[best]) < 0.12:
                break
            chosen.append(best)
        return [(i, x[i].clone(), float(0.25 + 0.75 * stable[i])) for i in chosen]

    def put(self, step, frame, descriptor, confidence, record, *, on_evict):
        """Retain at most capacity images and ask caller to evict their disk files."""
        import torch
        if not isinstance(descriptor, torch.Tensor) or descriptor.ndim != 1:
            raise TypeError("descriptor must be a 1D tensor")
        entry = StoredObservation(int(step), int(frame), float(confidence),
                                  descriptor.detach().float().cpu().clone(), record)
        prior = next((e for e in self.entries if e.frame == entry.frame), None)
        if prior is not None:
            if entry.confidence < prior.confidence:
                return False
            on_evict(prior.record)
            self.entries.remove(prior)
        elif len(self.entries) >= self.capacity:
            weakest = min(self.entries, key=lambda e: (e.confidence, e.step))
            if entry.confidence < weakest.confidence:
                return False
            on_evict(weakest.record)
            self.entries.remove(weakest)
        self.entries.append(entry)
        return True

    def retrieve(self, query, *, count=4, before_frame=None):
        """Retrieve nearby thumbnail images, NOT verified character identities."""
        q = query.detach().float().cpu().flatten()
        matches = [e for e in self.entries if before_frame is None or e.frame < before_frame]
        if not matches:
            return []
        if any(e.descriptor.shape != q.shape for e in matches):
            raise ValueError("query descriptor shape does not match memory")
        return sorted(matches, key=lambda e: float((e.descriptor - q).abs().mean()))[:max(0, count)]

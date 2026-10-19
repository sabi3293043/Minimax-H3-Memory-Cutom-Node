"""Prediction-drift heuristic; not semantic motion/identity recognition."""
import math

import torch


class AdaptiveSol:
    def __init__(self, base, boost, threshold, end):
        self.base, self.boost, self.threshold, self.end = base, boost, threshold, end
        self.previous = None
        self.progress = 0.
        self.stable = 0
        self.changed = False
        self.change = None
        self.state = 'warmup'

    def observe(self, fingerprint, progress):
        self.progress = progress
        if fingerprint is None or not bool(torch.isfinite(fingerprint).all()):
            self.previous = None
            self.stable = 0
            self.changed = True
            self.state = 'missing/nonfinite prediction'
            return
        change = None
        if self.previous is not None and self.previous.shape == fingerprint.shape:
            delta = (fingerprint - self.previous).square().mean(dim=1).sqrt()
            norm = self.previous.square().mean(dim=1).sqrt().clamp_min(.1)
            change = float((delta / norm).max())
        self.previous = fingerprint.detach()
        self.change = change
        self.changed = change is not None and (not math.isfinite(change) or change >= self.threshold)
        self.stable = self.stable + 1 if change is not None and not self.changed else 0
        self.state = 'prediction changed' if self.changed else ('stable' if self.stable >= 2 else 'warmup')

    def tau(self):
        value = self.base
        if self.changed:
            value -= .35
        elif self.stable >= 2:
            value += self.boost
        if self.progress >= max(0., self.end - .1):
            value = min(value, self.base - .2)
        return min(2., max(0., value))

    def close(self):
        self.previous = None

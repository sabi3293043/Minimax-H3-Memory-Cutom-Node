"""Cheap scheduling, spatial-only observation and resolution-boundary contracts."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from mosaichunk_h3.observation_policy import (
    ObservationPolicy, ObservationSettings, latent_fingerprint, resize_observation,
)
from mosaichunk_h3.visual_memory_index import AdaptiveVisualIndex


class PolicyTests(unittest.TestCase):
    def policy(self, **kwargs):
        return ObservationPolicy(ObservationSettings(**kwargs), min_progress=.25, interval=1)

    def test_progressive_dimensions_reach_full_size_without_upsampling(self):
        policy = self.policy(progressive_resolution=True)
        shape = [1, 24, 27, 20, 32]
        self.assertEqual(policy.dimensions(shape, .25), (10, 16))
        self.assertEqual(policy.dimensions(shape, .5), (15, 24))
        self.assertEqual(policy.dimensions(shape, .75), (20, 32))
        self.assertEqual(policy.dimensions(shape, .9), (20, 32))
        self.assertEqual(policy.dimensions([1, 24, 3, 2, 3], .25), (2, 3))

    def test_spatial_resize_preserves_time_dtype_and_original_input(self):
        video = torch.stack([torch.full((1, 24, 8, 12), float(t)) for t in range(7)], dim=2).bfloat16()
        original = video.clone()
        small = resize_observation(video, (4, 6))
        self.assertEqual(small.shape, (1, 24, 7, 4, 6))
        self.assertEqual(small.dtype, video.dtype)
        self.assertTrue(torch.equal(small[0, 0, :, 0, 0], torch.arange(7).bfloat16()))
        self.assertTrue(torch.equal(video, original))
        self.assertIs(resize_observation(video, (8, 12)), video)
        with self.assertRaises(ValueError):
            resize_observation(video, (16, 12))

    def test_fingerprint_is_bounded_and_does_not_mutate_video(self):
        video = torch.arange(1 * 24 * 100 * 6 * 8).reshape(1, 24, 100, 6, 8).float()
        original = video.clone()
        fingerprint = latent_fingerprint(video)
        self.assertEqual(fingerprint.shape, (64, 24 * 16))
        self.assertEqual(fingerprint.device.type, 'cpu')
        self.assertTrue(torch.equal(video, original))

    def test_stable_predictions_keep_warmup_then_refresh_periodically(self):
        policy = ObservationPolicy(ObservationSettings(schedule='adaptive'), min_progress=0, interval=1)
        video = torch.ones(1, 24, 7, 8, 8)
        observed = []
        for step in range(12):
            decision = policy.decide(video, step=step, total=20)
            if decision.observe:
                observed.append(step)
                policy.observed(decision, step=step)
        self.assertEqual(observed, [0, 1, 2, 5, 8, 11])

    def test_accumulated_prediction_drift_forces_an_early_refresh(self):
        policy = ObservationPolicy(ObservationSettings(schedule='adaptive'), min_progress=0, interval=1)
        video = torch.ones(1, 24, 7, 8, 8)
        for step in range(3):
            policy.observed(policy.decide(video, step=step, total=20), step=step)
        self.assertFalse(policy.decide(video * 1.02, step=3, total=20).observe)
        changed = policy.decide(video * 1.05, step=4, total=20)
        self.assertTrue(changed.observe)
        self.assertEqual(changed.reason, 'prediction changed')

    def test_change_in_one_temporal_sample_is_not_averaged_away(self):
        policy = ObservationPolicy(ObservationSettings(schedule='adaptive'), min_progress=0, interval=1)
        video = torch.ones(1, 24, 7, 8, 8)
        for step in range(3):
            policy.observed(policy.decide(video, step=step, total=20), step=step)
        changed = video.clone()
        changed[:, :, 3] += .2
        self.assertTrue(policy.decide(changed, step=3, total=20).observe)

    def test_resolution_transition_refreshes_and_restarts_warmup(self):
        policy = self.policy(schedule='adaptive', progressive_resolution=True)
        video = torch.ones(1, 24, 7, 8, 12)
        first = policy.decide(video, step=4, total=20)
        policy.observed(first, step=4)
        middle = policy.decide(video, step=9, total=20)
        self.assertTrue(middle.observe)
        self.assertNotEqual(first.size, middle.size)
        self.assertEqual(middle.reason, 'resolution refresh')
        policy.observed(middle, step=9)
        self.assertEqual(policy.stage_observations, 1)

    def test_fixed_schedule_keeps_original_intervals_without_fingerprint_cost(self):
        policy = ObservationPolicy(ObservationSettings(), min_progress=.25, interval=2)
        with patch('mosaichunk_h3.observation_policy.latent_fingerprint') as fingerprint:
            steps = [step for step in range(4, 12)
                     if policy.decide(torch.zeros(1, 24, 7, 8, 8), step=step, total=20).observe]
        self.assertEqual(steps, [4, 6, 8, 10])
        fingerprint.assert_not_called()

    def test_resolution_reset_evicts_records_and_discards_cross_scale_stability(self):
        index = AdaptiveVisualIndex()
        frames = torch.zeros(3, 3, 8, 8)
        descriptor = index.candidates(frames, step=0)[0][1]
        index.put(0, 0, descriptor, .9, 'old-image', on_evict=lambda _: None)
        evicted = []
        index.reset(on_evict=evicted.append)
        self.assertEqual(evicted, ['old-image'])
        self.assertEqual(index.entries, [])
        self.assertIsNone(index._previous)
        self.assertEqual(index.candidates(frames, step=1)[0][2], .25)

    def test_invalid_adaptive_and_progressive_settings_fail_explicitly(self):
        base = ObservationSettings()
        for update in [dict(start_scale=float('nan')), dict(full_progress=float('nan')),
                       dict(schedule='unknown'), dict(change_threshold=0), dict(max_interval=0),
                       dict(progressive_resolution=True, full_progress=.2)]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                replace(base, **update).validate(min_progress=.25, interval=1)
        with self.assertRaises(ValueError):
            ObservationSettings(schedule='adaptive', max_interval=2).validate(min_progress=.25, interval=3)


if __name__ == '__main__':
    unittest.main()

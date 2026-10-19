"""RAM storage equivalence, admission and adaptive-policy cache invalidation."""
from dataclasses import replace
from unittest.mock import patch
import unittest
import weakref

import torch

from mosaichunk_h3.adaptive_sol import AdaptiveSol
from mosaichunk_h3.host_residual import HostResidual, HostAllocationError
import test_generation_acceleration as cache_tests


class HostResidualTests(unittest.TestCase):
    def test_tiled_storage_and_add_preserve_dtype_and_current_features(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            gen = torch.Generator().manual_seed(71)
            first = torch.randn(31, 32, generator=gen).to(dtype)
            after = first + torch.randn(31, 32, generator=gen).to(dtype)
            current = torch.randn(31, 32, generator=gen).to(dtype)
            original = current.clone()
            store = HostResidual(first, 4 * 32 * first.element_size())
            self.assertEqual(store.chunk_rows, 4)
            self.assertEqual(store.nbytes, (31 + 4) * 32 * first.element_size())
            store.finish(after)
            expected = current + (-first + after)
            self.assertTrue(torch.equal(store.add(current), expected))
            self.assertTrue(torch.equal(current, original))
            self.assertEqual(store.data.dtype, dtype)
            self.assertEqual(store.transferred, 0)  # No fake PCIe bytes for CPU tests.


class RamSessionTests(unittest.TestCase):
    setUp = cache_tests.AccelerationTests.setUp
    args = cache_tests.AccelerationTests.args
    step = cache_tests.AccelerationTests.step
    session = cache_tests.AccelerationTests.session
    def test_ram_and_gpu_storage_follow_identical_arithmetic(self):
        gpu, ram = self.session(), self.session(cache_storage='ram')
        for sigma in (.9, .8, .7, .6):
            a, calls_a = self.step(gpu, sigma, start=3.)
            b, calls_b = self.step(ram, sigma, start=3.)
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(calls_a, calls_b)
        self.assertGreater(ram.peak_ram_bytes, 0)
        self.assertLess(ram.peak_bytes, gpu.peak_bytes)
        ram.close()
        self.assertEqual((ram.bytes, ram.ram_bytes), (0, 0))

    def test_auto_falls_back_to_ram_with_insufficient_gpu_residual_space(self):
        session = self.session(cache_storage='auto')
        with patch.object(session, 'available_gpu', return_value=(0, 1024**2)):
            self.step(session, .9)
            _, calls = self.step(session, .8)
        self.assertLess(len(calls), 12)
        self.assertEqual(set(session.backends.values()), {'ram'})
        self.assertGreater(session.peak_ram_bytes, 0)

    def test_host_budget_and_pressure_reject_capture(self):
        for available in (0, 1):
            session = self.session(cache_storage='ram')
            with patch.object(session, 'available_ram', return_value=available):
                self.step(session, .9)
                _, calls = self.step(session, .8)
            self.assertEqual(calls, list(range(12)))
            self.assertEqual(session.ram_bytes, 0)

    def test_mixed_backend_admission_counts_ram_probes_against_gpu_storage_limit(self):
        for budget, expected in [(10000, {'gpu'}), (20000, {'gpu', 'ram'})]:
            session = self.session(cache_storage='auto', cache_gib=budget / 1024**3)
            with patch.object(session, 'available_gpu', return_value=(6912, 16384)):
                self.step(session, .9)
            self.assertEqual(set(session.backends.values()), expected)
            self.assertLessEqual(session.bytes, budget)

    def test_ram_headroom_does_not_require_a_full_gpu_residual(self):
        session = self.session(cache_storage='ram')
        descriptor = 5 * 8 * 32 * 4
        with patch.object(session, 'available_gpu', return_value=(0, descriptor * 2)):
            self.step(session, .9)
        self.assertEqual(len(session.allowed), 2)
        self.assertGreater(session.ram_bytes, 0)

    def test_ram_allocation_failure_does_not_change_real_model_output(self):
        session = self.session(cache_storage='ram')
        with patch('mosaichunk_h3.generation_acceleration.HostResidual.__init__',
                   side_effect=HostAllocationError('test host pressure')):
            out, calls = self.step(session, .9)
        self.assertTrue(torch.equal(out, torch.full_like(out, 78)))
        self.assertEqual(calls, list(range(12)))
        self.assertEqual((session.bytes, session.ram_bytes), (0, 0))

    def test_storage_switch_discards_incompatible_buffer(self):
        session = self.session(cache_storage='auto')
        self.step(session, .9)
        with patch.object(session, 'available_gpu', return_value=(0, 1024**2)):
            _, calls = self.step(session, .8)
        self.assertEqual(calls, list(range(12)))
        self.assertGreater(session.ram_bytes, 0)
        self.assertTrue(all(isinstance(s.residual, HostResidual) for s in session.sections))

    def test_ram_limit_counts_full_buffer_and_cpu_scratch(self):
        session = self.session(cache_storage='ram', cache_ram_gib=3600 / 1024**3)
        self.step(session, .9)
        self.assertEqual(len(session.allowed), 1)
        self.assertLessEqual(session.ram_bytes, 3600)

    def test_replay_oom_falls_back_to_real_blocks(self):
        session = self.session(cache_storage='ram')
        self.step(session, .9)
        with patch.object(HostResidual, 'add', side_effect=torch.OutOfMemoryError('test replay')):
            out, calls = self.step(session, .8)
        self.assertTrue(torch.equal(out, torch.full_like(out, 78)))
        self.assertEqual(calls, list(range(12)))
        self.assertEqual(session.cache_hits, 0)
        self.assertEqual((session.bytes, session.ram_bytes), (0, 0))

    def test_failed_replay_temporaries_are_released_before_real_model_fallback(self):
        session = self.session(cache_storage='ram')
        self.step(session, .9)
        refs = []
        def fail_replay(store, current):
            temporary = torch.ones(1)
            refs.append(weakref.ref(temporary))
            raise torch.OutOfMemoryError('allocation with a live temporary')
        def effect(index):
            if refs:
                self.assertIsNone(refs[-1](), 'exception traceback still holds failed replay storage')
            return index + 1
        with patch.object(HostResidual, 'add', fail_replay):
            self.step(session, .8, effect=effect)
        self.assertEqual(len(refs), 2)

    def test_adaptive_tau_changes_invalidate_stored_residuals(self):
        session = self.session(mode='sol_and_cache', adaptive_sol=True, cache_storage='ram')
        self.step(session, .9)
        self.step(session, .8)
        old_hits = session.cache_hits
        session.adaptive.observe(torch.ones(3, 16), .3)
        session.adaptive.observe(torch.ones(3, 16) * 2, .4)
        _, calls = self.step(session, .7)
        self.assertEqual(calls, list(range(12)))
        self.assertEqual(session.cache_hits, old_hits)
        self.assertLess(session.tau, session.config.sol_tau)

    def test_missing_observation_removes_stale_stability_boost(self):
        session = self.session(mode='sol', adaptive_sol=True)
        for _ in range(3):
            session.adaptive.observe(torch.ones(3, 8), .4)
        self.assertGreater(session.adaptive.tau(), session.config.sol_tau)
        session.prediction_unavailable()
        self.assertIsNone(session.adaptive.previous)
        self.assertLess(session.adaptive.tau(), session.config.sol_tau)


class AdaptiveSolTests(unittest.TestCase):
    def test_stability_hysteresis_change_veto_and_final_detail_cap(self):
        policy = AdaptiveSol(1., .15, .04, .85)
        for i in range(3):
            policy.observe(torch.ones(5, 16), .3 + .05 * i)
        self.assertAlmostEqual(policy.tau(), 1.15)
        changed = torch.ones(5, 16)
        changed[-1] *= 2  # One changing temporal sample is sufficient.
        policy.observe(changed, .5)
        self.assertAlmostEqual(policy.tau(), .65)
        policy.observe(changed, .55)
        self.assertAlmostEqual(policy.tau(), 1.)
        policy.observe(changed, .6)
        self.assertAlmostEqual(policy.tau(), 1.15)
        policy.progress = .8
        self.assertAlmostEqual(policy.tau(), .8)
        policy.close()
        self.assertIsNone(policy.previous)

    def test_shape_reset_nonfinite_and_clamped_tau(self):
        policy = AdaptiveSol(.1, .3, .04, .85)
        policy.observe(None, .3)
        self.assertEqual(policy.tau(), 0.)
        policy.observe(torch.ones(3, 8), .3)
        policy.observe(torch.ones(4, 8), .4)
        self.assertEqual(policy.stable, 0)
        policy.observe(torch.full((4, 8), float('nan')), .5)
        self.assertEqual(policy.tau(), 0.)

    def test_invalid_config_controls(self):
        from mosaichunk_h3.generation_acceleration import AccelerationConfig
        for update in [dict(cache_storage='disk'), dict(cache_ram_gib=float('inf')),
                       dict(transfer_mib=0), dict(sol_stable_boost=.4), dict(sol_drift_threshold=0)]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                replace(AccelerationConfig(), **update).validate()


if __name__ == '__main__':
    unittest.main()

"""Cache arithmetic, hook preservation, budget and failure lifecycle contracts."""
from dataclasses import replace
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import torch

from mosaichunk_h3.generation_acceleration import (
    AccelerationConfig, AccelerationSession, cache_sections, install_acceleration,
)


class AccelerationTests(unittest.TestCase):
    def setUp(self):
        self.root = SimpleNamespace(blocks=[SimpleNamespace(attn=object()) for _ in range(12)])
        self.sampling = SimpleNamespace(percent_to_sigma=lambda p: 1 - p)
        self.guidance = SimpleNamespace(layers={4, 8}, pairs=(), progress=.3,
                                        config=SimpleNamespace(end_progress=.95))
        self.config = AccelerationConfig(mode='selective_cache', start=0, end=.9)
        self.layout = SimpleNamespace(seq_len=14, signature=(2, 3, 4, 4, 1),
            segments=[(0, 1, 'text'), (1, 2, 'audio'), (2, 14, 'video')])
        self.options = {'transformer_options': {}}
        self.patcher = SimpleNamespace(get_model_object=lambda name:
            self.root if name == 'diffusion_model' else self.sampling)
        self.guider = SimpleNamespace(model_options=self.options, model_patcher=self.patcher)

    def args(self, x, sigma):
        return dict(img=x, rope_freqs='native-rope', transformer_options={
            'sigmas': torch.tensor([sigma]), 'uuids': ('cond',), 'minimax_h3_layout': self.layout})

    def step(self, session, sigma, start=0., effect=None):
        calls = []
        x = torch.full((14, 32), start)
        for i in range(12):
            def execute(args, index=i):
                calls.append(index)
                return {'img': args['img'] + (effect(index) if effect else index + 1)}
            x = session.block(i, self.args(x, sigma), execute)['img']
        return x, calls

    def session(self, **updates):
        return AccelerationSession(replace(self.config, **updates), self.root, self.sampling, self.guidance)

    def test_sections_do_not_include_memory_or_boundary_layers(self):
        self.assertEqual(cache_sections(12, {0, 4, 8, 11}), ((1, 2, 3), (5, 6, 7)))

    def test_hits_add_residual_to_current_features_and_keep_guidance_real(self):
        session = self.session()
        first, _ = self.step(session, .9)
        second, calls = self.step(session, .8, start=7.)
        self.assertTrue(torch.equal(second, first + 7))
        self.assertEqual(calls, [0, 1, 4, 5, 8, 9, 10, 11])
        self.assertEqual(session.reused_blocks, 4)
        self.assertEqual(session.cache_hits, 2)
        self.assertGreater(session.peak_bytes, 0)

    def test_reuse_cap_forces_refresh(self):
        session = self.session()
        self.step(session, .9)
        self.step(session, .8)
        _, calls = self.step(session, .7)
        self.assertEqual(calls, list(range(12)))

    def test_audio_and_single_frame_change_veto_reuse(self):
        for changed_rows in [slice(1, 2), slice(10, 14)]:
            with self.subTest(rows=changed_rows):
                session = self.session()
                self.step(session, .9)
                def effect(index):
                    change = torch.full((14, 32), index + 1.)
                    if index in (1, 5):
                        change[changed_rows] *= 1.5
                    return change
                _, calls = self.step(session, .8, effect=effect)
                self.assertEqual(calls, list(range(12)))

    def test_memory_plan_and_schedule_end_changes_invalidate(self):
        for change in ['pairs', 'end']:
            session = self.session()
            self.guidance.pairs = ()
            self.guidance.progress = .3
            self.step(session, .9)
            if change == 'pairs':
                self.guidance.pairs = ((0, 2),)
            else:
                self.guidance.progress = .96
            _, calls = self.step(session, .8)
            self.assertEqual(calls, list(range(12)))
            self.assertEqual(session.invalidations, 2)

    def test_repeated_reversed_sigma_and_condition_change_invalidate(self):
        for sigma, condition in [(.9, 'cond'), (.95, 'cond'), (.8, 'changed')]:
            session = self.session()
            self.step(session, .9)
            if condition != 'cond':
                session.signature = session.signature[:-2] + (('different',), session.signature[-1])
            _, calls = self.step(session, sigma)
            self.assertEqual(calls, list(range(12)))

    def test_no_cache_outside_window_and_tiny_budget(self):
        for config in [dict(start=.5), dict(cache_gib=1e-9)]:
            session = self.session(**config)
            self.step(session, .9)
            _, calls = self.step(session, .8)
            self.assertEqual(calls, list(range(12)))
            self.assertEqual(session.bytes, 0)

    def test_budget_selects_only_fitting_sections_and_cleanup_releases_all(self):
        # A residual costs 1792 bytes, a probe costs 5120 bytes at this shape.
        session = self.session(cache_gib=7200 / 1024**3)
        self.step(session, .9)
        self.assertLessEqual(session.bytes, 7200)
        self.assertGreater(session.bytes, 0)
        session.close()
        self.assertEqual(session.bytes, 0)
        self.assertTrue(all(s.probe is None and s.residual is None for s in session.sections))

    def test_head_allocation_oom_preserves_real_result(self):
        session = self.session()
        with patch('mosaichunk_h3.generation_acceleration.probe_features',
                   side_effect=torch.OutOfMemoryError('test allocation')):
            out, calls = self.step(session, .9)
        self.assertTrue(torch.equal(out, torch.full_like(out, 78)))
        self.assertEqual(calls, list(range(12)))
        self.assertEqual(session.bytes, 0)

    def test_nonfinite_probe_cannot_reuse_cached_residual(self):
        session = self.session()
        self.step(session, .9)
        with patch('mosaichunk_h3.generation_acceleration.probe_features',
                   return_value=torch.full((5, 8, 32), float('nan'))):
            _, calls = self.step(session, .8)
        self.assertEqual(calls, list(range(12)))
        self.assertEqual(session.bytes, 0)

    def test_underlying_model_oom_is_not_swallowed(self):
        session = self.session()
        def failing(args):
            raise torch.OutOfMemoryError('model OOM')
        with self.assertRaisesRegex(torch.OutOfMemoryError, 'model OOM'):
            session.block(1, self.args(torch.zeros(14, 32), .8), failing)

    def test_model_error_propagates_and_restores_options(self):
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            with install_acceleration(self.guider, self.config, self.guidance) as session:
                self.step(session, .9)
                raise RuntimeError('cancelled')
        self.assertIs(self.guider.model_options, self.options)
        self.assertTrue(session.closed)
        self.assertEqual(session.bytes, 0)

    def test_off_does_not_change_options(self):
        with install_acceleration(self.guider, AccelerationConfig()) as session:
            self.assertIsNone(session)
            self.assertIs(self.guider.model_options, self.options)

    def test_other_caches_and_block_patches_fail_before_mutation(self):
        for update in [{'spectrum_h3_binding': object()},
                       {'transformer_options': {'easycache': object()}},
                       {'transformer_options': {'patches_replace': {'dit': {('double_block', 1): object()}}}}]:
            self.guider.model_options = update
            with self.assertRaises(ValueError):
                with install_acceleration(self.guider, self.config, self.guidance):
                    self.fail('must reject')
            self.assertIs(self.guider.model_options, update)

    def fake_sol(self):
        module = ModuleType('fake_sparse_attention')
        class SparsePatch:
            def __init__(self, **kw): self.__dict__.update(kw)
            def reset(self): self.reset_called = True
        module.SparseAttnPatch = SparsePatch
        module.h3_eligible = lambda attn, x, rope, opts, patch, i: True
        module.h3_sparse_attention = lambda attn, h, rope, opts, patch, i: h + 1
        return module

    def test_sol_protects_memory_layers_and_composes_previous_blocks(self):
        module = self.fake_sol()
        previous_calls = []
        def previous(args, extra):
            previous_calls.append(True)
            self.assertNotIn('attention', args)
            return extra['original_block'](args)
        self.options['transformer_options']['patches_replace'] = {'dit': {('double_block', 2): previous}}
        with patch('mosaichunk_h3.generation_acceleration.importlib.import_module', return_value=module):
            with install_acceleration(self.guider, replace(self.config, mode='sol'), self.guidance) as session:
                replacements = self.guider.model_options['transformer_options']['patches_replace']['dit']
                for i in range(12):
                    def original(args, index=i):
                        if index in {0, 2, 4, 8, 11}:
                            self.assertNotIn('attention', args)
                        else:
                            self.assertIn('attention', args)
                        return {'img': args['img']}
                    replacements[('double_block', i)](self.args(torch.zeros(14, 32), .8), {'original_block': original})
                self.assertEqual(session.sparse_calls, 7)
        self.assertEqual(previous_calls, [True])
        self.assertTrue(session.sol_patch.reset_called)
        self.assertIs(self.guider.model_options, self.options)

    def test_missing_sol_module_is_explicit_dense_fallback(self):
        with patch('mosaichunk_h3.generation_acceleration.importlib.import_module', side_effect=ImportError('missing')):
            with install_acceleration(self.guider, replace(self.config, mode='sol')) as session:
                self.assertIsNone(session.sol_patch)
                self.assertTrue(session.reasons)
        self.assertEqual(session.sparse_calls, 0)

    def test_combined_sol_cache_keeps_probes_dense_and_skips_sparse_work_on_hit(self):
        module = self.fake_sol()
        config = replace(self.config, mode='sol_and_cache')
        with patch('mosaichunk_h3.generation_acceleration.importlib.import_module', return_value=module):
            with install_acceleration(self.guider, config, self.guidance) as session:
                replacements = self.guider.model_options['transformer_options']['patches_replace']['dit']
                for sigma in (.9, .8):
                    x = torch.zeros(14, 32)
                    for i in range(12):
                        def original(args, index=i):
                            if index in {0, 1, 4, 5, 8, 11}:
                                self.assertNotIn('attention', args)
                            out = args['img']
                            if 'attention' in args:
                                out = args['attention'](out)
                            return {'img': out + index + 1}
                        x = replacements[('double_block', i)](self.args(x, sigma), {'original_block': original})['img']
                self.assertEqual(session.reused_blocks, 4)
                self.assertEqual(session.sparse_calls, 8)

    def test_external_patch_on_memory_layer_rejected_even_without_caching(self):
        self.options['transformer_options']['patches_replace'] = {'dit': {('double_block', 4): object()}}
        with self.assertRaisesRegex(ValueError, 'memory layer'):
            with install_acceleration(self.guider, replace(self.config, mode='sol'), self.guidance):
                self.fail('must reject')
        self.assertIs(self.guider.model_options, self.options)

    def test_invalid_config(self):
        for update in [dict(mode='bad'), dict(start=.9, end=.2), dict(sol_tau=float('nan')),
                       dict(cache_threshold=0), dict(cache_gib=0), dict(cache_max_reuse=0)]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                replace(self.config, **update).validate()


if __name__ == '__main__':
    unittest.main()

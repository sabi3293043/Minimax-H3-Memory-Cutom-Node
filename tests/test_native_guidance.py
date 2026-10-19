"""CPU contracts and real attention arithmetic; no generation-quality claim."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from mosaichunk_h3.native_guidance import GuidanceConfig, GuidanceSession, decoded_to_latent


class GuidanceTests(unittest.TestCase):
    def setUp(self):
        self.config = GuidanceConfig(mode='manual_frames', layers='0', source_frame=0,
            target_frames='5', max_anchor_tokens=4)
        self.session = GuidanceSession(self.config, [1, 24, 3, 4, 4])
        self.session.observe(torch.zeros(3, 3, 8, 8), [], step=0, total=4)
        self.layout = SimpleNamespace(segments=[(0, 2, 'text'), (2, 4, 'audio'), (4, 16, 'video')])
        generator = torch.Generator().manual_seed(3)
        self.q = torch.randn(1, 2, 16, 8, generator=generator)
        self.k = torch.randn(1, 2, 16, 8, generator=generator)
        self.v = torch.randn(1, 2, 16, 8, generator=generator)
        self.v[:, :, 12:16] = self.v[:, :, 4:8] + .001
        self.calls = []

    def backend(self, q, k, v, heads, **kwargs):
        self.calls.append((q.shape, k.shape))
        return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)

    def apply(self, options=None):
        return self.session.attention(self.backend, self.q, self.k, self.v, 2,
            skip_reshape=True, transformer_options=options or {
                'block_index': 0, 'minimax_h3_layout': self.layout})

    def test_actual_correction_changes_only_target_video_rows(self):
        q, k, v = self.q.clone(), self.k.clone(), self.v.clone()
        normal = self.backend(q, k, v, 2)
        out = self.apply()
        self.assertTrue(torch.equal(out[:, :12], normal[:, :12]))
        self.assertFalse(torch.equal(out[:, 12:], normal[:, 12:]))
        self.assertGreater(self.session.corrected_rows, 0)
        delta = (out[:, 12:] - normal[:, 12:]).norm(dim=-1)
        self.assertTrue((delta <= normal[:, 12:].norm(dim=-1) * .10001).all())
        self.assertTrue(torch.equal(self.q, q) and torch.equal(self.k, k) and torch.equal(self.v, v))
        self.assertTrue(all(shape[1][2] <= 4 for shape in self.calls[2:]))

    def test_no_confident_match_does_not_apply_a_residual(self):
        self.v[:, :, 12:16] = 0
        normal = self.backend(self.q, self.k, self.v, 2)
        self.calls.clear()
        out = self.apply()
        self.assertTrue(torch.equal(out, normal))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.session.corrected_rows, 0)

    def test_zero_strength_and_other_layers_are_exact_passthrough(self):
        normal = self.backend(self.q, self.k, self.v, 2)
        for strength, layer in [(0., 0), (.1, 1)]:
            self.session.config = replace(self.config, strength=strength)
            self.calls.clear()
            out = self.apply({'block_index': layer, 'minimax_h3_layout': self.layout})
            self.assertTrue(torch.equal(normal, out))
            self.assertEqual(len(self.calls), 1)

    def test_end_of_schedule_disables_corrections(self):
        self.session.progress = .99
        self.apply()
        self.assertEqual(len(self.calls), 1)

    def test_native_temporal_decoder_frame_mapping(self):
        self.assertEqual(decoded_to_latent(0, 90, 27), 0)
        self.assertEqual(decoded_to_latent(1, 90, 27), 1)
        self.assertEqual(decoded_to_latent(16, 90, 27), 4)
        self.assertEqual(decoded_to_latent(17, 90, 27), 5)
        self.assertEqual(decoded_to_latent(89, 90, 27), 26)
        self.assertEqual(decoded_to_latent(8, 27, 27), 8)
        with self.assertRaises(ValueError):
            decoded_to_latent(90, 90, 27)
        with self.assertRaises(ValueError):
            decoded_to_latent(5, 80, 27)

    def test_temporal_pairs_within_same_latent_are_not_guided(self):
        config = replace(self.config, target_frames='1', source_frame=2)
        session = GuidanceSession(config, [1, 24, 27, 4, 4])
        session.observe(torch.zeros(90, 3, 8, 8), [], step=0, total=4)
        self.assertFalse(session.pairs)

    def test_retrieval_requires_prior_step_and_protects_anchor_frames(self):
        session = GuidanceSession(replace(self.config, mode='retrieved_frames'), [1, 24, 3, 4, 4])
        refs = [SimpleNamespace(source_frame=0, query_frame=1, source_step=0, confidence=1.),
                SimpleNamespace(source_frame=1, query_frame=2, source_step=0, confidence=1.),
                SimpleNamespace(source_frame=0, query_frame=2, source_step=1, confidence=1.)]
        session.observe(torch.zeros(3, 3, 8, 8), refs, step=1, total=4)
        self.assertEqual(session.pairs, ((1, 2),))
        session.close()
        self.assertFalse(session.pairs)

    def test_invalid_manual_controls_fail_explicitly(self):
        for update in [dict(strength=float('nan')), dict(target_frames=''),
                       dict(target_frames='0'), dict(layers='bad'), dict(max_anchor_tokens=513)]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                replace(self.config, **update).validate()


if __name__ == '__main__':
    unittest.main()

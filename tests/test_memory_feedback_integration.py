"""Read-only retrieval integration; synthetic H3 frames, not generated footage."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mosaichunk_h3.native_visual_observer import NativeVisualObserver


class FakeDecoder:
    latent_channels = 24

    def decode_video(self, latent, frame_indices=None):
        frames = torch.zeros((7, 3, 16, 16), dtype=torch.uint8)
        frames[0] = 100
        frames[6] = 100
        frames[3] = 200
        return frames


class FeedbackIntegrationTests(unittest.TestCase):
    def test_automatic_guidance_publishes_usable_pairs_for_both_decoder_types(self):
        from mosaichunk_h3.native_guidance import GuidanceConfig, GuidanceSession
        for temporal in (False, True):
            with self.subTest(temporal=temporal), tempfile.TemporaryDirectory() as root:
                count = 22 if temporal else 7
                class Decoder:
                    decodes_prefix = temporal
                    def decode_video(self, latent, frame_indices=None):
                        frames = torch.zeros(count, 3, 8, 8, dtype=torch.uint8)
                        frames[0] = frames[-1] = 100
                        frames[count // 2] = 200
                        return frames
                shape = [1, 24, 7, 2, 2]
                session = GuidanceSession(GuidanceConfig(mode='retrieved_frames', layers='0'), shape)
                seen_pairs, seen_refs, statuses = [], [], []
                observer = NativeVisualObserver(tiny_vae_name='auto', max_frames=4,
                    max_memory_frames=12, min_progress=0, disk_gib=.1, temp_root=root,
                    feedback_hook=lambda step, refs, read: seen_refs.extend(refs))
                def executor(*args, **kwargs):
                    for step in range(5):
                        x0 = torch.zeros(shape)
                        args[5](step, x0, x0, 5)
                        seen_pairs.extend(session.pairs)
                        sources = {source for source, _ in session.pairs}
                        targets = {target for _, target in session.pairs}
                        self.assertFalse(sources & targets)
                    return 'sampled'
                with patch('mosaichunk_h3.native_visual_observer._load_tae_decoder', return_value=Decoder()), \
                     patch('mosaichunk_h3.native_visual_observer.notify_native',
                           side_effect=lambda node, message, **data: statuses.append(data)):
                    result = observer._sample(executor, None, None, None, None,
                        latent_shapes=[shape, [1, 32, 2]], guidance=session)
                self.assertEqual(result, 'sampled')
                self.assertTrue(seen_pairs)
                self.assertTrue(all(0 <= source < target < 7 for source, target in seen_pairs))
                self.assertTrue(seen_refs)
                self.assertTrue(all(ref.source_frame < ref.query_frame for ref in seen_refs))
                self.assertGreater(statuses[-1]['frame_pair_proposals'], 0)
                self.assertEqual(list(Path(root).iterdir()), [])

    def test_previous_sample_retrieval_keeps_sampler_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            proposals, callbacks = [], []
            observer = NativeVisualObserver(
                tiny_vae_name="auto", max_frames=4, max_memory_frames=12,
                min_progress=0, disk_gib=0.25, temp_root=root,
                feedback_hook=lambda step, refs, read: proposals.append(
                    (step, [r.source_step for r in refs],
                     len(read(refs[0].source_record)))),
            )

            def executor(*args, **kwargs):
                x0 = torch.zeros((1, 1, 24 * 3 * 2 * 2))
                for step in range(4):
                    args[5](step, x0, x0, 4)
                    self.assertTrue(torch.equal(x0, torch.zeros_like(x0)))
                return "native result"

            with patch("mosaichunk_h3.native_visual_observer._load_tae_decoder",
                       return_value=FakeDecoder()):
                result = observer(
                    executor, None, None, None, None, None,
                    lambda step, *args: callbacks.append(step), False, None,
                    [[1, 24, 3, 2, 2], [1, 32, 2]],
                )
            self.assertEqual(result, "native result")
            self.assertEqual(callbacks, list(range(4)))
            self.assertTrue(proposals)
            self.assertTrue(all(source < step
                for step, sources, bytecount in proposals for source in sources))
            self.assertEqual(list(Path(root).iterdir()), [])


if __name__ == "__main__":
    unittest.main()

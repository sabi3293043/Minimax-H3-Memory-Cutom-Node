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

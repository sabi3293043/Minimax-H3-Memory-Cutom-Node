"""Native observer tests use a fake decoder and native executor, not H3 weights."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mosaichunk_h3.native_visual_observer import (
    NativeVisualObserver, choose_distinct_frames, _unpack_video_x0,
)

class FakeDecoder:
    latent_channels = 24
    def decode_video(self, latent, frame_indices=None):
        return torch.stack([
            torch.full((3, 8, 8), i * 40, dtype=torch.uint8) for i in range(3)
        ])

class NativeObserverTests(unittest.TestCase):
    def test_content_driven_frame_selection(self):
        images = torch.zeros(5, 3, 12, 12)
        images[1, :, :6] = 1
        images[3] = 0.9
        result = choose_distinct_frames(images, 3)
        self.assertEqual(result[0], 0)
        self.assertIn(1, result)

    def test_packed_h3_av_video_reshape(self):
        shape = [1, 24, 3, 2, 2]
        packed = torch.arange(1 * 24 * 3 * 2 * 2).reshape(1, 1, -1)
        out = _unpack_video_x0(packed, [shape, [1, 32, 2]])
        self.assertEqual(tuple(out.shape), tuple(shape))

    def test_native_executor_callback_and_cleanup(self):
        with tempfile.TemporaryDirectory() as root:
            observer = NativeVisualObserver(tiny_vae_name="taeh3.safetensors",
                                            max_frames=2, min_progress=0,
                                            disk_gib=0.1, temp_root=root)
            called = []
            def callback(*args):
                called.append(args[0])
            def executor(*args, **kwargs):
                x0 = torch.zeros(1, 1, 24 * 3 * 2 * 2)
                args[5](1, x0, x0, 2)
                return "native-result"
            with patch("mosaichunk_h3.native_visual_observer._load_tae_decoder",
                       return_value=FakeDecoder()):
                result = observer(executor, None, None, None, None, None,
                                  callback, False, None,
                                  [[1,24,3,2,2], [1,32,2]])
            self.assertEqual(result, "native-result")
            self.assertEqual(called, [1])
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_executor_failure_cleans_cache(self):
        with tempfile.TemporaryDirectory() as root:
            observer = NativeVisualObserver(tiny_vae_name="taeh3.safetensors",
                                            max_frames=1, min_progress=0,
                                            disk_gib=0.1, temp_root=root)
            def failing_executor(*a, **kw):
                raise RuntimeError("interrupted")
            with patch("mosaichunk_h3.native_visual_observer._load_tae_decoder",
                       return_value=FakeDecoder()):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    observer(failing_executor, None, None, None, None)
            self.assertEqual(list(Path(root).iterdir()), [])

if __name__ == "__main__":
    unittest.main()

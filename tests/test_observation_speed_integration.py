"""Progressive decoder views, persistent retrieval and storage-independent plans."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mosaichunk_h3.native_guidance import GuidanceConfig, GuidanceSession
from mosaichunk_h3.native_visual_observer import NativeVisualObserver
from mosaichunk_h3.observation_policy import ObservationSettings
from mosaichunk_h3.visual_memory_index import AdaptiveVisualIndex


class SpeedIntegrationTests(unittest.TestCase):
    def run_case(self, *, images, settings=None):
        shape = [1, 24, 7, 8, 12]
        video = torch.ones(shape)
        original = video.clone()
        sizes, plans, callbacks, statuses, resets = [], [], [], [], []
        class Decoder:
            def decode_video(self, latent, frame_indices=None):
                sizes.append(tuple(latent.shape[-2:]))
                frames = torch.zeros(7, 3, latent.shape[-2] * 2, latent.shape[-1] * 2,
                                     dtype=torch.uint8)
                frames[0] = frames[-1] = 100
                frames[3] = 200
                return frames
        class Index(AdaptiveVisualIndex):
            def reset(self, **kwargs):
                resets.append(True)
                return super().reset(**kwargs)
        session = GuidanceSession(GuidanceConfig(mode='retrieved_frames', layers='0'), shape)
        def executor(*args, **kwargs):
            for step in range(20):
                args[5](step, video, video, 20)
                plans.append(session.pairs)
            return video
        with tempfile.TemporaryDirectory() as root:
            observer = NativeVisualObserver(tiny_vae_name='auto', max_frames=4,
                min_progress=0, disk_gib=.1, temp_root=root,
                store_observation_images=images, observation_settings=settings)
            with patch('mosaichunk_h3.native_visual_observer._load_tae_decoder', return_value=Decoder()), \
                 patch('mosaichunk_h3.native_visual_observer.AdaptiveVisualIndex', Index), \
                 patch('mosaichunk_h3.native_visual_observer.notify_native',
                       side_effect=lambda node, message, **data: statuses.append(data)):
                result = observer._sample(executor, None, None, None, None,
                    latent_shapes=[shape, [1, 32, 2]], guidance=session,
                    callback=lambda step, *args: callbacks.append(step))
            self.assertIs(result, video)
            self.assertTrue(torch.equal(video, original))
            self.assertEqual(callbacks, list(range(20)))
            self.assertEqual(list(Path(root).iterdir()), [])
        return sizes, plans, statuses[-1], resets

    def test_storage_toggle_preserves_frame_pair_plans(self):
        disk = self.run_case(images=True)
        descriptors = self.run_case(images=False)
        self.assertEqual(disk[0], descriptors[0])
        self.assertEqual(disk[1], descriptors[1])
        self.assertEqual(disk[2]['frame_pair_proposals'], descriptors[2]['frame_pair_proposals'])
        self.assertGreater(disk[2]['peak_cache_bytes'], 0)
        self.assertEqual(descriptors[2]['peak_cache_bytes'], 0)

    def test_adaptive_progressive_decode_grows_without_resetting_retrieval(self):
        sizes, plans, status, resets = self.run_case(images=False, settings=ObservationSettings(
            schedule='adaptive', progressive_resolution=True))
        self.assertEqual(list(dict.fromkeys(sizes)), [(4, 6), (6, 9), (8, 12)])
        self.assertEqual(resets, [])
        self.assertLess(status['observations'], 18)
        self.assertTrue(any(plans))
        self.assertGreater(status['skipped_observations'], 0)
        self.assertEqual(status['preview_size'], (16, 24))
        self.assertEqual(status['reference_preview_sizes'], ((16,24),(12,18),(8,12)))


if __name__ == '__main__':
    unittest.main()

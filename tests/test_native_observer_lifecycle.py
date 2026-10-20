"""Native callback continuity, bounded observation and honest status contracts."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from mosaichunk_h3.native_visual_observer import NativeVisualObserver


class LifecycleTests(unittest.TestCase):
    def test_image_storage_bypass_creates_no_cache_and_encodes_no_images(self):
        with patch('mosaichunk_h3.native_visual_observer.VisualMemoryDisk') as storage, \
             patch('mosaichunk_h3.native_visual_observer._to_webp') as encoder:
            events, callbacks = self.run_observer(self.decoder(),
                observer_options={'store_observation_images': False})
        storage.assert_not_called()
        encoder.assert_not_called()
        self.assertEqual(callbacks, list(range(6)))
        self.assertEqual(events[-1]['peak_cache_bytes'], 0)
        self.assertFalse(events[-1]['store_observation_images'])

    def test_adaptive_schedule_reduces_decodes_and_preserves_all_callbacks(self):
        from mosaichunk_h3.observation_policy import ObservationSettings
        decoder = self.decoder()
        events, callbacks = self.run_observer(decoder, callbacks=12, observer_options={
            'observation_settings': ObservationSettings(schedule='adaptive')})
        self.assertEqual(decoder.decode_video.call_count, 6)
        self.assertEqual(events[-1]['skipped_observations'], 6)
        self.assertEqual(callbacks, list(range(12)))

    def run_observer(self, decoder, *, callbacks=6, observer_options=None, guidance=None,
                     callback=None, direct=False, shape=None, tracker=None):
        shape = shape or [1, 24, 3, 2, 2]
        x0 = torch.zeros(shape)
        x0_original = x0.clone()
        events, forwarded = [], []
        def executor(*args, **kwargs):
            for step in range(callbacks):
                args[5](step, x0, x0, callbacks)
            return 'native-result'
        def downstream(step, *args):
            forwarded.append(step)
            if callback is not None:
                callback(step, *args)
        with tempfile.TemporaryDirectory() as root:
            observer = NativeVisualObserver(tiny_vae_name='auto', max_frames=2,
                min_progress=0, disk_gib=.1, temp_root=root,
                **(observer_options or {}))
            def status(node, message, **data):
                events.append(dict(message=message, **data))
            with patch('mosaichunk_h3.native_visual_observer._load_tae_decoder', return_value=decoder), \
                 patch('mosaichunk_h3.native_visual_observer.notify_native', side_effect=status):
                try:
                    method = observer._sample if direct else observer
                    result = method(executor, None, None, None, None,
                        callback=downstream, latent_shapes=[shape, [1, 32, 2]],
                        **({'guidance': guidance,'tracker':tracker} if direct else {}))
                finally:
                    self.assertEqual(list(Path(root).iterdir()), [])
                    self.assertTrue(torch.equal(x0, x0_original))
            self.assertEqual(result, 'native-result')
        return events, forwarded

    def decoder(self):
        return SimpleNamespace(decode_video=Mock(return_value=torch.zeros(3, 3, 8, 8)))

    def test_interval_reduces_decode_calls_without_dropping_sampler_callbacks(self):
        decoder = self.decoder()
        events, forwarded = self.run_observer(decoder, observer_options={'observation_interval': 2})
        self.assertEqual(decoder.decode_video.call_count, 3)
        self.assertEqual(forwarded, list(range(6)))
        self.assertEqual(events[-1]['observations'], 3)
        self.assertIn('guidance inactive', events[-1]['message'])
        self.assertIn('cache removed', events[-1]['message'])
        self.assertGreater(events[-1]['peak_cache_bytes'], 0)
        self.assertGreater(events[-1]['observation_seconds'], 0)

    def test_cuda_oom_disables_observation_and_clears_stale_guidance(self):
        decoder = self.decoder()
        decoder.decode_video.side_effect = torch.OutOfMemoryError('synthetic CUDA OOM')
        from mosaichunk_h3.native_guidance import GuidanceConfig, GuidanceSession
        session = GuidanceSession(GuidanceConfig(mode='manual_frames', layers='0',
            target_frames='5'), [1,24,3,2,2])
        session.pairs = ((0,2),)
        events, forwarded = self.run_observer(decoder, direct=True, guidance=session)
        self.assertEqual(decoder.decode_video.call_count, 1)
        self.assertFalse(session.pairs)
        self.assertEqual(forwarded, list(range(6)))
        self.assertIn('synthetic CUDA OOM', events[-1]['disabled_reason'])
        self.assertEqual(events[-1]['observations'], 0)

    def test_observation_only_budget_preserves_native_generation(self):
        decoder = self.decoder()
        events, forwarded = self.run_observer(decoder,
            observer_options={'max_observation_megapixels': 1.}, shape=[1, 24, 3, 64, 64])
        decoder.decode_video.assert_not_called()
        self.assertEqual(forwarded, list(range(6)))
        self.assertIn('max_observation_megapixels', events[-1]['disabled_reason'])
        self.assertEqual(events[-1]['corrections'], 0)

    def test_baseexception_cancellation_propagates_and_cleans_images(self):
        class Cancelled(BaseException):
            pass
        def callback(step, *args):
            raise Cancelled('user cancelled')
        with self.assertRaisesRegex(Cancelled, 'user cancelled'):
            self.run_observer(self.decoder(), callback=callback)

    def test_active_guidance_does_not_decode_final_unused_prediction(self):
        from mosaichunk_h3.native_guidance import GuidanceConfig, GuidanceSession
        session = GuidanceSession(GuidanceConfig(mode='manual_frames', layers='0',
            target_frames='5'), [1, 24, 3, 2, 2])
        decoder = self.decoder()
        events, forwarded = self.run_observer(decoder, direct=True, guidance=session, callbacks=4)
        self.assertEqual(decoder.decode_video.call_count, 3)
        self.assertEqual(events[-1]['guidance_mode'], 'manual_frames')
        self.assertEqual(forwarded, [0, 1, 2, 3])
        self.assertFalse(session.pairs)

    def test_manual_frame_plans_survive_callback_with_and_without_inspector(self):
        from mosaichunk_h3.native_guidance import GuidanceConfig, GuidanceSession
        from mosaichunk_h3.memory_controls import MemoryOptions
        for inspect in (True,False):
            session=GuidanceSession(GuidanceConfig(mode='manual_frames',layers='0',
                target_frames='5'),[1,24,3,2,2])
            def check_plan(step,*args):
                if step<3:self.assertEqual(session.pairs,((0,2),))
            self.run_observer(self.decoder(),direct=True,guidance=session,callbacks=4,
                callback=check_plan,observer_options={'memory_options':MemoryOptions(inspect=inspect)})

    def test_tracking_failure_keeps_whole_frame_memory_and_callbacks(self):
        from mosaichunk_h3.native_guidance import GuidanceConfig,GuidanceSession
        from mosaichunk_h3.memory_controls import MemoryOptions
        session=GuidanceSession(GuidanceConfig(mode='manual_frames',layers='0',
            target_frames='5'),[1,24,3,2,2])
        tracker=SimpleNamespace(track=Mock(side_effect=RuntimeError('tracker unavailable')),close=Mock())
        def check_plan(step,*args):
            if step<3:self.assertEqual(session.pairs,((0,2),))
        events,forwarded=self.run_observer(self.decoder(),direct=True,guidance=session,
            callbacks=4,callback=check_plan,tracker=tracker,
            observer_options={'memory_options':MemoryOptions(occlusion=True,inspect=True)})
        self.assertEqual(forwarded,[0,1,2,3])
        self.assertEqual(tracker.track.call_count,1)
        tracker.close.assert_called_once()
        self.assertFalse(events[-1]['disabled_reason'])
        self.assertIn('whole-frame memory continues',events[-1]['tracking']['reason'])

    def test_no_detected_objects_keeps_whole_frame_memory(self):
        from mosaichunk_h3.native_guidance import GuidanceConfig,GuidanceSession
        from mosaichunk_h3.memory_controls import MemoryOptions
        session=GuidanceSession(GuidanceConfig(mode='manual_frames',layers='0',
            target_frames='5'),[1,24,3,2,2])
        tracker=SimpleNamespace(track=Mock(return_value={}),regions=(),
            last_reason='No confident objects; whole-frame memory remains active')
        def check_plan(step,*args):
            if step<3:self.assertEqual(session.pairs,((0,2),))
        events,_=self.run_observer(self.decoder(),direct=True,guidance=session,callbacks=4,
            callback=check_plan,tracker=tracker,
            observer_options={'memory_options':MemoryOptions(occlusion=True,inspect=True)})
        self.assertEqual(tracker.track.call_count,3)
        self.assertFalse(events[-1]['disabled_reason'])

    def test_invalid_runtime_budgets_fail_before_execution(self):
        for options in [dict(observation_interval=0), dict(observation_interval=101),
                        dict(disk_gib=float('inf')), dict(disk_gib=float('nan'))]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                NativeVisualObserver(tiny_vae_name='auto', max_frames=2,
                    min_progress=0, **({'disk_gib': .1} | options))


if __name__ == '__main__':
    unittest.main()

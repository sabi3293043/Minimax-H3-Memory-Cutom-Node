"""Retrieved indices share current RGB, with bounded coverage and fresh masks."""
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from mosaichunk_h3.memory_controls import MemoryOptions
from mosaichunk_h3.occlusion_tracking import OcclusionTracker
from mosaichunk_h3.tracking_selection import select_tracking_frames, expand_tracking_frames


class RetrievalSelectionTests(unittest.TestCase):
    def test_retrieved_endpoints_win_over_even_spacing_with_a_bridge_and_discovery(self):
        ids = select_tracking_frames(107, pairs=((17, 79), (19, 81)))
        self.assertEqual(len(ids), 8)
        self.assertTrue({0, 17, 19, 79, 81, 106}.issubset(ids))
        self.assertTrue(any(19 < i < 79 for i in ids))
        self.assertEqual(ids, tuple(sorted(set(ids))))
        small = select_tracking_frames(107, pairs=((17,79),), budget=4)
        self.assertTrue({0,17,79,106}.issubset(small))

    def test_warmup_and_uniform_have_bounded_clip_coverage(self):
        warmup = select_tracking_frames(107)
        self.assertEqual(len(warmup), 8)
        self.assertEqual((warmup[0], warmup[-1]), (0, 106))
        self.assertEqual(warmup, select_tracking_frames(107, mode='uniform', pairs=((17, 79),)))

    def test_short_clips_single_frame_and_late_sources(self):
        for count in (1, 2, 3, 7, 8):
            self.assertEqual(select_tracking_frames(count), tuple(range(count)))
        ids = select_tracking_frames(30, sources=(22, 25))
        self.assertEqual(ids, tuple(range(22, 30)))

    def test_protected_sources_survive_a_smaller_budget(self):
        sources = tuple(range(0, 80, 10))
        ids = select_tracking_frames(107, sources=sources, budget=4)
        self.assertTrue(set(sources).issubset(ids))
        self.assertIn(106, ids)
        self.assertLessEqual(len(ids), 16)

    def test_expansion_preserves_ids_and_never_exceeds_sixteen(self):
        for count in (1, 9, 107):
            initial = select_tracking_frames(count)
            wide = expand_tracking_frames(count, initial)
            self.assertTrue(set(initial).issubset(wide))
            self.assertEqual(len(wide), min(count, 16))

    def test_rejects_bad_options_and_indices_before_worker_execution(self):
        for options in (dict(budget=3), dict(budget=17), dict(budget=True), dict(mode='unknown'),
                        dict(sources=(107,)), dict(sources=(-1,)), dict(sources=(True,)),
                        dict(pairs=((1, 107),)), dict(pairs=((1, True),))):
            with self.subTest(options=options), self.assertRaises(ValueError):
                select_tracking_frames(107, **options)
        for options in (dict(tracking_frame_budget=0), dict(tracking_frame_budget=True),
                        dict(tracking_frame_budget=8.5), dict(tracking_selection='unknown')):
            with self.subTest(options=options), self.assertRaises(ValueError):
                MemoryOptions(**options).validate()


class SharedTrackingTests(unittest.TestCase):
    def tracker(self, *, hidden=False, empty=False):
        tracker = OcclusionTracker('unused', cpu_threads=1)
        tracker.temp = SimpleNamespace(name='unused-transport-root')
        tracker.process = SimpleNamespace(stdin=io.StringIO())
        requests = []
        def respond(timeout):
            request = json.loads(tracker.process.stdin.getvalue().splitlines()[-1])
            requests.append(request)
            regions = [] if empty else [dict(name='subject', box=[.1, .1, .9, .9])]
            records = [dict(frame=i, region='subject', visible=not hidden or len(requests) > 1,
                            confidence=.99, box=[.1, .1, .9, .9], mask=[[len(requests)] * 32] * 32)
                       for i in request['ids']] if regions else []
            return dict(records=records, regions=regions, reason='test', observation=request['observation'],
                        metrics=dict(propagation_seconds=1., worker_seconds=1., detection_seconds=0.))
        tracker.wait = Mock(side_effect=respond)
        return tracker, requests

    def frames(self, count=107, resolution=16):
        return torch.arange(count, dtype=torch.uint8)[:, None, None, None].expand(count, 3, resolution, resolution)

    def test_tracks_retrieved_current_rgb_without_mutating_or_retaining_the_pool(self):
        tracker, requests = self.tracker()
        frames = self.frames()
        original = frames.clone()
        result = tracker.track(frames, (), frame_pairs=((17, 79),), step=4)
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertTrue({17, 79}.issubset(request['ids']))
        self.assertEqual(request['observation'], dict(step=4, frame_count=107, resolution=[16, 16]))
        self.assertTrue(torch.equal(frames, original))
        self.assertEqual(set(i for i, _ in result), set(request['ids']))
        self.assertEqual(tracker.last_metrics['encoded_frames'], 8)
        self.assertEqual(tracker.last_metrics['processed_frames'], 8)
        self.assertNotIn('frames', tracker.__dict__)
        self.assertEqual(tracker.last_metrics['covered_pairs'], 1)

    def test_uncertainty_widens_next_fresh_observation_without_repeating_segmentation(self):
        tracker, requests = self.tracker(hidden=True)
        result = tracker.track(self.frames(), (), frame_pairs=((17, 79),), step=5)
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(result), 8)
        self.assertTrue(tracker.last_metrics['next_coverage_wide'])
        self.assertEqual(tracker.last_metrics['processed_frames'], 8)
        result = tracker.track(self.frames().add(20), (), frame_pairs=((17, 79),), step=6)
        self.assertEqual(len(requests), 2)
        first, last = requests
        self.assertTrue(set(first['ids']).issubset(last['ids']))
        self.assertEqual(len(last['ids']), 16)
        first_bytes = dict(zip(first['ids'], first['frames']))
        last_bytes = dict(zip(last['ids'], last['frames']))
        self.assertTrue(all(first_bytes[i] != last_bytes[i] for i in first_bytes))
        self.assertNotEqual(first['observation']['step'], last['observation']['step'])
        self.assertTrue(all(record['mask'][0][0] == 2 for record in result.values()))
        self.assertTrue(tracker.last_metrics['fallback_expanded'])
        self.assertEqual(tracker.last_metrics['encoded_frames'], 16)
        self.assertEqual(tracker.last_metrics['processed_frames'], 16)
        self.assertEqual(tracker.last_metrics['propagation_seconds'], 1.)
        self.assertFalse(tracker.last_metrics['next_coverage_wide'])

    def test_uniform_has_no_adaptive_second_pass(self):
        tracker, requests = self.tracker(hidden=True)
        tracker.track(self.frames(), (), selection='uniform', frame_budget=16)
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(requests[0]['ids']), 16)
        self.assertFalse(tracker.last_metrics['fallback_expanded'])

    def test_low_confidence_or_hidden_masks_cannot_supply_regional_corrections(self):
        from mosaichunk_h3.tracking_selection import tracking_record_trusted
        for record in (None, dict(visible=False,confidence=.99),
                       dict(visible=True,confidence=.6), dict(visible=True,confidence=float('nan'))):
            self.assertFalse(tracking_record_trusted(record))
        tracker, _ = self.tracker(hidden=True)
        result = tracker.track(self.frames(), ())
        self.assertTrue(all(not record['trusted'] for record in result.values()))

    def test_no_detections_returns_no_masks_without_an_extra_search(self):
        tracker, requests = self.tracker(empty=True)
        self.assertEqual(tracker.track(self.frames(), ()), {})
        self.assertEqual(len(requests), 1)
        self.assertFalse(tracker.regions)

    def test_short_uncertain_clip_does_not_repeat_an_identical_selection(self):
        tracker, requests = self.tracker(hidden=True)
        tracker.track(self.frames(3), ())
        self.assertEqual(len(requests), 1)

    def test_different_steps_and_sizes_never_reuse_old_images(self):
        tracker, requests = self.tracker()
        tracker.track(self.frames(), (), step=4)
        tracker.track(self.frames(resolution=32).add(20), (), step=5)
        self.assertNotEqual(requests[0]['observation'], requests[1]['observation'])
        self.assertNotEqual(requests[0]['frames'][0], requests[1]['frames'][0])
        self.assertEqual(tracker.last_metrics['encoded_frames'], 8)

    def test_stale_worker_response_is_rejected(self):
        tracker, _ = self.tracker()
        tracker.wait = Mock(return_value=dict(observation=dict(step=0), records=[]))
        with self.assertRaisesRegex(RuntimeError, 'current observation'):
            tracker.track(self.frames(), (), step=5)

    def test_response_indices_must_belong_to_the_selection(self):
        tracker, _ = self.tracker()
        respond = tracker.wait.side_effect
        def wrong(timeout):
            response = respond(timeout)
            response['records'][0]['frame'] = 999
            return response
        tracker.wait.side_effect = wrong
        with self.assertRaisesRegex(RuntimeError, 'outside the shared selection'):
            tracker.track(self.frames(), ())


if __name__ == '__main__':
    unittest.main()

"""Reference continuity across decoder scales, including deliberately incompatible RGB."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mosaichunk_h3.memory_controls import MemoryOptions
from mosaichunk_h3.native_guidance import GuidanceConfig, GuidanceSession
from mosaichunk_h3.native_visual_observer import NativeVisualObserver
from mosaichunk_h3.observation_policy import (
    ObservationPolicy, ObservationSettings, decode_observation_views,
)
from mosaichunk_h3.region_memory import RegionMemory
from mosaichunk_h3.progressive_memory import ProgressiveReferenceMemory, merge_references
from mosaichunk_h3.visual_memory_index import AdaptiveVisualIndex


class ScaleSensitiveDecoder:
    """Decoder scale changes are not fixable by resizing decoded RGB."""
    def __init__(self):
        self.calls = []
        self.drift = False

    def decode_video(self, latent, frame_indices=None):
        size = tuple(latent.shape[-2:])
        self.calls.append(size)
        value = .4 if size == (4, 6) else .9
        frames = torch.full((latent.shape[2], 3, size[0]*2, size[1]*2), value)
        if self.drift:
            frames[0] = 1.
        return frames


class ProgressiveMemoryTests(unittest.TestCase):
    def test_start_stage_reuses_one_decode_and_full_mode_stays_one_decode(self):
        video = torch.ones(1, 24, 7, 8, 12)
        for size in ((4, 6), (8, 12)):
            decoder = ScaleSensitiveDecoder()
            views = decode_observation_views(decoder, video, size=size, reference_sizes=(size,))
            self.assertEqual(list(views), [size])
            self.assertEqual(decoder.calls, [size])

    def test_larger_previews_do_not_reinterpret_old_descriptors(self):
        decoder = ScaleSensitiveDecoder()
        video = torch.ones(1, 24, 7, 8, 12)
        original = video.clone()
        policy = ObservationPolicy(ObservationSettings(progressive_resolution=True),
            min_progress=.25, interval=1)
        expected, retained_sizes = None, []
        for progress in (.25, .5, .75):
            size = policy.dimensions(video.shape, progress)
            views = decode_observation_views(decoder, video,
                size=policy.dimensions(video.shape, progress),
                reference_sizes=retained_sizes)
            retained_sizes.append(size)
            preview, reference = views[size], views[(4,6)]
            features = AdaptiveVisualIndex.descriptors(reference)
            if expected is None:
                expected = features
            self.assertTrue(torch.equal(features, expected))
            if progress > .25:
                self.assertGreater(float(preview.mean()), float(reference.mean())+.4)
        self.assertTrue(torch.equal(video, original))
        self.assertEqual(decoder.calls, [(4,6), (6,9), (4,6), (8,12), (4,6), (6,9)])

    def test_references_records_and_trusted_region_survive_both_stage_changes(self):
        shape = [1, 24, 7, 8, 12]
        video = torch.ones(shape)
        decoder = ScaleSensitiveDecoder()
        indices, banks, resets, snapshots, statuses = [], [], [], [], []

        class Index(AdaptiveVisualIndex):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                indices.append(self)

            def reset(self, **kwargs):
                resets.append(True)
                return super().reset(**kwargs)

        class Bank(RegionMemory):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                banks.append(self)

        session = GuidanceSession(GuidanceConfig(mode='retrieved_frames', layers='0'), shape)

        def callback(step, *args):
            anchor = next((e for e in indices[0].entries if e.frame == 0), None)
            prior = banks[0].bank.get((0, 'subject'))
            snapshots.append(dict(step=step, anchor=anchor, bank=prior,
                pairs=session.pairs, regional=any(p.regional for p in session.region_plans),
                scale_means=[float(b.bank[(0,'subject')][0].mean())
                             for b in banks if (0,'subject') in b.bank],
                reference_count=sum(len(index.entries) for index in indices)))

        def executor(*args, **kwargs):
            for step in range(20):
                args[5](step, video, video, 20)
            return video

        # Exercise actual image records too: no stage may delete a protected image.
        with tempfile.TemporaryDirectory() as root:
            observer = NativeVisualObserver(tiny_vae_name='auto', max_frames=4,
                min_progress=0, disk_gib=.1, temp_root=root,
                store_observation_images=True,
                observation_settings=ObservationSettings(progressive_resolution=True),
                memory_options=MemoryOptions(protected_frames='0',
                    regions='[{"name":"subject","box":[0,0,1,1]}]'))
            with patch('mosaichunk_h3.native_visual_observer._load_tae_decoder', return_value=decoder), \
                 patch('mosaichunk_h3.native_visual_observer.AdaptiveVisualIndex', Index), \
                 patch('mosaichunk_h3.region_memory.RegionMemory', Bank), \
                 patch('mosaichunk_h3.native_visual_observer.notify_native',
                    side_effect=lambda node, message, **data: statuses.append(data)):
                self.assertIs(observer._sample(executor, None, None, None, None,
                    latent_shapes=[shape, [1,32,2]], guidance=session, callback=callback), video)
            self.assertEqual(list(Path(root).iterdir()), [])
        self.assertEqual(resets, [])
        for step in (7, 14):
            current = snapshots[step]
            self.assertIs(current['anchor'], snapshots[1]['anchor'])
            self.assertIs(current['bank'], snapshots[1]['bank'])
            self.assertTrue(current['bank'][1])
            self.assertTrue(current['pairs'])
            self.assertTrue(current['regional'])
            self.assertIsNotNone(current['anchor'].record)
        self.assertEqual(len(banks), 3)
        self.assertTrue(all(bank.bank == {} for bank in banks))  # Normal final cleanup still applies.
        self.assertEqual(statuses[-1]['reference_preview_sizes'], ((16,24),(12,18),(8,12)))
        self.assertEqual(statuses[-1]['preview_size'], (16,24))
        # Larger-stage RGB is genuinely learned, not just shown in the inspector.
        self.assertGreater(snapshots[15]['scale_means'][-1], .8)
        self.assertLess(snapshots[15]['scale_means'][0], .5)
        self.assertTrue(all(s['reference_count'] <= observer.max_memory_frames for s in snapshots))

    def test_source_drift_is_still_rejected_with_detailed_tracking_previews(self):
        from mosaichunk_h3.region_memory import parse_regions
        decoder = ScaleSensitiveDecoder()
        video = torch.ones(1,24,7,8,12)
        banks = {size: RegionMemory(parse_regions('[{"name":"subject","box":[0,0,1,1]}]'))
                 for size in ((4,6),(8,12))}
        for step in range(2):
            views = decode_observation_views(decoder, video,
                size=(8,12), reference_sizes=((4,6),))
            for size,frames in views.items():
                plans = banks[size].plan(frames, groups=tuple(range(7)), step=step)
                if step:
                    self.assertTrue(plans)
        frozen = banks[(4,6)].bank[(0,'subject')]
        decoder.drift = True
        views = decode_observation_views(decoder, video,
            size=(8,12), reference_sizes=((4,6),))
        # Coarse history still rejects a source whose appearance actually changed.
        bank = banks[(4,6)]
        self.assertFalse(bank.plan(views[(4,6)], groups=tuple(range(7)), step=2))
        self.assertIs(bank.bank[(0,'subject')], frozen)
        self.assertIn('drift', bank.events[0]['reason'])

    def test_temporal_layout_mismatch_is_rejected(self):
        from types import SimpleNamespace
        decoder = SimpleNamespace(decode_video=lambda latent, **kwargs:
            torch.zeros(latent.shape[-1],3,8,8))
        with self.assertRaisesRegex(ValueError, 'temporal/channel'):
            decode_observation_views(decoder, torch.ones(1,24,7,8,12),
                size=(8,12), reference_sizes=((4,6),))

    def test_image_descriptor_capacity_is_shared_across_scales(self):
        memory = ProgressiveReferenceMemory(capacity=3,captures_per_step=2,
            protected_frames=(0,),regions=(),index_factory=AdaptiveVisualIndex,region_factory=RegionMemory)
        deleted = []
        small,medium,large = [memory.view(size) for size in ((4,6),(6,9),(8,12))]
        for view,frame,confidence,record in ((small,0,1.,'anchor'),(small,1,.5,'old'),
                                          (medium,0,1.,'medium'),(large,0,1.,'large')):
            self.assertTrue(memory.put(view,0,frame,torch.zeros(108),confidence,record,on_evict=deleted.append))
        self.assertEqual(deleted,['old'])
        self.assertEqual(sum(len(v.index.entries) for v in memory.views.values()),3)
        self.assertFalse(memory.put(large,1,1,torch.zeros(108),1.,'rejected',on_evict=deleted.append))
        self.assertTrue(any(e.record=='anchor' for e in small.index.entries))

    def test_detailed_proposals_win_without_creating_anchor_target_conflicts(self):
        from mosaichunk_h3.memory_feedback import MemoryReference
        def ref(source,target): return MemoryReference(target,source,0,None,0.,1.)
        detailed,coarse = ref(1,4),ref(0,4)
        merged = merge_references(((detailed,),(coarse,ref(4,5),ref(0,3))),
            groups=tuple(range(6)),count=8,forward_only=True)
        self.assertEqual(merged,[detailed,ref(0,3)])
        self.assertEqual(merge_references(((coarse,),),groups=tuple(range(6)),count=0,forward_only=True),[])

    def test_cancellation_after_full_size_cleans_every_bank_and_image_session(self):
        banks = []
        class Bank(RegionMemory):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                banks.append(self)
        video = torch.ones(1,24,7,8,12)
        session = GuidanceSession(GuidanceConfig(mode='retrieved_frames',layers='0'),list(video.shape))
        def executor(*args,**kwargs):
            for step in range(20):
                args[5](step,video,video,20)
        def callback(step,*args):
            if step == 15:
                raise RuntimeError('test cancellation')
        with tempfile.TemporaryDirectory() as root:
            observer = NativeVisualObserver(tiny_vae_name='auto',max_frames=4,min_progress=0,
                disk_gib=.1,temp_root=root,store_observation_images=True,
                observation_settings=ObservationSettings(progressive_resolution=True),
                memory_options=MemoryOptions(protected_frames='0',
                    regions='[{"name":"subject","box":[0,0,1,1]}]'))
            with patch('mosaichunk_h3.native_visual_observer._load_tae_decoder',return_value=ScaleSensitiveDecoder()), \
                 patch('mosaichunk_h3.region_memory.RegionMemory',Bank), \
                 patch('mosaichunk_h3.native_visual_observer.notify_native'), \
                 self.assertLogs('mosaichunk_h3.native_visual_observer',level='WARNING') as warnings:
                with self.assertRaisesRegex(RuntimeError,'test cancellation'):
                    observer._sample(executor,None,None,None,None,latent_shapes=[list(video.shape),[1,32,2]],
                        guidance=session,callback=callback)
            self.assertEqual(list(Path(root).iterdir()),[])
        self.assertEqual(len(banks),3)
        self.assertTrue(all(not bank.bank for bank in banks))
        self.assertTrue(any('speed/quality trade-off' in line for line in warnings.output))


if __name__ == '__main__':
    unittest.main()

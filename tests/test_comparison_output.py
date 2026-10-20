"""Comparison data cannot escape before sampling or cross unrelated results."""
import unittest
from unittest.mock import patch
import torch
import comfy.utils
from comfy.nested_tensor import NestedTensor
from mosaichunk_h3.comparison_output import ComparisonCapture,MiniMaxH3ComparisonOutputs
from mosaichunk_h3.native_visual_observer import MosaiChunkH3NativeVisualObserver,NativeVisualObserver


class ComparisonOutputTests(unittest.TestCase):
    def fixture(self,outcome='accepted',attempt=True):
        streams=[torch.zeros(1,24,3,4,4),torch.ones(1,32,2,5)]
        packed,shapes=comfy.utils.pack_latents(streams)
        before=torch.zeros(9,3,24,32,dtype=torch.uint8)
        after=torch.full_like(before,200)
        capture=ComparisonCapture(enabled=True,frames=3,resolution=16)
        capture.start(before)
        capture.finish(after if attempt else None,outcome=outcome,result=packed,latent_shapes=shapes,caption=outcome)
        return capture,{'samples':NestedTensor(streams)},before,after

    def test_collector_returns_real_bounded_images_and_retained_result(self):
        for outcome in ('accepted','rejected'):
            capture,samples,_,_=self.fixture(outcome)
            before,after,retained,note=MiniMaxH3ComparisonOutputs().collect(capture,samples)
            self.assertEqual(before.shape,(3,12,16,3))
            self.assertEqual(before.dtype,torch.float32)
            self.assertEqual(before.device.type,'cpu')
            self.assertTrue(torch.equal(before,torch.zeros_like(before)))
            self.assertTrue(torch.allclose(after,torch.full_like(after,200/255)))
            self.assertTrue(torch.equal(retained,after if outcome=='accepted' else before))
            self.assertEqual(note,outcome)

    def test_no_attempt_returns_equal_frames_with_explicit_status(self):
        capture,samples,_,_=self.fixture('skipped',attempt=False)
        before,after,retained,note=capture.read(samples)
        self.assertTrue(torch.equal(before,after))
        self.assertTrue(torch.equal(retained,before))
        self.assertIn('no attempted after image',note)

    def test_disabled_pending_reset_and_wrong_result_fail_explicitly(self):
        capture,samples,_,_=self.fixture()
        other={'samples':NestedTensor([s+.001 for s in samples['samples'].unbind()])}
        with self.assertRaisesRegex(ValueError,'does not match'):capture.read(other)
        capture.reset('cancelled')
        with self.assertRaisesRegex(ValueError,'cancelled'):capture.read(samples)
        self.assertIsNone(capture.before)
        self.assertIsNone(capture.after)
        with self.assertRaisesRegex(ValueError,'Enable Before/After'):
            ComparisonCapture().read(samples)

    def test_audio_changes_and_split_video_do_not_match_the_recording(self):
        capture,samples,_,_=self.fixture()
        video,audio=samples['samples'].unbind()
        for other in (NestedTensor((video,audio+1)),video):
            with self.assertRaisesRegex(ValueError,'does not match'):
                capture.read({'samples':other})

    def test_readers_cannot_mutate_stored_frames_or_original_rgb(self):
        capture,samples,before,after=self.fixture()
        before.fill_(100);after.zero_()
        first=capture.read(samples)
        first[0].fill_(1);first[1].zero_()
        second=capture.read(samples)
        self.assertEqual(float(second[0].max()),0.)
        self.assertGreater(float(second[1].min()),.7)

    def test_lazy_input_orders_sampling_and_rejects_disabled_capture(self):
        node=MiniMaxH3ComparisonOutputs()
        capture,samples,_,_=self.fixture()
        self.assertEqual(node.check_lazy_status(capture),['samples'])
        self.assertEqual(node.check_lazy_status(capture,samples),[])
        with self.assertRaisesRegex(ValueError,'Enable Before/After'):
            node.check_lazy_status(ComparisonCapture())

    def test_invalid_mode_fails_before_model_use(self):
        with self.assertRaisesRegex(ValueError,'Second-pass repair ON'):
            MosaiChunkH3NativeVisualObserver().apply(None,'auto',second_pass_comparison=True,second_pass=False)

    def test_sampling_failure_clears_previous_capture(self):
        capture,samples,_,_=self.fixture()
        observer=NativeVisualObserver(tiny_vae_name='auto',max_frames=2,min_progress=0,disk_gib=.1,
            comparison_capture=capture)
        with patch.object(NativeVisualObserver,'_call',side_effect=RuntimeError('cancelled')):
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                observer(None,None,None,None,None)
        with self.assertRaisesRegex(ValueError,'interruption/error'):
            capture.read(samples)


if __name__=='__main__':unittest.main()

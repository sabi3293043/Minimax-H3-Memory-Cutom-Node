"""No duplicate compatible frame decoding; exact identity, isolation and bounded lifetime."""
from concurrent.futures import ThreadPoolExecutor
import copy
import threading
import sys
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch

from mosaichunk_h3.shared_tiny_vae import SharedTinyVAE,tensor_digest


class FakeDecoder:
    decodes_prefix=False
    latent_channels=24
    upscale_ratio=1

    def __init__(self):
        self.model=torch.nn.Conv2d(24,3,1,bias=False).eval()
        with torch.no_grad():self.model.weight.fill_(.01)
        self.calls=[]

    def decode_video(self,latent,frame_indices=None):
        indices=list(range(latent.shape[2])) if frame_indices is None else list(frame_indices)
        frames=[]
        for index in indices:
            self.calls.append(index)
            frame=self.model(latent[:,:,index])[0].detach().clamp(0,1).mul(255).byte().cpu()
            frames.append(frame)
        return torch.stack(frames)


class SharedTinyTests(unittest.TestCase):
    def video(self):return torch.arange(1*24*4*4*6).reshape(1,24,4,4,6).float()/2304

    def test_second_pass_can_consume_exact_final_preview_without_another_decode(self):
        decoder=FakeDecoder();video=self.video()
        with SharedTinyVAE() as shared:
            shared.register(FakeDecoder);shared.retain_last_callback=True
            with shared.step():decoder.decode_video(video)
            self.assertGreater(shared.bytes,0)
            with shared.completed_step():decoder.decode_video(video.clone())
            self.assertEqual(decoder.calls,[0,1,2,3])
            self.assertEqual(shared.bytes,0)
        self.assertEqual(shared.bytes,0)

    def test_post_inspection_rejects_different_completed_latents_and_cleans_cancellation(self):
        decoder=FakeDecoder();video=self.video()
        shared=SharedTinyVAE()
        with self.assertRaisesRegex(RuntimeError,'cancel'):
            with shared:
                shared.register(FakeDecoder);shared.retain_last_callback=True
                with shared.step():decoder.decode_video(video)
                with shared.completed_step():
                    decoder.decode_video(video+.01)
                    raise RuntimeError('cancel')
        self.assertEqual(len(decoder.calls),8)
        self.assertEqual(shared.bytes,0)

    def test_fingerprint_handles_fp16_channels_last_singleton_weights(self):
        value=torch.ones(3,96,1,1,dtype=torch.float16).to(memory_format=torch.channels_last)
        self.assertEqual(tensor_digest(value),tensor_digest(value.clone(memory_format=torch.contiguous_format)))
        self.assertEqual(tensor_digest(torch.tensor(1.)),tensor_digest(torch.tensor([1.])))

    def test_standalone_player_suppresses_only_its_callback_standard_preview(self):
        class Previewer:
            def decode_latent_to_preview_image(self,*args):return 'normal preview'
        original=Previewer.decode_latent_to_preview_image
        with patch.dict(sys.modules,{'latent_preview':SimpleNamespace(LatentPreviewer=Previewer)}):
            with SharedTinyVAE() as shared:
                shared.suppress_standard_preview()
                self.assertEqual(Previewer().decode_latent_to_preview_image(),'normal preview')
                with shared.step():self.assertIsNone(Previewer().decode_latent_to_preview_image())
        self.assertIs(Previewer.decode_latent_to_preview_image,original)

    def test_preview_first_decodes_only_missing_frames_for_memory(self):
        decoder=FakeDecoder();video=self.video()
        with SharedTinyVAE() as shared:
            shared.register(FakeDecoder)
            with shared.step():
                preview=decoder.decode_video(video,[0,3])
                memory=decoder.decode_video(video)
                self.assertTrue(torch.equal(preview,memory[[0,3]]))
                self.assertEqual(decoder.calls,[0,3,1,2])
                self.assertEqual(shared.reused_frames,2)
                self.assertLessEqual(shared.bytes,shared.max_bytes)
            self.assertEqual(shared.bytes,0)

    def test_memory_first_and_distinct_identical_decoder_instance_share(self):
        first,second=FakeDecoder(),FakeDecoder();video=self.video()
        expected=first.decode_video(video)
        first.calls.clear()
        with SharedTinyVAE() as shared:
            shared.register(FakeDecoder)
            with shared.step():
                memory=first.decode_video(video)
                preview=second.decode_video(video,[3,0])
                self.assertTrue(torch.equal(memory,expected))
                self.assertTrue(torch.equal(preview,expected[[3,0]]))
                self.assertEqual(len(first.calls),4)
                self.assertEqual(second.calls,[])

    def test_consumers_cannot_corrupt_cached_rgb(self):
        decoder=FakeDecoder();video=self.video()
        with SharedTinyVAE() as shared:
            shared.register(FakeDecoder)
            with shared.step():
                frame=decoder.decode_video(video,[0]);expected=frame.clone()
                frame.zero_()
                self.assertTrue(torch.equal(decoder.decode_video(video,[0]),expected))
                self.assertEqual(decoder.calls,[0])

    def test_modified_input_and_weights_never_reuse_stale_pixels(self):
        decoder=FakeDecoder();video=self.video()
        with SharedTinyVAE() as shared:
            shared.register(FakeDecoder)
            with shared.step(),torch.inference_mode():
                decoder.decode_video(video,[0])
                video.add_(.1)
                decoder.decode_video(video,[0])
                decoder.model.weight.mul_(2)
                decoder.decode_video(video,[0])
                self.assertEqual(decoder.calls,[0,0,0])
                self.assertEqual(shared.reused_frames,0)

    def test_different_resolutions_decode_once_each_without_cross_scale_reuse(self):
        decoder=FakeDecoder()
        with SharedTinyVAE() as shared:
            shared.register(FakeDecoder)
            with shared.step():
                for size in ((4,6),(8,12),(4,6),(8,12)):
                    decoder.decode_video(torch.ones(1,24,4,*size))
                self.assertEqual(len(decoder.calls),8)
                self.assertEqual(shared.reused_frames,8)

    def test_cache_does_not_cross_steps_or_unrelated_threads(self):
        decoder=FakeDecoder();video=self.video()
        with SharedTinyVAE() as shared:
            shared.register(FakeDecoder)
            with shared.step():
                decoder.decode_video(video,[0])
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(decoder.decode_video,video,[0]).result()
                self.assertEqual(shared.decoded_frames,1)
            with shared.step():decoder.decode_video(video,[0])
            self.assertEqual(decoder.calls,[0,0,0])

    def test_small_budget_does_not_repeat_work_within_one_request(self):
        decoder=FakeDecoder();video=self.video()
        with SharedTinyVAE(max_bytes=2*3*4*6) as shared:
            shared.register(FakeDecoder)
            with shared.step():
                decoder.decode_video(video)
                decoder.decode_video(video)
                self.assertEqual(len(decoder.calls),6)
                self.assertLessEqual(shared.bytes,shared.max_bytes)

    def test_exception_restores_hooks_and_clears_all_frames(self):
        original=FakeDecoder.decode_video
        decoder=FakeDecoder()
        shared=SharedTinyVAE()
        with self.assertRaisesRegex(RuntimeError,'cancel'):
            with shared:
                shared.register(FakeDecoder)
                with shared.step():
                    decoder.decode_video(self.video())
                    raise RuntimeError('cancel')
        self.assertIs(FakeDecoder.decode_video,original)
        self.assertEqual(shared.bytes,0)

    def test_nested_sessions_restore_the_outer_session_hook(self):
        original=FakeDecoder.decode_video;decoder=FakeDecoder();video=self.video()
        with SharedTinyVAE() as outer:
            outer.register(FakeDecoder)
            hook=FakeDecoder.decode_video
            with outer.step():
                decoder.decode_video(video,[0])
                with SharedTinyVAE() as inner:
                    inner.register(FakeDecoder)
                    with inner.step():decoder.decode_video(video,[0])
                self.assertIs(FakeDecoder.decode_video,hook)
                decoder.decode_video(video,[0])
                self.assertEqual(outer.reused_frames,1)
        self.assertIs(FakeDecoder.decode_video,original)

    def test_concurrent_sampler_sessions_keep_separate_caches(self):
        original=FakeDecoder.decode_video;barrier=threading.Barrier(2)
        def sample(value):
            decoder=FakeDecoder();video=torch.full((1,24,4,4,6),value)
            with SharedTinyVAE() as shared:
                shared.register(FakeDecoder);barrier.wait(timeout=5)
                with shared.step():
                    first=decoder.decode_video(video,[0]);again=decoder.decode_video(video,[0])
                    self.assertTrue(torch.equal(first,again))
                    self.assertEqual(len(decoder.calls),1)
                barrier.wait(timeout=5)
                return first
        with ThreadPoolExecutor(max_workers=2) as pool:
            a,b=[pool.submit(sample,value) for value in (.1,.9)]
            self.assertFalse(torch.equal(a.result(),b.result()))
        self.assertIs(FakeDecoder.decode_video,original)


if __name__=='__main__':unittest.main()

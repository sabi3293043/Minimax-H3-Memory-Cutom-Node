"""Repair routing, output preservation and failure/cancellation contracts."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import torch
import comfy.utils
from comfy.nested_tensor import NestedTensor
from mosaichunk_h3.memory_controls import MemoryOptions
from mosaichunk_h3.native_guidance import GuidanceConfig
from mosaichunk_h3.native_visual_observer import NativeVisualObserver
from mosaichunk_h3.repair_pass import SecondPassSettings,execute_second_pass
from mosaichunk_h3.region_memory import RegionPlan


class Patcher:
    def __init__(self):
        self.model_options={'transformer_options':{}}
        self.root=SimpleNamespace(blocks=[object()])
        self.model=SimpleNamespace(_denoise_mask_conds=lambda *args:None,process_latent_in=lambda x:x)
    def get_model_object(self,name):return self.root
    def clone(self):
        clone=Patcher();clone.model_options=deepcopy(self.model_options);return clone
    def remove_wrappers_with_key(self,*args):pass


class Guider:
    def __init__(self,fail=False):
        self.model_patcher=Patcher();self.model_options=self.model_patcher.model_options
        self.original_conds={'positive':[],'negative':[]};self.cfg=1.;self.calls=[];self.fail=fail
    def sample(self,noise,latent,sampler,sigmas,*,denoise_mask,callback,disable_pbar,seed):
        self.calls.append(len(sigmas)-1)
        if self.fail:raise RuntimeError('synthetic repair failure')
        # Deliberately modify all streams: production must restore unselected values.
        result=NestedTensor([s+.1 for s in latent.unbind()])
        for step in range(len(sigmas)-1):callback(step,result,result,len(sigmas)-1)
        return result


class SecondPassLifecycleTests(unittest.TestCase):
    def run_repair(self,*,accept=True,empty=False,fail=False,callback=None,cfg=1.,sigmas=None,preview=False,comparison=False):
        shape=(1,24,3,4,4);audio_shape=(1,32,2,5)
        clean=[torch.ones(shape)*.2,torch.ones(audio_shape)*.3]
        baseline=comfy.utils.pack_latents(clean)[0]
        guider=Guider(fail);guider.cfg=cfg;original=deepcopy(guider.model_options)
        executor=SimpleNamespace(class_obj=guider)
        observer=NativeVisualObserver(tiny_vae_name='auto',max_frames=2,min_progress=0,disk_gib=.1,
            guidance=GuidanceConfig(mode='manual_frames',layers='0',target_frames='5'),
            node_id='test' if preview else None,
            memory_options=MemoryOptions(preview='auto' if preview else 'off',inspect=True,compare_repair=comparison),second_pass=SecondPassSettings(enabled=True,steps=4))
        from mosaichunk_h3.comparison_output import ComparisonCapture
        observer.comparison_capture=ComparisonCapture(enabled=comparison)
        mask=torch.zeros(1,1,3,4,4);mask[:,:,2,2:,2:]=1
        plan=SimpleNamespace(mask=mask,plans=() if empty else (RegionPlan(0,2,'test',(0,0,1,1),(.5,.5,1,1),1.,0,2),),
            area=0. if empty else float(mask.mean()),severity=.2,
            assess=Mock(side_effect=[(1.,1.),(.8 if accept else 1.1,1.)]))
        decoded=torch.zeros(3,3,16,16,dtype=torch.uint8)
        decoder=SimpleNamespace(decode_video=Mock(return_value=decoded),upscale_ratio=4,decodes_prefix=False)
        events=[];outputs=[]
        def forward(step,x0,x,total):
            outputs.append(x0.clone())
            if callback is not None:callback(step,x0,x,total)
        def first_pass(instance,*args,**kwargs):
            self.assertIsNone(instance.guidance)
            self.assertTrue(instance.defer_observation)
            return baseline
        with patch.object(NativeVisualObserver,'_call',autospec=True,side_effect=first_pass) as primary,\
             patch('mosaichunk_h3.native_visual_observer._load_tae_decoder',return_value=decoder),\
             patch('mosaichunk_h3.repair_pass.build_repair_plan',return_value=plan),\
             patch('mosaichunk_h3.native_preview.send') as preview_send,\
             patch('mosaichunk_h3.native_status.notify_native',side_effect=lambda node,message,**data:events.append(data)):
            try:
                result=execute_second_pass(observer,executor,None,None,None,
                    torch.tensor([1.,.5,0.]) if sigmas is None else sigmas,None,forward,True,44,
                    latent_shapes=[shape,audio_shape],shared=None)
            finally:
                self.assertEqual(guider.model_options,original)
        self.preview_events=[call.kwargs for call in preview_send.call_args_list]
        self.decoder_calls=decoder.decode_video.call_count
        self.capture=observer.comparison_capture
        return result,baseline,mask,guider.calls,outputs,events

    def test_before_after_preview_uses_only_existing_inspection_and_evaluation_decodes(self):
        for accept in (True,False):
            self.run_repair(accept=accept,preview=True,comparison=True)
            self.assertEqual(self.decoder_calls,2)
            self.assertEqual(len(self.preview_events),2)
            self.assertIsNone(self.preview_events[0]['comparison']['after'])
            self.assertEqual(self.preview_events[-1]['comparison']['outcome'],'accepted' if accept else 'rejected')
            self.assertTrue(self.preview_events[-1]['comparison']['before'])
            self.assertTrue(self.preview_events[-1]['comparison']['after'])
        for options,outcome in ((dict(empty=True),'skipped'),(dict(fail=True),'failed')):
            self.run_repair(preview=True,comparison=True,**options)
            self.assertEqual(self.decoder_calls,1)
            self.assertEqual(self.preview_events[-1]['comparison']['outcome'],outcome)
            self.assertIsNone(self.preview_events[-1]['comparison']['after'])

    def test_comparison_off_keeps_only_the_retained_result_preview(self):
        for accept in (True,False):
            self.run_repair(accept=accept,preview=True)
            self.assertEqual(self.decoder_calls,2)
            self.assertEqual(len(self.preview_events),1)
            self.assertNotIn('comparison',self.preview_events[0])
            self.assertTrue(self.preview_events[0]['frames'])
        self.run_repair(comparison=True,preview=False)
        self.assertEqual(self.preview_events,[])

    def test_comparison_outputs_remain_available_with_builtin_player_off(self):
        output,*_=self.run_repair(comparison=True,preview=False)
        from mosaichunk_h3.comparison_output import MiniMaxH3ComparisonOutputs
        streams=comfy.utils.unpack_latents(output,[(1,24,3,4,4),(1,32,2,5)])
        before,after,retained,note=MiniMaxH3ComparisonOutputs().collect(self.capture,{'samples':NestedTensor(streams)})
        self.assertEqual(before.shape,after.shape)
        self.assertEqual(before.shape[-1],3)
        self.assertEqual(self.decoder_calls,2)
        self.assertEqual(self.preview_events,[])

    def test_masked_output_and_audio_are_preserved_and_x0_agrees(self):
        output,baseline,mask,calls,x0,events=self.run_repair()
        shapes=[(1,24,3,4,4),(1,32,2,5)]
        before=comfy.utils.unpack_latents(baseline,shapes);after=comfy.utils.unpack_latents(output,shapes)
        self.assertTrue(torch.equal(before[1],after[1]))
        outside=~mask.expand_as(before[0]).bool()
        self.assertTrue(torch.equal(before[0][outside],after[0][outside]))
        self.assertFalse(torch.equal(before[0],after[0]))
        self.assertTrue(torch.equal(x0[-1],output))
        self.assertEqual(calls,[4])
        self.assertTrue(events[-1]['second_pass']['accepted'])
        self.assertIn('regional_corrections',events[-1]['guidance_details'])

    def test_rejected_result_restores_both_outputs(self):
        output,baseline,_,calls,x0,events=self.run_repair(accept=False)
        self.assertTrue(torch.equal(output,baseline))
        self.assertTrue(torch.equal(x0[-1],baseline))
        self.assertFalse(events[-1]['second_pass']['accepted'])

    def test_failed_repair_restores_original_and_reports_error(self):
        output,baseline,_,calls,x0,events=self.run_repair(fail=True)
        self.assertTrue(torch.equal(output,baseline))
        self.assertTrue(torch.equal(x0[-1],baseline))
        self.assertIn('synthetic repair failure',events[-1]['second_pass']['error'])

    def test_no_candidates_run_no_extra_sampler_steps(self):
        output,baseline,_,calls,x0,events=self.run_repair(empty=True)
        self.assertTrue(torch.equal(output,baseline))
        self.assertEqual(calls,[])
        self.assertEqual(events[-1]['second_pass']['steps'],0)

    def test_callback_failures_are_not_hidden_as_repair_failures(self):
        def cancel(*args):raise RuntimeError('callback cancelled')
        with self.assertRaisesRegex(RuntimeError,'callback cancelled'):
            self.run_repair(callback=cancel)

    def test_invalid_cfg_and_incomplete_schedule_fail_before_the_first_pass(self):
        for options in (dict(cfg=2.),dict(sigmas=torch.tensor([1.,.2]))):
            with self.subTest(options=options),self.assertRaises(ValueError):
                self.run_repair(**options)


if __name__=='__main__':unittest.main()

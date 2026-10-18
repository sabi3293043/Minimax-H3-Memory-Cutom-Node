"""CPU tests for native H3 condition geometry, prefix storage and retrieval.

Set COMFYUI_PATH (or run with ComfyUI on sys.path) to exercise native kernels.
No checkpoints, GPU, downloads or generated media are required.
"""
import os
from pathlib import Path
import sys
import unittest

import torch

if os.environ.get('COMFYUI_PATH'):
    sys.path.insert(0, os.environ['COMFYUI_PATH'])
try:
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
except ImportError:
    raise unittest.SkipTest('Native H3 tests need ComfyUI on sys.path')

from comfy.ldm.minimax.model import PackedLayout
from comfy.nested_tensor import NestedTensor
from mosaichunk_h3.contracts import parse_latent, parse_conditioning, ContractError
from mosaichunk_h3.image_context import ImageContext, ConditionedLayout, split_conditioning, prefill_conditioned
from mosaichunk_h3.layout import stereo_chunk_indices
from mosaichunk_h3.memory import MosaiChunkKVCache, MemorySettings
from mosaichunk_h3.routing.selector import Selector
from mosaichunk_h3.consistency import sample_streaming, SamplerConfig


def latent():
    return {'samples': NestedTensor((torch.zeros(1,24,27,4,4), torch.zeros(1,32,2,150)))}


def text():
    return [[torch.ones(1,5,16)*.1, {'minimax_token_tags': torch.tensor([1,0,1,2,1])}]]


def reference(value=.2):
    return {'kind':'image', 'latent_h':4, 'latent_w':6, 'latent':torch.ones(1,24,1,4,6)*value}


def tiny_model():
    import comfy.ops
    from mosaichunk_h3.causal_model import RavenCausalMiniMaxH3Model
    cfg=dict(hidden_size=32,num_layers=3,token_refiner_num_layers=1,num_attention_heads=2,
             attention_head_dim=24,ffn_hidden_size=64,latents_dim=24,audio_latents_dim=32,
             text_dim=16,timestep_input_dim=16,time_embed_hidden_size=32,time_embed_dim=8,
             rope_inv_freq_len=4,adaln_curve_grid=1025)
    model=RavenCausalMiniMaxH3Model(**cfg,dtype=torch.float32,device='cpu',operations=comfy.ops.disable_weight_init)
    rng=torch.Generator().manual_seed(12)
    state={k:torch.randn(v.shape,generator=rng)*.05 for k,v in model.state_dict().items()}
    state['rope.inv_freq']=1/(10000**(torch.arange(4)/4))
    model.load_state_dict(state)
    return model.eval().requires_grad_(False)


class ImageContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model=tiny_model()

    def test_native_positions_for_combined_refs_and_first_last_frames(self):
        request=parse_latent(latent())
        kfs=({'resolved_frame_index':0,'latent':torch.zeros(1,24,1,4,4)},
             {'resolved_frame_index':89,'latent':torch.ones(1,24,1,4,4)})
        refs=(reference(), {'kind':'audio','ref_audio_t':7,'audio_latent':torch.zeros(1,32,2,7)})
        c=ImageContext(kfs,refs)
        base=request.layout(5)
        layout=ConditionedLayout(base,c)
        native=PackedLayout(5,27,4,4,150,keyframes=kfs,refs=refs)
        torch.testing.assert_close(layout.prefix_positions,native.position_ids[:layout.prefix_rows],rtol=0,atol=0)
        self.assertEqual(layout.target_origin,13.)
        for index in range(layout.num_chunks):
            chunk=layout.chunk(index)
            a=next(a for a,b,k in native.segments if k=='audio')
            v=next(a for a,b,k in native.segments if k=='video')
            ai=stereo_chunk_indices(150,chunk.audio_start,chunk.audio_stop)
            expected=torch.cat((native.position_ids[a+ai],native.position_ids[v+chunk.video_start*4:v+chunk.video_stop*4]))
            torch.testing.assert_close(layout.chunk_position_ids(index),expected,atol=1e-12,rtol=0)

    def test_validation_preserves_text_and_rejects_bad_reference_shapes(self):
        request=parse_latent(latent())
        cond=text();cond[0][1]['minimax_refs']=[reference()]
        clean, context=split_conditioning(cond,request)
        self.assertTrue(context.active)
        self.assertEqual(parse_conditioning(clean).text_len,5)
        self.assertIn('minimax_refs',cond[0][1])
        bad=reference();bad['latent_w']=8
        with self.assertRaisesRegex(ContractError,'metadata'):
            split_conditioning([[cond[0][0],{'minimax_refs':[bad]}]],request)

    def test_varying_condition_timesteps_cannot_be_cached_as_fixed_prefix(self):
        context=ImageContext(refs=(reference(),),visual_aug=.5)
        with self.assertRaisesRegex(ContractError,'conditioning timestep changes'):
            context.validate_schedule(SamplerConfig(steps=4))

    def test_first_and_last_frames_each_change_permanent_prefix(self):
        request=parse_latent(latent())
        prefix_keys=[]
        for first,last in ((.1,.2),(.3,.2),(.1,.4)):
            context=ImageContext(({'resolved_frame_index':0,'latent':torch.full((1,24,1,4,4),first)},
                                  {'resolved_frame_index':89,'latent':torch.full((1,24,1,4,4),last)}))
            layout=ConditionedLayout(request.layout(5),context)
            cache=MosaiChunkKVCache(3,layout=layout,selector=Selector(d_in=48,d=16),
                inv_freq=self.model.rope.inv_freq,settings=MemorySettings(sections=4,descriptor_layers=(0,1,2)))
            before=[item['latent'].clone() for item in context.keyframes]
            prefill_conditioned(self.model,context,parse_conditioning(text()),layout,cache,{},torch.float32,'cpu',55)
            self.assertEqual(cache._records[0].rows,13)
            self.assertFalse(cache.history)
            prefix_keys.append(cache._records[0].keys[0].clone())
            for original,item in zip(before,context.keyframes):
                torch.testing.assert_close(original,item['latent'],rtol=0,atol=0)
            cache.reset()
        self.assertFalse(torch.equal(prefix_keys[0],prefix_keys[1]))
        self.assertFalse(torch.equal(prefix_keys[0],prefix_keys[2]))

    def test_reference_is_retained_while_generated_history_is_retrieved(self):
        import comfy.model_patcher
        class Base(torch.nn.Module):
            def __init__(self,dit):
                super().__init__();self.diffusion_model=dit
        patcher=comfy.model_patcher.ModelPatcher(Base(self.model),torch.device('cpu'),torch.device('cpu'))
        caches=[]
        selector=Selector(d_in=48,d=16)
        def factory(num_layers,layout,config):
            cache=MosaiChunkKVCache(num_layers,layout=layout,selector=selector,inv_freq=self.model.rope.inv_freq,
                settings=MemorySettings(sections=4,descriptor_layers=(0,1,2)))
            caches.append(cache);return cache
        outputs=[]
        for value in (.2,.8):
            result=sample_streaming(model=patcher,positive=text(),latent=latent(),
                config=SamplerConfig(steps=2,seed=9,sink=1,window=3,kv_cache_storage='cpu'),
                image_context=ImageContext(refs=(reference(value),)),cache_factory=factory,
                load_models=lambda *a,**kw:None)
            self.assertEqual(result.kv_cache['retrievals'],3)
            self.assertEqual(result.layout.prefix_rows,11)
            self.assertEqual(result.layout.num_chunks,6)
            outputs.append(result.latent['samples'].unbind()[0])
            self.assertFalse(caches[-1].history)
            self.assertEqual(caches[-1].committed_chunks,0)
        self.assertFalse(torch.equal(*outputs))

    def test_reference_video_and_audio_share_native_prefix(self):
        request=parse_latent(latent())
        ref={'kind':'video_audio','latent':torch.ones(1,24,2,4,4)*.3,
             'latent_t':2,'latent_h':4,'latent_w':4,'ref_audio_t':8,
             'audio_latent':torch.ones(1,32,2,8)*.2}
        cond=text();cond[0][1]['minimax_refs']=[ref]
        clean,context=split_conditioning(cond,request)
        layout=ConditionedLayout(request.layout(5),context)
        cache=MosaiChunkKVCache(3,layout=layout,selector=Selector(d_in=48,d=16),
            inv_freq=self.model.rope.inv_freq,settings=MemorySettings(sections=4,descriptor_layers=(0,1,2)))
        prefill_conditioned(self.model,context,parse_conditioning(clean),layout,cache,{},torch.float32,'cpu',4)
        self.assertEqual(cache._records[0].rows,29)
        cache.reset()

    def test_cancellation_after_image_prefill_releases_prefix(self):
        import comfy.model_patcher
        from mosaichunk_h3.consistency import SamplingCancelled
        class Base(torch.nn.Module):
            def __init__(self,dit):
                super().__init__();self.diffusion_model=dit
        patcher=comfy.model_patcher.ModelPatcher(Base(self.model),torch.device('cpu'),torch.device('cpu'))
        caches=[]
        def factory(num_layers,layout,config):
            cache=MosaiChunkKVCache(num_layers,layout=layout,selector=Selector(d_in=48,d=16),
                inv_freq=self.model.rope.inv_freq,settings=MemorySettings(sections=4,descriptor_layers=(0,1,2)))
            caches.append(cache);return cache
        with self.assertRaises(SamplingCancelled):
            sample_streaming(model=patcher,positive=text(),latent=latent(),
                config=SamplerConfig(steps=2,seed=9,sink=1,window=3,kv_cache_storage='cpu'),
                image_context=ImageContext(refs=(reference(),)),cache_factory=factory,
                load_models=lambda *a,**kw:None,cancel_check=lambda point:point=='after_prefill')
        self.assertEqual(caches[0].committed_chunks,0)
        self.assertFalse(caches[0].history)


if __name__=='__main__':
    unittest.main()

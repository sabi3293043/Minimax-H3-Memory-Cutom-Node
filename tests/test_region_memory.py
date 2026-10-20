"""Appearance references, region isolation, pinned eviction and honest diagnostics."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from mosaichunk_h3.region_memory import parse_regions, RegionMemory, RegionPlan, grid_ids, move_region
from mosaichunk_h3.native_guidance import GuidanceConfig, GuidanceSession
from mosaichunk_h3.visual_memory_index import AdaptiveVisualIndex
from mosaichunk_h3.memory_inspector import MemoryInspector, write_report
from mosaichunk_h3.memory_controls import MemoryOptions


REGIONS = '[{"name":"trousers","box":[0,0.5,1,1]}]'


class RegionTests(unittest.TestCase):
    def test_rejects_ambiguous_or_unbounded_regions(self):
        for value in ('{}','[{}]','[1]', '[{"name":"x","box":[0,0,2,1]}]',
                      '[{"name":"x","box":[0,0,1,1],"extra":1}]'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_regions(value)
        self.assertEqual(len(parse_regions(REGIONS)),1)

    def test_roi_mapping_never_selects_outside_lower_half(self):
        self.assertEqual(grid_ids((0,.5,1,1),4,4).tolist(),list(range(8,16)))
        mask=torch.zeros(4,4);mask[3]=1
        self.assertEqual(grid_ids((0,.5,1,1),4,4,mask=mask).tolist(),list(range(12,16)))

    def test_tracking_preserves_lower_half_when_mask_covers_whole_person(self):
        box=move_region((.2,.5,.8,1),(.2,0,.8,1),(.3,.1,.9,.9))
        for actual,expected in zip(box,(.3,.5,.9,.9)):
            self.assertAlmostEqual(actual,expected)

    def test_bank_waits_for_stability_and_withholds_drifting_source(self):
        bank=RegionMemory(parse_regions(REGIONS),protected=(0,))
        frames=torch.full((3,3,8,8),.4)
        self.assertFalse(bank.plan(frames,groups=(0,1,2),step=0))
        plans=bank.plan(frames,groups=(0,1,2),step=1)
        self.assertEqual({p.target for p in plans},{1,2})
        frozen=bank.bank[(0,'trousers')][0].clone()
        frames[0]=1.
        self.assertFalse(bank.plan(frames,groups=(0,1,2),step=2))
        self.assertTrue(torch.equal(bank.bank[(0,'trousers')][0],frozen))
        self.assertIn('drift',bank.events[0]['reason'])

    def test_hidden_or_untracked_frames_never_receive_a_tracking_plan(self):
        bank=RegionMemory(parse_regions(REGIONS))
        frames=torch.full((3,3,8,8),.4)
        bank.plan(frames,groups=(0,1,2),step=0)
        records={(0,'trousers'):dict(visible=True,box=[0,.5,1,1],confidence=1,mask=[[1]*4]*4),
                 (1,'trousers'):dict(visible=False,box=[0,.5,1,1],confidence=0,mask=[[0]*4]*4)}
        self.assertFalse(bank.plan(frames,groups=(0,1,2),step=1,tracked=records))

    def test_each_region_receives_slots_when_background_has_many_matches(self):
        regions=parse_regions('[{"name":"subject","box":[0,0,0.5,1]},{"name":"background","box":[0.5,0,1,1]}]')
        bank=RegionMemory(regions)
        frames=torch.full((12,3,8,8),.4)
        groups=tuple(range(12))
        bank.plan(frames,groups=groups,step=0)
        plans=bank.plan(frames,groups=groups,step=1)
        self.assertEqual(len(plans),8)
        self.assertEqual(sum(p.name=='subject' for p in plans),4)
        self.assertEqual(sum(p.name=='background' for p in plans),4)

    def test_protected_reference_survives_eviction_and_replacement(self):
        index=AdaptiveVisualIndex(capacity=2,protected_frames=(0,))
        evicted=[]
        for step,frame,confidence,record in [(0,0,.8,'anchor'),(0,1,.9,'other'),
                                             (1,2,1.,'new'),(2,0,1.,'drifted')]:
            index.put(step,frame,torch.ones(108)*frame,confidence,record,on_evict=evicted.append)
        self.assertIn('anchor',[e.record for e in index.entries])
        self.assertNotIn('drifted',[e.record for e in index.entries])
        self.assertNotIn('anchor',evicted)

    def test_region_attention_changes_only_matching_target_roi(self):
        session=GuidanceSession(GuidanceConfig(mode='manual_frames',layers='0',
            target_frames='5',max_anchor_tokens=8,inspect_memory=True),[1,24,3,4,4])
        session.set_region_plans((RegionPlan(0,2,'trousers',(0,.5,1,1),(0,.5,1,1),1.,0,2),))
        q,k,v=[torch.randn(1,2,16,8) for _ in range(3)]
        v[:,:,12:16]=v[:,:,4:8]+.0001
        layout=SimpleNamespace(segments=[(0,2,'text'),(2,4,'audio'),(4,16,'video')])
        def backend(q,k,v,heads,**kwargs):
            return F.scaled_dot_product_attention(q,k,v).transpose(1,2).flatten(2)
        original=backend(q,k,v,2)
        output=session.attention(backend,q,k,v,2,skip_reshape=True,
            transformer_options={'block_index':0,'minimax_h3_layout':layout})
        self.assertTrue(torch.equal(original[:,:14],output[:,:14]))
        self.assertFalse(torch.equal(original[:,14:],output[:,14:]))
        self.assertGreater(session.diagnostics()['decisions'][0]['corrected'],0)
        self.assertTrue(all(cell>=128 for cell in session.heatmap['2']))

    def test_inspector_is_bounded_and_disabled_path_encodes_nothing(self):
        frames=torch.zeros(2,3,8,8)
        with patch('PIL.Image.fromarray') as image:
            self.assertIsNone(MemoryInspector().observation(frames,(),step=0))
        image.assert_not_called()
        inspector=MemoryInspector(enabled=True)
        for step in range(40): inspector.observation(frames,(),step=step)
        self.assertEqual(len(inspector.history),32)
        self.assertEqual(inspector.history[-1]['plans'][0]['confidence'],0)

    def test_diagnostic_snapshots_survive_execution_cleanup(self):
        session=GuidanceSession(GuidanceConfig(),[1,24,3,4,4])
        session.heatmap={'2':[130,131]}
        snapshot=session.diagnostics()
        session.close()
        self.assertEqual(snapshot['heatmap'],{'2':[130,131]})

    def test_reports_contain_versions_settings_and_no_image_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path=write_report(directory,settings={'strength':.1},seed=3,metrics={'corrections':0})
            report=json.loads(Path(path).read_text())
            self.assertEqual(report['seed'],3)
            self.assertEqual(len(report['implementation_sha256']),64)
            self.assertNotIn('images',report)
            self.assertFalse(list(Path(directory).glob('*.partial')))

    def test_optional_tracking_accepts_empty_regions_for_automatic_proposals(self):
        MemoryOptions().validate()
        self.assertTrue(MemoryOptions(occlusion=True).validate().occlusion)

    def test_regional_priority_keeps_global_corrections_outside_the_box(self):
        torch.manual_seed(3)
        session=GuidanceSession(GuidanceConfig(mode='manual_frames',layers='0',
            target_frames='5',max_anchor_tokens=8),[1,24,3,4,4])
        global_plan=RegionPlan(0,2,'whole frame',(0,0,1,1),(0,0,1,1),1.,0,2,regional=False)
        priority=RegionPlan(0,2,'outfit',(0,.5,1,1),(0,.5,1,1),1.,0,2)
        plans=session.publish_plans((global_plan,),(priority,))
        self.assertEqual(len(plans),2)
        q,k,v=[torch.randn(1,2,16,8) for _ in range(3)]
        v[:,:,12:16]=v[:,:,4:8]+.0001
        layout=SimpleNamespace(segments=[(0,2,'text'),(2,4,'audio'),(4,16,'video')])
        def backend(q,k,v,heads,**kwargs):
            return F.scaled_dot_product_attention(q,k,v).transpose(1,2).flatten(2)
        normal=backend(q,k,v,2)
        output=session.attention(backend,q,k,v,2,skip_reshape=True,
            transformer_options={'block_index':0,'minimax_h3_layout':layout})
        self.assertTrue(torch.equal(output[:,:12],normal[:,:12]))
        self.assertFalse(torch.equal(output[:,12:14],normal[:,12:14]))
        self.assertFalse(torch.equal(output[:,14:],normal[:,14:]))
        self.assertGreater(session.global_corrections,0)
        self.assertGreater(session.regional_corrections,0)

    def test_empty_regional_results_restore_global_plan(self):
        session=GuidanceSession(GuidanceConfig(),[1,24,3,4,4])
        full=RegionPlan(0,2,'whole frame',(0,0,1,1),(0,0,1,1),1.,0,2,regional=False)
        roi=RegionPlan(0,2,'outfit',(0,.5,1,1),(0,.5,1,1),1.,0,2)
        session.publish_plans((full,),(roi,))
        session.publish_plans((full,),())
        self.assertEqual(session.pairs,((0,2),))
        self.assertFalse(session.region_plans)


if __name__=='__main__':unittest.main()

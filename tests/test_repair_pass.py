from dataclasses import replace
import unittest
import torch

from mosaichunk_h3.repair_pass import SecondPassSettings,build_repair_plan,repair_sigmas
from mosaichunk_h3.region_memory import Region


class RepairTests(unittest.TestCase):
    def frames(self):
        image = torch.linspace(.4,.6,96)[None,None,:].expand(3,96,96).clone()
        frames = image[None].repeat(7,1,1,1)
        # Preserve grayscale structure while changing a localized color patch.
        frames[5,0,64:96,0:32] += .25
        frames[5,1,64:96,0:32] -= .25
        return frames

    def test_local_color_drift_can_propose_a_bounded_mask_with_immutable_sources(self):
        plan = build_repair_plan(self.frames(),(1,24,7,12,12),SecondPassSettings())
        self.assertTrue(plan.plans)
        self.assertGreater(plan.area,0)
        self.assertLessEqual(plan.area,.20)
        self.assertEqual(int(plan.mask[:,:,0].sum()),0)
        self.assertTrue(all(p.source==0 and p.target==5 for p in plan.plans))

    def test_consistent_video_and_global_exposure_shift_are_not_repaired(self):
        frames = self.frames();frames[5]=frames[0]
        self.assertFalse(build_repair_plan(frames,(1,24,7,12,12),SecondPassSettings()).plans)
        frames[5] += .25
        self.assertFalse(build_repair_plan(frames,(1,24,7,12,12),SecondPassSettings()).plans)

    def test_protected_targets_and_area_limit_are_honored(self):
        frames = self.frames()
        plan = build_repair_plan(frames,(1,24,7,12,12),SecondPassSettings(),protected=(5,))
        self.assertFalse(plan.plans)
        plan = build_repair_plan(frames,(1,24,7,12,12),SecondPassSettings(max_area=.01))
        self.assertLessEqual(plan.area,.01)

    def test_temporal_decoder_frames_map_to_native_latent_groups(self):
        frames=self.frames()
        frames=torch.cat((frames,frames[:2]),dim=0)
        plan=build_repair_plan(frames,(1,24,3,12,12),SecondPassSettings())
        self.assertTrue(plan.plans)
        self.assertTrue(all(p.target==2 and p.target_frame==5 for p in plan.plans))
        self.assertEqual(int(plan.mask[:,:,:2].sum()),0)

    def test_persistent_drift_is_not_truncated_to_eight_isolated_frames(self):
        original=self.frames()
        frames=original[:1].repeat(33,1,1,1)
        frames[10:30]=original[5]
        plan=build_repair_plan(frames,(1,24,33,12,12),SecondPassSettings())
        self.assertEqual({p.target for p in plan.plans},set(range(10,30)))
        self.assertTrue(all(bool(plan.mask[0,0,i].any()) for i in range(10,30)))

    def test_multiple_named_regions_can_be_repaired_in_one_target(self):
        frames=self.frames()
        frames[5,2,0:32,64:96] += .25
        frames[5,1,0:32,64:96] -= .25
        regions=(Region('lower',(0,2/3,1/3,1)),Region('upper',(2/3,0,1,1/3)))
        plan=build_repair_plan(frames,(1,24,7,12,12),SecondPassSettings(),regions=regions)
        self.assertEqual(len(plan.plans),2)
        self.assertEqual({p.target for p in plan.plans},{5})

    def test_appearance_metric_improves_when_reference_colors_are_restored(self):
        frames = self.frames()
        plan = build_repair_plan(frames,(1,24,7,12,12),SecondPassSettings())
        before,structure = plan.assess(frames)
        frames[5]=frames[0]
        after,new_structure = plan.assess(frames)
        self.assertGreater(before,.1)
        self.assertEqual(after,0.)
        self.assertGreaterEqual(new_structure,structure-.02)

    def test_moving_geometry_without_structural_agreement_is_withheld(self):
        generator=torch.Generator().manual_seed(4)
        frames=torch.rand(7,3,96,96,generator=generator)
        self.assertFalse(build_repair_plan(frames,(1,24,7,12,12),SecondPassSettings()).plans)

    def test_tracked_hidden_or_untrusted_targets_cannot_be_repaired(self):
        frames=self.frames();region=Region('subject',(0,0,1,1))
        records={(i,'subject'):dict(trusted=i!=5,visible=True,confidence=1.,
            box=[0,0,1,1],mask=[[1]*32]*32) for i in range(7)}
        self.assertFalse(build_repair_plan(frames,(1,24,7,12,12),SecondPassSettings(),
            regions=(region,),tracked=records).plans)

    def test_manual_budget_is_exact_and_auto_is_bounded_not_a_correction_count(self):
        self.assertEqual(SecondPassSettings(steps=6).budget(area=.1,severity=.2),6)
        auto=SecondPassSettings(automatic=True,max_steps=8)
        self.assertEqual(auto.budget(area=0,severity=1),0)
        small=auto.budget(area=.01,severity=.1)
        large=auto.budget(area=.5,severity=1)
        self.assertLessEqual(small,large)
        self.assertLessEqual(large,8)

    def test_repair_sigmas_are_a_bounded_low_noise_tail_and_finish_at_zero(self):
        schedule=repair_sigmas(torch.tensor([1.,.8,.5,.2,.1,0.]),6,.25)
        self.assertEqual(len(schedule),7)
        self.assertEqual(float(schedule[0]),.25)
        self.assertEqual(float(schedule[-1]),0.)
        self.assertTrue(bool((schedule[1:]<=schedule[:-1]).all()))
        for values in ([1.,.2],[0.,0.],[1.,float('nan'),0.],[.2,.5,0.]):
            with self.subTest(values=values),self.assertRaises(ValueError):
                repair_sigmas(torch.tensor(values),6,.25)

    def test_invalid_repair_controls_fail_early(self):
        for options in (dict(steps=1),dict(steps=True),dict(max_steps=33),dict(strength=0),
                        dict(strength=float('nan')),dict(threshold=0),dict(max_area=1)):
            with self.subTest(options=options),self.assertRaises(ValueError):
                SecondPassSettings(**options).validate()


if __name__=='__main__':unittest.main()

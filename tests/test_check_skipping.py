import unittest
from unittest.mock import Mock
import torch

from mosaichunk_h3.check_skipping import CheckSkipping
from mosaichunk_h3.observation_policy import ObservationSettings, ObservationPolicy


class CheckSkippingTests(unittest.TestCase):
    def test_disabled_preserves_existing_schedule(self):
        settings = ObservationSettings(schedule='adaptive',max_interval=5)
        resolved,interval = CheckSkipping().resolve(settings,2)
        self.assertIs(resolved,settings)
        self.assertEqual(interval,2)

    def test_manual_skip_count_means_checks_between_observations(self):
        settings,interval = CheckSkipping(True,False,1).resolve(ObservationSettings(),1)
        policy = ObservationPolicy(settings,min_progress=.25,interval=interval)
        video = torch.ones(1,24,7,8,8)
        self.assertEqual([i for i in range(4,12) if policy.decide(video,step=i,total=20).observe],
                         [4,6,8,10])

    def test_auto_warms_up_then_skips_stable_checks_with_a_hard_gap_limit(self):
        settings,interval = CheckSkipping(True,True,2).resolve(ObservationSettings(),1)
        policy = ObservationPolicy(settings,min_progress=0,interval=interval)
        video = torch.ones(1,24,7,8,8)
        observed=[]
        for i in range(10):
            decision = policy.decide(video,step=i,total=20)
            if decision.observe:
                observed.append(i);policy.observed(decision,step=i)
        self.assertEqual(observed,[0,1,2,5,8])

    def test_auto_detects_a_local_change_in_a_late_temporal_latent(self):
        settings,interval = CheckSkipping(True,True,4).resolve(ObservationSettings(),1)
        policy = ObservationPolicy(settings,min_progress=0,interval=interval)
        video = torch.ones(1,24,129,8,8)
        for i in range(3):
            policy.observed(policy.decide(video,step=i,total=20),step=i)
        video[:,:,127,7,7] += .1
        self.assertTrue(policy.decide(video,step=3,total=20).observe)

    def test_invalid_counts_fail_before_sampling(self):
        for count in (0,9,True,1.5,'2'):
            with self.subTest(count=count),self.assertRaises(ValueError):
                CheckSkipping(skip_count=count).validate()


if __name__=='__main__':unittest.main()

import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mosaichunk_h3.memory_feedback import retrieve_step_references
from mosaichunk_h3.visual_memory_index import StoredObservation


class FeedbackTests(unittest.TestCase):
    def test_forward_candidates_are_filtered_before_budget(self):
        q = torch.tensor([[.1], [.9], [.9], [.9], [.11]])
        past = [StoredObservation(0, 4, .9, torch.tensor([.1]), 'reverse'),
                StoredObservation(0, 0, .9, torch.tensor([.12]), 'earlier')]
        diagnostic = retrieve_step_references(q, past, step=1, max_references=1)
        self.assertGreater(diagnostic[0].source_frame, diagnostic[0].query_frame)
        got = retrieve_step_references(q, past, step=1, max_references=1, forward_only=True)
        self.assertEqual([(r.source_frame, r.query_frame) for r in got], [(0, 4)])

    def test_same_latent_matches_do_not_spend_guidance_budget(self):
        q = torch.full((7, 1), .9)
        q[3] = .1
        q[5] = .15
        past = [StoredObservation(0, 1, .9, torch.tensor([.1]), 'same-latent'),
                StoredObservation(0, 0, .9, torch.tensor([.13]), 'valid')]
        got = retrieve_step_references(q, past, step=1, max_references=1,
            forward_only=True, frame_groups=[0, 1, 1, 1, 1, 2, 3])
        self.assertEqual([(r.source_frame, r.query_frame) for r in got], [(0, 5)])

    def test_anchor_conflicts_are_replaced_before_budget_is_full(self):
        q = torch.full((7, 1), .9)
        q[2], q[4], q[6] = .31, .1, .32
        past = [StoredObservation(0, 0, .9, torch.tensor([.3]), 'first'),
                StoredObservation(0, 2, .9, torch.tensor([.1]), 'second')]
        got = retrieve_step_references(q, past, step=1, max_references=2, forward_only=True)
        self.assertEqual([(r.source_frame, r.query_frame) for r in got], [(2, 4), (0, 6)])
        self.assertFalse({r.source_frame for r in got} & {r.query_frame for r in got})

    def test_grouped_targets_are_unique_and_frame_map_is_validated(self):
        q = torch.zeros(7, 1)
        past = [StoredObservation(0, 0, .9, torch.zeros(1), 'anchor')]
        got = retrieve_step_references(q, past, step=1, max_references=4,
            forward_only=True, frame_groups=[0, 1, 1, 1, 1, 2, 3])
        self.assertEqual([r.query_frame for r in got], [2, 5, 6])
        for groups in ([0], [0, 1, 0, 1, 2, 3, 4], [0, 1, 1, 1, 1, 2, -1]):
            with self.assertRaises(ValueError):
                retrieve_step_references(q, past, step=1, forward_only=True, frame_groups=groups)

    def test_past_cross_frame_retrieval_and_rank(self):
        q = torch.ones((6, 108))
        q[5] = 0.1
        past = StoredObservation(1, 0, 0.9, torch.full((108,), 0.11), "ref0")
        future = StoredObservation(3, 2, 1.0, torch.full((108,), 0.1), "future")
        same = StoredObservation(0, 5, 1.0, torch.full((108,), 0.1), "same")
        got = retrieve_step_references(q, [past, future, same], step=2, max_distance=0.1)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].query_frame, 5)
        self.assertEqual(got[0].source_record, "ref0")
        self.assertAlmostEqual(got[0].distance, 0.01, places=5)

    def test_budget_confidence_and_guardrails(self):
        q = torch.zeros((12, 108))
        past = [
            StoredObservation(0, 0, 0.1, torch.zeros(108), "low"),
            StoredObservation(0, 1, 0.9, torch.zeros(108), "good"),
        ]
        got = retrieve_step_references(q, past, step=2, max_references=3, min_temporal_gap=2)
        self.assertEqual(len(got), 3)
        self.assertTrue(all(x.source_record == "good" for x in got))
        self.assertTrue(all(x.query_frame >= 3 for x in got))
        self.assertEqual(retrieve_step_references(q, past, step=0), [])
        with self.assertRaises(ValueError):
            retrieve_step_references(q, past, step=0, min_temporal_gap=0)

    def test_reference_dimensional_guard(self):
        q = torch.zeros((3, 108))
        past = [StoredObservation(0, 0, 0.9, torch.zeros(5), "bad")]
        with self.assertRaises(ValueError):
            retrieve_step_references(q, past, step=1)


if __name__ == "__main__":
    unittest.main()

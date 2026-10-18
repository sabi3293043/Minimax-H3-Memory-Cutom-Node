import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mosaichunk_h3.memory_feedback import retrieve_step_references
from mosaichunk_h3.visual_memory_index import StoredObservation


class FeedbackTests(unittest.TestCase):
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

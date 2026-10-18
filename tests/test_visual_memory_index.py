"""Adaptive visual-memory index unit tests; no H3 model required."""
import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mosaichunk_h3.visual_memory_index import AdaptiveVisualIndex


class VisualIndexTests(unittest.TestCase):
    def test_content_ranking_and_step_stability(self):
        idx = AdaptiveVisualIndex(capacity=3, captures_per_step=3)
        frames = torch.zeros((6, 3, 12, 12), dtype=torch.uint8)
        frames[2, :, :6, :6] = 180
        frames[3] = frames[2]
        frames[5, :, 6:, :] = 250
        first = idx.candidates(frames, step=0)
        self.assertTrue(first)
        self.assertLessEqual(len(first), 3)
        self.assertTrue(all(f in range(6) for f, *_ in first))
        second = idx.candidates(frames, step=1)
        self.assertTrue(all(conf > 0.9 for _, _, conf in second))

    def test_bounded_replacement_and_retrieval(self):
        idx = AdaptiveVisualIndex(capacity=2, captures_per_step=2)
        deleted = []
        a, b, c = torch.zeros(108), torch.ones(108), torch.full((108,), 0.5)
        self.assertTrue(idx.put(0, 2, a, 0.5, 'a', on_evict=deleted.append))
        self.assertTrue(idx.put(1, 4, b, 0.4, 'b', on_evict=deleted.append))
        self.assertFalse(idx.put(1, 6, c, 0.1, 'c', on_evict=deleted.append))
        self.assertTrue(idx.put(2, 2, c, 0.9, 'a2', on_evict=deleted.append))
        self.assertEqual(deleted, ['a'])
        self.assertEqual(len(idx.entries), 2)
        self.assertEqual(idx.retrieve(torch.full((108,), 0.45))[0].record, 'a2')
        self.assertEqual([e.record for e in idx.retrieve(b, before_frame=3)], ['a2'])

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            AdaptiveVisualIndex(capacity=0)
        with self.assertRaises(ValueError):
            AdaptiveVisualIndex().candidates(torch.zeros((2, 2, 5, 5)), step=0)


if __name__ == "__main__":
    unittest.main()

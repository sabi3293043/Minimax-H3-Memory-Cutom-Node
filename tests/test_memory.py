import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mosaichunk_h3.layout import T2VALayout
from mosaichunk_h3.memory import MemorySettings, MosaiChunkKVCache
from mosaichunk_h3.routing.rope_reanchor import apply_rope, rope_freqs
from mosaichunk_h3.routing.selector import Selector


class MemoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.layout = T2VALayout.from_request(text_len=3, frames=90, width=64, height=64)
        self.router = Selector(d_in=8, d=4)
        self.settings = MemorySettings(sections=4, descriptor_layers=(0, 1, 2))
        self.inv = torch.tensor([0.01])
        self.cache = MosaiChunkKVCache(3, layout=self.layout, selector=self.router,
                                      inv_freq=self.inv, settings=self.settings)

    def commit(self, video_index=None):
        pos = (self.layout.text_position_ids() if video_index is None else
               self.layout.chunk_position_ids(video_index))
        rng = torch.Generator().manual_seed(123 if video_index is None else 200 + video_index)
        for layer in range(3):
            # One head, eight dimensions: six RoPE coordinates plus two unchanged.
            key = torch.randn((len(pos), 1, 8), generator=rng)
            val = torch.full_like(key, float(10 + (video_index or 0)))
            self.cache.stage(layer, apply_rope(key, rope_freqs(pos, self.inv)), val)
        return self.cache.commit(role="text" if video_index is None else "clean")

    def fill(self):
        self.commit()
        for i in range(4):
            self.cache.begin_forward(i, "clean")
            self.commit(i)

    def test_far_retrieval_local_and_clean_context(self):
        with torch.inference_mode():
            self.fill()
            self.assertEqual(self.cache.retained_indices, [0, 3, 4])
            self.assertEqual(len(self.cache.history), 4)
            local_rows = sum(r.rows for r in self.cache._records)
            self.cache.begin_forward(4, "noise")
            self.assertEqual(self.cache.retrievals, 1)
            self.assertEqual(len(self.cache._far_parts), 4)
            self.assertEqual(self.cache.past_rows(0), local_rows + 20)
            k, v = self.cache.retained(0)
            self.assertEqual(len(k), local_rows + 20)
            # Text first, local window last; far selection is visual-only.
            self.assertTrue(torch.equal(k[:3], self.cache._records[0].keys[0]))
            self.assertTrue(torch.equal(k[-self.cache._records[-1].rows:], self.cache._records[-1].keys[0]))
            for rec, rows, gate, old, new in self.cache._far_parts:
                self.assertTrue((rows >= self.layout.chunk(rec - 1).audio_rows).all())
                self.assertTrue(torch.equal(old[:, 1:2], new[:, 1:2]))
            # The same cached selection is reused for all NFEs of this chunk.
            self.cache.begin_forward(4, "noise")
            self.assertEqual(self.cache.retrievals, 1)
            self.cache.begin_forward(4, "clean")
            self.assertEqual(self.cache.past_rows(0), local_rows)
            self.cache.reset()
            self.assertEqual(self.cache.history_bytes, 0)
            self.assertEqual(len(self.cache.views), 0)

    def test_global_gate_mean_and_layerwise_reanchoring(self):
        with torch.inference_mode():
            self.fill()
            self.cache.begin_forward(4, "noise")
            gates = torch.tensor([p[2] for p in self.cache._far_parts])
            self.assertAlmostEqual(float(gates.mean()), 1.0, places=5)
            k, v = self.cache.retained(0)
            off = 3
            from mosaichunk_h3.routing.rope_reanchor import reanchor
            for index, rows, gate, old, new in self.cache._far_parts:
                want = reanchor(self.cache.history[index].keys[0][rows], old, new)
                torch.testing.assert_close(k[off:off + len(rows)], want)
                torch.testing.assert_close(v[off:off + len(rows)], self.cache.history[index].values[0][rows] * gate)
                off += len(rows)

    def test_budget_and_invalid_partition_fail_before_bank_growth(self):
        with torch.inference_mode():
            self.cache._history_limit = 1
            with self.assertRaisesRegex(RuntimeError, "above history_ram_gib"):
                self.commit()
            self.assertEqual(self.cache.history_bytes, 0)
            bad = T2VALayout.from_request(text_len=3, frames=90, width=96, height=96)
            with self.assertRaisesRegex(RuntimeError, "multiple"):
                MosaiChunkKVCache(3, layout=bad, selector=self.router, inv_freq=self.inv, settings=self.settings)

    def test_history_shares_canonical_records_and_remains_on_cpu(self):
        with torch.inference_mode():
            self.fill()
            for record in self.cache._records[1:]:
                self.assertIs(record, self.cache.history[record.index])
            self.assertTrue(all(t.device.type == "cpu" for r in self.cache.history.values()
                                for table in (r.keys, r.values) for t in table.values()))


if __name__ == "__main__":
    unittest.main()

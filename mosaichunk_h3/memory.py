"""MosaiChunk retrieval over RAVEN's execution-scoped, host-backed KV records.

Selection follows the released T2V router: balanced sections, nine pooled views,
full-candidate softmax, redundancy penalty, and mean-one selected value gates.
Verbatim rotated BF16 keys are retained like the released h3_cache.SideStore;
descriptor keys are unrotated in FP32 and retrieved keys are re-anchored.
"""

from dataclasses import dataclass
import logging

import torch

from .cache import CacheError, ChunkKVCache, _nbytes, _storage_nbytes
from .layout import video_position_start
from .routing.rope_reanchor import reanchor, rope_freqs, unapply_rope
from .routing.sections import allocate_global, kmeans_sections, maxsim
from .routing.selector import pool_views

LOG = logging.getLogger(__name__)
GIB = 1024 ** 3


@dataclass(frozen=True)
class MemorySettings:
    far_budget: int = 1
    history_ram_gib: float = 8.0
    sections: int = 40
    descriptor_layers: tuple = (12, 25, 38)
    redundancy: float = 1.0


class MosaiChunkKVCache(ChunkKVCache):
    def __init__(self, num_layers, *, layout, selector, inv_freq, settings, storage="cpu"):
        if storage not in ("cpu", "cpu_pinned"):
            raise CacheError("MosaiChunk history must use cpu or cpu_pinned storage.")
        super().__init__(num_layers, sink=1, window=2, storage=storage)
        self.layout = layout
        self.selector = selector
        self.inv_freq = inv_freq.detach().float().cpu().clone()
        self.settings = settings
        if settings.far_budget not in (1, 2):
            raise CacheError("The released router supports a far-memory budget of 1 or 2 chunks.")
        if max(settings.descriptor_layers) >= num_layers:
            raise CacheError("Router descriptor layers do not match the loaded H3 backbone.")
        for chunk in layout.chunks[:-1]:
            if chunk.video_rows % settings.sections:
                raise CacheError(
                    f"{layout.width}x{layout.height} produces {chunk.video_rows} visual rows per chunk; "
                    f"the released router needs a multiple of {settings.sections}. "
                    "Use 512x320, 640x384, 768x448, or the research canvas 1376x768."
                )
        self.history = {}
        self.sections = {}
        self.views = {}
        self._far_parts = []
        self._use_far = False
        self._selection_chunk = None
        self._current_chunk = None
        self.retrievals = 0
        self.selected_sections = 0
        self._peak_history_bytes = 0
        self._history_limit = int(settings.history_ram_gib * GIB)

    def _check_budget(self, key):
        width = int(key.shape[1]) * int(key.shape[2])
        expected = self.num_layers * 2 * width * key.element_size() * (
            self.layout.text_len + sum(c.rows for c in self.layout.chunks[:-1])
        )
        # Pooled FP32 views and the router are additional to the verbatim bank.
        expected += len(self.layout.chunks) * self.settings.sections * 9 * width * 4
        expected += sum(p.numel() * p.element_size() for p in self.selector.parameters())
        if expected > self._history_limit:
            raise CacheError(
                f"This clip needs approximately {expected / GIB:.2f} GiB of history/router RAM, "
                f"above history_ram_gib={self.settings.history_ram_gib:g}. "
                "Lower resolution/frame count or raise the explicit RAM budget."
            )

    def stage(self, layer_idx, key, value, **kwargs):
        if self.committed_chunks == 0 and layer_idx == 0:
            self._check_budget(key)
        pending = sum(_nbytes(t) for tab in (self._pending_keys, self._pending_values) for t in tab.values())
        if self.history_bytes + pending + _nbytes(key) + _nbytes(value) > self._history_limit:
            raise CacheError("MosaiChunk history reached its RAM budget; reduce the clip or increase history_ram_gib.")
        super().stage(layer_idx, key, value, **kwargs)

    @property
    def history_bytes(self):
        # Bank and local window share the same records; count each only once.
        records = dict(self.history)
        records.update((r.index, r) for r in self._records)
        return sum(_nbytes(t) for r in records.values() for table in (r.keys, r.values) for t in table.values())

    def commit(self, *, role="clean"):
        record = super().commit(role=role)
        if role == "clean":
            self.history[record.index] = record
            self._describe(record.index - 1)
        self._peak_history_bytes = max(self._peak_history_bytes, self.history_bytes)
        return record

    def _describe(self, video_index):
        record = self.history[video_index + 1]
        chunk = self.layout.chunk(video_index)
        freqs = rope_freqs(self.layout.chunk_position_ids(video_index), self.inv_freq)
        keys = {layer: unapply_rope(record.keys[layer].float(), freqs).to(record.keys[layer].dtype)
                for layer in self.settings.descriptor_layers}
        parts = kmeans_sections(
            keys[self.settings.descriptor_layers[len(self.settings.descriptor_layers) // 2]],
            chunk.audio_rows, self.layout.latent_h // 2, self.layout.latent_w // 2,
            chunk.video_latents, self.settings.sections,
        )
        self.sections[video_index] = parts
        self.views[video_index] = torch.stack([
            pool_views({layer: keys[layer][rows] for layer in self.settings.descriptor_layers},
                       self.settings.descriptor_layers) for rows in parts
        ])

    def begin_forward(self, chunk_index, role):
        self._current_chunk = int(chunk_index)
        self._use_far = role == "noise"
        if not self._use_far or self._selection_chunk == chunk_index:
            return
        self._selection_chunk = chunk_index
        self._far_parts = []
        far = list(range(max(0, chunk_index - 2)))
        if not far:
            return
        candidates = [(c, s) for c in far for s in range(self.settings.sections)]
        k_emb = self.selector(torch.cat([self.views[c] for c in far]), "k")
        q_emb = self.selector(self.views[chunk_index - 1], "q")
        logits = maxsim(q_emb, k_emb)
        red = [c for c in range(max(0, chunk_index - 2), chunk_index - 1)]
        if red and self.settings.redundancy:
            red_emb = self.selector(torch.cat([self.views[c] for c in red]), "k")
            logits = logits - self.settings.redundancy * maxsim(red_emb, k_emb).max(0).values[None]
        picks, gates = allocate_global(
            logits, self.settings.far_budget * self.settings.sections,
            float(self.selector.log_tau.exp()),
        )
        sources = sorted({candidates[i][0] for i in picks.tolist()}, reverse=True)
        hi_chunk = max(0, chunk_index - 2)
        hi = self.layout.chunk(hi_chunk).video_start
        lo_chunk = max(0, hi_chunk - self.settings.far_budget)
        lo = self.layout.chunk(lo_chunk).video_start
        span = max(self.layout.chunk(c).video_latents for c in sources)
        top = max(lo, hi - span)
        anchors = ([top] if len(sources) == 1 else
                   [lo + round(i * (top - lo) / (len(sources) - 1)) for i in range(len(sources))])
        destinations = dict(zip(sources, anchors))
        for j, pick in enumerate(picks.tolist()):
            c, section = candidates[pick]
            rows = self.sections[c][section]
            source_pos = self.layout.chunk_position_ids(c)[rows]
            dest_pos = source_pos.clone()
            ch = self.layout.chunk(c)
            frame = (rows - ch.audio_rows) // self.layout.frame_rows
            dest_pos[:, 0] = torch.tensor([
                video_position_start(destinations[c] + int(f), float(self.layout.text_len)) for f in frame
            ], dtype=dest_pos.dtype)
            self._far_parts.append((c + 1, rows, float(gates[j]),
                                    rope_freqs(source_pos, self.inv_freq), rope_freqs(dest_pos, self.inv_freq)))
        self.retrievals += 1
        self.selected_sections += int(picks.numel())
        LOG.info("MosaiChunk chunk %d: retrieved %d sections from %d past chunks; history %.2f GiB RAM",
                 chunk_index, picks.numel(), len(sources), self.history_bytes / GIB)

    def past_rows(self, layer_idx):
        return super().past_rows(layer_idx) + (
            sum(len(rows) for _, rows, _, _, _ in self._far_parts) if self._use_far else 0
        )

    def copy_retained_into(self, layer_idx, key_dst, value_dst):
        expected = self.past_rows(layer_idx)
        if key_dst.shape != value_dst.shape or key_dst.shape[0] != expected:
            raise CacheError("MosaiChunk merged KV destinations do not match the active context.")
        self._peak_gpu_slot_bytes = max(self._peak_gpu_slot_bytes,
                                       _storage_nbytes(key_dst) + _storage_nbytes(value_dst))
        off = 0

        def copy_record(record):
            nonlocal off
            n = record.rows
            key_dst[off:off + n].copy_(record.keys[layer_idx])
            value_dst[off:off + n].copy_(record.values[layer_idx])
            off += n

        # Paper order: text | re-anchored far sections | two recent chunks.
        for record in self._records:
            if record.index == 0:
                copy_record(record)
        if self._use_far:
            groups = {}
            for record_index, rows, gate, old, new in self._far_parts:
                groups.setdefault(record_index, []).append((off, rows, gate, old, new))
                off += len(rows)
            for record_index, parts in groups.items():
                record = self.history[record_index]
                indices = torch.cat([p[1] for p in parts])
                key = record.keys[layer_idx].index_select(0, indices).to(key_dst.device)
                value = record.values[layer_idx].index_select(0, indices).to(value_dst.device)
                old = torch.cat([p[3] for p in parts]).to(key.device)
                new = torch.cat([p[4] for p in parts]).to(key.device)
                key = reanchor(key, old, new)
                gates = torch.cat([torch.full((len(p[1]),), p[2], dtype=torch.float32) for p in parts])
                value = (value * gates.to(value.device)[:, None, None]).to(value_dst.dtype)
                start = 0
                for dest, rows, _, _, _ in parts:
                    n = len(rows)
                    key_dst[dest:dest + n].copy_(key[start:start + n])
                    value_dst[dest:dest + n].copy_(value[start:start + n])
                    start += n
        for record in self._records:
            if record.index != 0:
                copy_record(record)
        if key_dst.device.type != "cpu":
            self._h2d_bytes += _nbytes(key_dst) + _nbytes(value_dst)
            self._h2d_calls += 2
        return off

    def retained(self, layer_idx):
        spec = self.retained_spec(layer_idx)
        if spec is None:
            return None
        key = torch.empty((spec.rows, spec.heads, spec.head_dim), dtype=spec.dtype, device=spec.device)
        value = torch.empty_like(key)
        self.copy_retained_into(layer_idx, key, value)
        return key, value

    def reset(self):
        super().reset()
        self.history.clear()
        self.views.clear()
        self.sections.clear()
        self._far_parts.clear()
        self._use_far = False
        self._selection_chunk = None

    def stats(self):
        return {**super().stats(), "mosaichunk": True, "active_rows": self.past_rows(0), "history_bytes": self.history_bytes,
                "peak_history_bytes": self._peak_history_bytes, "retrievals": self.retrievals,
                "selected_sections": self.selected_sections, "far_budget": self.settings.far_budget}

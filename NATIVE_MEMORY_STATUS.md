# Native H3 visual memory — staged implementation

The legacy RAVEN/MosaiChunk chunk-causal path remains unchanged. This work
introduces a **separate native H3 path**, retaining native sampler scheduling
and existing third-party model patches.

## First milestone: bounded temporary image storage

- \`mosaichunk_h3/visual_memory_disk.py\` provides per-run UUID directories,
  OS-level cross-process ownership locks, atomic encoded-image writes, a disk
  budget, reads, and cleanup on successful exit, exception, or interruption.
- Orphan cleanup at the next session startup removes abandoned cache directories
  but skips sessions held by another live ComfyUI process.
- Caller supplies *already encoded* PNG, WebP, or JPEG; no VAE decoding,
  H3 sampling, visual feature extraction, retrieval, or guidance is done by
  this storage module.
- Run \`python -m unittest discover -s tests -p test_visual_memory_disk.py -v\`.

## Remaining milestones (not implemented here)

1. Native H3 sample observer and TAEH3 decoder integration, with no duplicated
   decode if the KJ preview node is present. Avoid retaining GPU tensors.
2. Adaptive acquisition and retrieval across space, video frames and denoising
   steps. MosaiChunk's existing router was trained for RAVEN chunk KV and cannot
   be used as an RGB or full-video-step selector without additional validation.
3. Trained or otherwise experimentally validated memory *injection* into native
   H3's next prediction. A preview callback alone is observation, not guidance.
4. Optional disk-backed historical KV store for the legacy RAVEN path, distinct
   from visual image files.
5. Spectrum Apply MiniMax H3 and Patch Sage Attention KJ compatibility tests,
   including solver ordering, attention mask semantics, and numerical drift.
6. Real H3 test matrix: no-memory parity, CFG, reference/video conditioning,
   interruption and OOM recovery, throughput, VRAM/RAM/disk limits, and
   objective identity/temporal quality comparisons. Without real tests, quality
   gains must not be claimed.

Do not enable an unvalidated correction algorithm by default.

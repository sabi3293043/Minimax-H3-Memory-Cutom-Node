# Native H3 visual memory — staged implementation

The legacy RAVEN/MosaiChunk chunk-causal path remains unchanged. This work
introduces a **separate native H3 path**, retaining native sampler scheduling
and existing third-party model patches.

## First milestone: bounded temporary image storage

- `mosaichunk_h3/visual_memory_disk.py` provides per-run UUID directories,
  OS-level cross-process ownership locks, atomic encoded-image writes, a disk
  budget, reads, and cleanup on successful exit, exception, or interruption.
- Orphan cleanup at the next session startup removes abandoned cache directories
  but skips sessions held by another live ComfyUI process.
- Caller supplies *already encoded* PNG, WebP, or JPEG; no VAE decoding,
  H3 sampling, visual feature extraction, retrieval, or guidance is done by
  this storage module.
- Run `python -m unittest discover -s tests -p test_visual_memory_disk.py -v`.

## Second milestone: native sampler observation (experimental)

- `MosaiChunk H3 Native Visual Observer (experimental)` is a MODEL-to-MODEL
  node registered separately from RAVEN's chunk sampler. It forwards the
  unchanged native sampler execution and callback, so native solver behavior is
  not replaced.
- Uses KJNodes' installed 24-channel TAEH3 decoder for image capture.
  Current implementation independently decodes the predicted full video; this
  can be expensive and is not yet shared with KJ Model Preview Override.
- A simple feature-diversity heuristic chooses up to `max_frames` candidate
  observations per eligible denoising step. It is **not** MosaiChunk's trained
  router, identity tracking, temporal repair, or memory injection.
- Captured WebP frames are execution-local, disk-budgeted and erased after
  the native sampler returns or raises. Storage does not persist across runs.
- Requires KJNodes at `custom_nodes/ComfyUI-KJNodes` and a compatible TAEH3
  checkpoint in `models/vae_approx`. Startup checks should be done with
  the user's actual ComfyUI and checkpoint installation.
- CPU fake-executor tests can verify routing and cleanup, not real H3 quality.
  Run `python -m unittest discover -s tests -p test_native_visual_observer.py -v`.


## Third milestone: Tiny VAE selector and adaptive observation index

- The Native Visual Observer's `tiny_vae_name` dropdown lists all files in
  `models/vae_approx`, plus `auto`. Auto prefers `taeh3.safetensors` and
  then `taeh3_decoder.safetensors`. All explicit selections are validated
  to have 24 H3 latent channels when loaded.
- Recommended model: https://github.com/madebyollin/taehv/blob/main/safetensors/taeh3.safetensors
  (alternative decoder: https://huggingface.co/Kijai/MiniMax-H3-TAE).
- `visual_memory_index.py` ranks full-video RGB frames by coarse visual
  novelty, within-video transitions and cross-denoising-step stability. This is
  a heuristic, NOT trained object tracking, learned MosaiChunk retrieval, or
  an identity-consistency measurement.
- A user-configurable `max_memory_frames` limits the number of stored images;
  superseded records are actually deleted from disk and budget accounting
  updated. Observation occurs only after `min_progress`. A decoder exception
  never modifies sampler latents or suppresses the original callback.
- Tests cover dropdown selection, retrieval, bounded eviction, file cleanup,
  and native-executor continuity. Tests are synthetic; no GPU H3 generation
  has been tested.

## Remaining milestones (not implemented here)

1. Share TAEH3 decoder outputs with KJ preview without double decode; add safe full-video observation resource budgets.
2. Replace coarse RGB thumbnail heuristics with validated identity/object
   tracking, occlusion-aware selection and trained feature retrieval. The
   released MosaiChunk KV router is not a drop-in RGB selector.
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

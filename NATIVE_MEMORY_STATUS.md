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

## Fourth milestone: cross-step retrieval interface (read-only)

- `memory_feedback.py` ranks bounded candidate references from *earlier*
  denoising steps for different frames in the same predicted video, using the
  retained RGB-thumbnail descriptors.
- The native observer computes each frame descriptor once per step, queries
  its historical index **before** adding the current observations, and
  optionally provides proposals plus a disk-read function to a diagnostic
  `feedback_hook`. The hook is internal, not a user-facing correction control.
- The hook cannot receive or modify sampler tensors and failures are isolated
  from sampling. Temporal distance, confidence and similarity safeguards
  prevent self-matching or using future predictions as past references.
- This is **not** evidence that the correct character or object has been
  matched; coarse RGB similarity has limited semantic ability.
- Local synthetic tests cover retrieval ordering, temporal filtering,
  callback preservation and cleanup. This retrieval interface alone does not
  inject guidance; the opt-in fifth milestone below adds a separate consumer.

## Fifth milestone: opt-in native appearance correction prototype

- The same Native Visual Observer exposes `observe_only`,
  `manual_frames` and `retrieved_frames`. Native full-video sampling remains
  in place, including ordinary sampler nodes and full-video callbacks.
- Tiny VAE observations activate a bounded frame-pair plan for subsequent
  model evaluations. Manual controls specify final-video frame numbers;
  automatic retrieval uses coarse thumbnail proposals, not identity tracking.
- Selected H3 blocks can add bounded appearance attention from a target frame
  to current-evaluation K/V in a source **generated-video** frame. This does not
  retrieve historical K/V or directly inject an external reference image.
  Native reference/keyframe conditioning remains handled by native H3.
- Current value-feature similarity, uniqueness and mutual matches gate the
  residual. Source/text/conditioning/audio rows are not directly edited.
  Later joint computation can still propagate changes to audio and other rows.
- CFG 1, batch one, no denoise mask. Layer and strength defaults are exploratory.
  Zero strength bypasses injection. An aggregate decoded-pixel budget guards
  observation buffers; it is not a total memory limit. Full Tiny VAE decode
  remains separate from KJ Preview and can be expensive.
- Existing Sage overrides and block replacements are composed. Active guidance
  gives Spectrum an independent, actual-evaluation runtime with forecasting,
  replay and model-aware correction disabled; its speedup is not preserved.
  The source MODEL and original Spectrum configuration are restored/preserved.
- On 2026-10-19, 437 selected CPU regressions passed. A real native H3/guider/
  Euler probe with tiny random weights and installed Tiny VAE weights showed
  prediction changes in text, first-frame, first/last-frame and reference modes,
  zero-strength parity, real Spectrum-wrapper execution, and cancellation/
  image-cache cleanup. The installed KJ Sage calling adapter ran with a CPU
  substitute kernel; the CUDA Sage kernel has not been validated here.
- **No trained-model quality or speed improvement is established.** Manual
  pairs need to identify useful reappearances, and coarse automatic matching
  can confuse objects. Counters report actual attention corrections; zero
  corrections means the matching gates did not accept a change.

## Reliability and observability revision

- Native status events distinguish observation-only execution from enabled
  guidance and count actual corrections. Timing and encoded-image-cache peaks
  provide limited overhead diagnostics without claiming process-wide memory.
- Configurable observation cadence preserves the existing frequency by default.
  Active guidance omits late observations that cannot influence another model
  evaluation. CUDA OOM disables repeated observation attempts and stale plans.
- The decoded-pixel guard now also protects observation-only runs, which retain
  normal generation when observation exceeds the budget. Guided runs retain an
  explicit budget error. Cancellation and file cleanup remain scoped to a run.
- 443 selected CPU tests, the tiny actual-native CPU integration and the preview
  JavaScript regressions pass. A separate small CUDA test also passes through
  installed KJ Sage `auto` in FP16/BF16; full-model CUDA/offloading, frontend
  rendering and trained guidance quality remain unverified.
- See [NATIVE_GUIDANCE_TEST.md](NATIVE_GUIDANCE_TEST.md). Bypass Spectrum for
  both baseline and guided runs to avoid changing forecasting between them.

## Remaining milestones (not implemented here)

1. Share TAEH3 decoder outputs with KJ preview without double decode; measure
   real peak memory beyond the aggregate observation-pixel guard.
2. Replace coarse RGB thumbnail heuristics with validated identity/object
   tracking, occlusion-aware selection and trained feature retrieval. The
   released MosaiChunk KV router is not a drop-in RGB selector.
3. Validate or replace the experimental correction with measured identity and
   temporal-quality gains. A preview callback alone is observation, not guidance.
4. Optional disk-backed historical KV store for the legacy RAVEN path, distinct
   from visual image files.
5. Real CUDA Spectrum/Sage/preview compatibility and performance tests; preserving
   Spectrum forecasting while correcting predictions requires further design.
6. Real H3 test matrix: no-memory parity, CFG, reference/video conditioning,
   interruption and OOM recovery, throughput, VRAM/RAM/disk limits, and
   objective identity/temporal quality comparisons. Without real tests, quality
   gains must not be claimed.

## Automatic starting preset

At the user's request, new nodes now default to `retrieved_frames`, strength
0.10, starting observation at 25% of denoising. Matching thresholds and cache
budgets remain conservative. This is an experimental default for this release,
not evidence of optimal settings or validated quality gains.
Existing workflows retain their saved mode. Observation-only mode remains
available, and CFG 1 / Spectrum actual-evaluation restrictions still apply.

## Automatic pair selection correction

Guidance now requests forward-only references grouped by native latent frame.
Direction, same-latent duplicates and anchor/target conflicts are checked before
the pair budget is filled, rather than relying on later controller rejection.
The original read-only diagnostic interface remains bidirectional by default.
Status includes cumulative frame-pair proposals as well as actual corrections.
448 selected CPU regressions and synthetic observation tests for both decoder
types pass. The thresholds/cache budget are unchanged; quality still requires
trained-model comparisons.

## Optional observation speed controls

Adaptive scheduling uses cheap, bounded latent fingerprints with three initial
observations per spatial stage, change-triggered refreshes and a maximum refresh
gap. Progressive resolution uses half/medium/full spatial observation latents;
the final video and H3 attention remain full size. The RGB reference index and
plans rebuild at a size transition instead of mixing descriptor distributions.

Descriptor-only storage bypasses image encoding and all per-run image-cache
creation/writes while retaining retrieval descriptors in RAM. Toggling storage
alone is bit-exact in a tiny actual native CPU sampling test. Fixed/full mode
remains available. New nodes default to adaptive/progressive observation with
image storage disabled; existing saved settings are preserved. 462 selected CPU
tests pass; an isolated
actual Tiny VAE CPU benchmark shows lower observation time, not a validated
trained-video or end-to-end GPU speedup. See README.md for controls and limits.

# Validation status

## Shared Tiny VAE frames, standalone preview and CPU budgets (2026-10-20)

- 555 selected CPU/unit tests pass, including exact input/weight invalidation,
  partial frame requests, independent returned tensors, small cache budgets,
  cross-thread and nested-session isolation, hook restoration and CPU affinity.
- Actual native tiny H3 sampling with KJ Preview Override passes on CPU and
  CUDA, in both outer-wrapper orders. Six callbacks decode 42 frames and reuse
  35; sampler output is bit-exact against sharing disabled for this fixture.
  No shared CPU frames remain after callbacks; cancellation restores hooks and
  model options. The GPU test covers FP16 channels-last singleton weights.
- When KJ uses the same Tiny VAE, its full-resolution frames drive memory even
  if progressive observation is selected. Without KJ, native three-size
  progressive sampling still passes with retained references and guidance.
- Real ComfyUI frontend checks pass play/pause, frame scrubbing, external-preview
  hiding, default Auto CPU threads, shared decoding and unchanged serialization.
  No new browser errors occur after the isolated fixture finishes startup.
- The 5900X CPU benchmark uses six fixed generated-video frames, one source
  region, one untimed warmup and three timed refreshes per thread limit. Median
  refreshes: 2 threads 31.37 s; 8 threads 12.28 s; 12 threads 11.27 s; 24 threads
  9.53 s. All 32x32 masks, visibility flags and tracked boxes match exactly.
  No GPU was initialized for this benchmark. This establishes about 3.3x faster
  tracking for that input, not a whole-video speedup or a universal thread optimum.
- These tests use a tiny H3 fixture and a previously generated video for tracking;
  no full-checkpoint generation or end-to-end quality/performance comparison was
  run. Unsupported decoder APIs/formats intentionally bypass frame sharing.

## Progressive reference continuity (2026-10-20)

- 532 selected CPU/unit tests pass, including retained protected records and
  frozen regional references across both size changes, higher-detail reference
  learning, same-scale descriptor comparisons, shared observation capacity,
  proposal priority/anchor protection, and cancellation after full-size preview.
- Real installed 2D Tiny VAE inference on CPU exercises three input sizes,
  retains earlier anchors, learns trusted full-size references, and clears all
  regional banks. Its 2,446,179 decoder parameters occupy about 4.7 MiB in FP16;
  this excludes decode activations and framework overhead.
- A native tiny H3 CFGGuider/Euler fixture completes eight steps with three
  real Tiny VAE observation sizes and 14 attention corrections. Model/output
  tensors are verified on CPU; ComfyUI dependency capability probing can still
  initialize a CUDA context. This is pipeline evidence, not trained-video
  appearance or performance evidence.
- Progressive resolution remains off by default, with a tooltip, console
  warning and status notice describing the speed/quality trade-off. Querying
  retained scales adds decode work; net speedup and visual consistency require
  same-workflow video comparisons. No full checkpoint generation was run.
- Fresh installed-node registration/default checks and 111 focused tests pass;
  the installed real Tiny VAE passes the same three-scale retention probe.

## Whole-scene tracking fix (2026-10-20)

- 524 CPU/unit tests pass, including empty-region opt-in, global guidance outside
  regional boxes, anchor protection, no-detection fallback and tracker failure.
- Real SAM 2 CPU inference discovers one object from three synthetic frames
  without a manual box, tracks it, refreshes it and cleans temporary files.
- Native tiny H3 CPU sampling still passes text, first, first/last and reference
  conditioning, residual reuse and cancellation with whole-frame guidance.
- Browser interaction checks pass with the updated whole-frame/priority controls.
- Full trained-video identity/occlusion quality still requires video comparisons.

## Native region memory and optional tracking (2026-10-20)

- 517 selected CPU/unit tests cover legacy contracts plus region isolation,
  trusted-reference drift rejection, protected eviction, callback continuity,
  diagnostic snapshots and optional-installer isolation/cleanup.
- Native tiny H3 + CFGGuider/Euler runs use the real installed Tiny VAE for
  text, first-frame, first/last-frame and reference conditioning. Cancellation
  restores original model options and releases residuals.
- A native tiny H3 region test uses controlled decoded observations to exercise
  actual regional attention, yielding 34 correction calls. It is not a test of
  trained-video identity or the appearance match rate of natural images.
- Tiny CUDA/aimdo runs pass normal guidance, regional correction and RAM cache
  reuse. The tiny model does not exercise full-size H3 sparse Sol kernels.
- The official SAM 2.1 tiny checkpoint is SHA256-verified and loads on CPU.
  Real CPU tracking of three synthetic frames returns masks and cleans its
  worker process and temporary image directory. This is pipeline evidence,
  not proof of reliable reacquisition through a real clothing occlusion.
- Browser tests exercise drawing regions, pinning final-video frame indices,
  presets/Custom detection, JSON report download/comparison and close cleanup.
  A separate CPU ComfyUI server verifies real node registration, inspector
  rendering and hidden-widget serialization with no jobs queued.
- Shared descriptor extraction matches pre-change whole-frame guidance exactly
  in five seeded CPU arithmetic comparisons. This does not prove that optional
  regional guidance, confidence weighting or acceleration preserve the output.
- No full trained-checkpoint video was generated for this revision. Compare
  identical-seed outputs before claiming improved clothing/identity quality.

## Historical observation speed defaults (superseded)

Earlier versions defaulted to `observation_schedule=adaptive`,
`progressive_resolution=true` and `store_observation_images=false`. Existing
workflow values remain explicit and are not overwritten. Historical references
below to fixed/full/image-storage defaults describe earlier revisions. The
default change does not establish a full-model speedup or optimal quality.

## Adaptive/progressive observation and descriptor-only memory

- **462 selected CPU tests passed, two warnings.** New contracts cover spatial-
  only resizing, preserved input/time/dtype, bounded fingerprints, warmup,
  accumulated drift, per-frame change detection, forced refreshes, resolution
  transitions, cross-scale index eviction and unchanged fixed scheduling.
- Integration tests show three progressively increasing decoder input sizes,
  retrieval rebuilding, fewer decodes on stable predictions, all sampler
  callbacks preserved, unchanged source latents and cleanup. Disabling image
  storage calls neither the image encoder nor the disk-cache constructor and
  retains exactly the same frame-pair plans in a controlled test.
- The actual native H3/CFGGuider/Euler CPU probe with random tiny weights and
  installed Tiny VAE weights passes again. Image-storage on/off yields bit-exact
  guided outputs. Adaptive/progressive zero-strength output is bit-exact to the
  native baseline; enabled manual correction still produces finite changes.
  This does not establish trained automatic retrieval quality at reduced sizes.
- One warmed CPU observation-only timing run, with two Torch threads, 20
  callbacks, seven latent frames and 192x128 full-size observations, measured
  fixed/full at **12.82 s / 16 observations** and adaptive/progressive at
  **4.96 s / 10 observations**. Both used actual installed Tiny VAE weights and
  image storage off. This isolates observation overhead on a constant predicted
  clip, not end-to-end generation, GPU speed or quality; it is not a promised
  percentage speedup.

The new scheduling/resolution options remain off by default. Smaller decoder
inputs are approximate; cross-scale retrieval references rebuild, which can
temporarily reduce guidance. Full-model GPU performance, trained video quality
and temporal Tiny VAE weights at reduced spatial sizes remain to be tested.

## Automatic frame-pair budget revision

- **448 selected CPU tests passed, two warnings.** Regression cases show that
  reverse-direction matches, matches within one latent frame and anchor/target
  conflicts no longer crowd valid pairs out of the automatic guidance budget.
  Generic read-only retrieval remains bidirectional unless forward selection
  is requested. Targets are deduplicated by their native latent-frame positions.
- A synthetic observer integration exercises both 2D and temporal Tiny VAE
  layouts, real indexing/retrieval and GuidanceSession plans. Both retain valid
  earlier-to-later pairs, protect source anchors, publish proposal counters and
  remove their temporary images.
- Existing real native CPU integration still passes manual guidance, zero-
  strength parity, conditioning modes, Spectrum wrapper composition and cleanup.
  Its short four-step automatic run with actual Tiny VAE weights accepts zero
  proposals. That is reported as zero guidance, not a quality improvement.
  A separate controlled-observation case through the same real native guider
  produced two automatic proposals and two attention corrections affecting six
  rows. Its preview frames and relaxed feature thresholds are synthetic, so it
  establishes routing into attention rather than automatic identity recognition.
  Trained automatic-guidance quality and throughput remain unestablished.

The image cache limits, matching thresholds and user-selected defaults are
unchanged. This fixes candidate selection order; it does not relax identity
matching or add a learned tracker.

## Automatic default preset

New nodes now default to experimental `retrieved_frames` at strength 0.10 and
`min_progress=0.25`, as requested by the user. Existing explicit workflow values
are preserved. References below to observation-only defaults describe earlier
revisions. This default change does not establish that automatic retrieval is
better than manual guidance or that the settings are optimal.

## Native observation reliability revision (2026-10-19)

- **443 selected CPU tests passed, two warnings.** New lifecycle tests verify
  cadence reduces decoder calls without dropping sampler callbacks; a synthetic
  CUDA OOM disables retries and clears stale guidance; observation-only budget
  refusal preserves native generation; BaseException cancellation releases
  image files; and active correction skips the final unused observation.
- The actual native H3/CFGGuider/Euler/Tiny VAE CPU probe passed again in all
  four conditioning cases below. Zero-strength native parity and cancellation
  cleanup remain intact. Spectrum still uses actual evaluations when guided.
- The installed KJ `get_sage_func('auto')` calling adapter and **real Sage CUDA
  kernel** passed a small synthetic attention/correction test in FP16 and BF16
  on RTX 4090. Each corrected 16 target rows, preserved source/non-target rows
  and input tensors, and returned finite outputs within a rounding-tolerant
  residual bound. This did not load H3, exercise full-model offloading or
  establish video quality/performance. Only Sage `auto` was checked on CUDA.
- Preview protocol tests passed 32/32 and preview controller tests 29/29. The
  status routing checks pass for native and RAVEN nodes, separate event channels
  and subgraph instances. These use mocked frontend objects; browser rendering
  of the new status widget is not claimed.
- The user's successful trained-model video used `observe_only`. Its consistency
  is therefore a native-model result, **not evidence of guidance improvement**.
  A fixed-seed comparison with Spectrum bypassed in both runs remains required.

Observation timing measures decode, indexing and image storage. Peak cache bytes
measure encoded image records; neither metric measures total model RAM/VRAM.

## Native Tiny VAE correction prototype (2026-10-19)

The Native Visual Observer now has opt-in appearance guidance. The legacy RAVEN
backend below is separate and has not been rewritten. The default remains
observation only; this prototype is not trained MosaiChunk retrieval.

- **437 tests passed, two warnings** in the selected inherited CPU regressions
  plus all packaged tests. New tests exercise target-row-only residuals, the
  residual bound, rejection of weak matches, zero-strength passthrough, temporal
  frame mapping, anchor protection, options restoration, existing block/Sage
  composition and independent Spectrum configuration.
- A CPU integration probe used the actual installed CFGGuider, Euler sampler,
  H3 blocks, ModelPatcher and Tiny VAE checkpoint with a tiny randomly weighted
  H3. Text, first-frame, first/last-frame and reference conditioning produced
  finite predictions; enabled corrections changed predictions in each case.
  Matching thresholds were relaxed for this synthetic wiring test, so it does
  not establish that the production thresholds find useful identities.
- A zero-strength observer produced bit-exact native baseline results. The
  original MODEL options were preserved, callbacks forwarded, cancellation
  propagated and temporary image directories removed.
- Actual installed Spectrum wrappers ran with forced actual evaluations and
  disabled forecasting/replay. The installed KJ Sage adapter ran against a CPU
  SDPA substitute kernel, validating its calling/layout interface only.
- No trained GPU generation, actual CUDA Sage kernel, live preview rendering,
  peak full-model memory, throughput or video-quality result is claimed.
  Spectrum's forecasting speedup is deliberately unavailable with active
  correction. Source-frame guidance is from current generated-video features,
  not an external-reference K/V branch or persistent historical feature bank.

For an initial trained-model A/B test, follow the manual-frame instructions in
README.md and compare the same seed with zero and nonzero strength. Check the
actual correction counters and the rendered video; observation counts alone
are not evidence of effective guidance.

## KJ preview compatibility correction

The controller now runs downstream KJ Model Preview Override wrappers around the chunk loop using ComfyUI's native WrapperExecutor. Upstream preview wrappers already surround the controller and are not run twice. Only the inspected callback-only preview class is accepted; an unrelated function registered under the same key still fails validation. Unknown sampling hooks retain the explicit rejection, now identifying their keys.

The selected CPU regression suite passed **404 tests** (two warnings), including five checks against the actual installed KJ preview wrapper: both node orders, six captured preview frames and forwarded callbacks, cleanup after an exception, rejection of an unknown hook, and rejection of a preview-key impostor. These five tests skip when KJNodes is not installed.

The tiny native CFGGuider/H3/router/cache probe also passed with the actual downstream KJ wrapper in text, first/last-frame and reference modes: 24 preview frames per mode, six chunks, three historical retrievals, unchanged trained-recipe tensor parity, audio callback scaling and cancellation cleanup. Preparation is substituted with a tiny CPU model; no trained GPU generation, tiny VAE/audio decoder performance or browser rendering is established by these checks.

This is an experimental local port. The following results were collected on 2026-10-18. Full-model video generation, output quality, and peak full-model memory have not yet been established by this repository.

| Check | Observed result | Limit |
|---|---|---|
| Installed ComfyUI feature probe | All required features passed on ComfyUI 0.37.0, Python 3.12.13, Torch 2.10.0+cu130 | Checks API availability, not generation |
| Live isolated ComfyUI API | Both nodes imported; router selectable; LATENT/IMAGE/AUDIO sockets exposed; normal server still responding | Example socket/choice validation passed except the full checkpoint, whose download was still pending |
| New memory/downloader tests | 5 passed | Synthetic keys, small CPU tensors and artifact-integrity checks |
| Inherited cache/storage/rollout/schedule regression tests | 340 passed | Tested against this fork's cache factory/hook changes; two expectations updated to require complete cleanup after failure; fake model used for rollout tests |
| Preview JavaScript protocol/state/identity tests | 32/32 passed | No browser rendering claim |
| Preview controller lifecycle tests | 29/29 passed | Mock DOM and WebSocket events, not browser playback |
| Tiny actual H3 integration | Six chunks, 24 denoising forwards, five clean fills, three historical retrievals, finite AV latents | Real Comfy H3 blocks + stock ModelPatcher, but tiny random weights on CPU; no VAE decode |
| Tiny CUDA H3 integration | Six chunks and three historical retrievals passed on RTX 4090 using BF16 and CPU-backed KV | Tiny random model; exercises the GPU/CPU transfer and attention paths, not trained full-model generation |
| Released T2V router | SHA256 verified; all 13,648,896 parameters loaded strictly | Tests checkpoint compatibility |
| Router parity against pinned source | Query/key descriptors bitwise equal; 40 unique selected sections; mean gate ~1 | Deterministic synthetic descriptors, not generated-video evaluation |
| Existing trained video/audio VAEs | Incremental decode returned 22 finite 512×320 RGB frames and a finite stereo 32 kHz waveform | Zero AV latents; proves the collector/decoder path, not generated-video quality |

The inherited backend's upstream real-weight validation belongs to that upstream project. Its H200 experiments with a simulated 24 GiB VRAM envelope do not establish that this port has generated video on an RTX 4090. See [upstream validation](https://github.com/YanzuoLu/ComfyUI-MiniMax-H3-RAVEN-Streaming/blob/main/docs/validation.md).

The original loader requires full non-pruned BF16. The separate MODEL-to-MODEL conditioning path reuses pruned FL2VA checkpoints; see PRUNED.md. The original research benchmark uses different kernels, hardware, prompt schedules, and runtime versions. No pixel-exact reproduction or quality equivalence is claimed.

To run the included tests, use a Python environment with torch and numpy:

```bash
python -m unittest discover -s tests -v
node web/tests/run_checks.mjs
node web/tests/run_controller_checks.mjs
```

The conservative workflow is intended for the first real-weight smoke test. Inspect logs for `MosaiChunk chunk ... retrieved ... sections` and check the produced video/audio, rather than treating a successful import as successful generation.

## Pruned MODEL-to-MODEL adaptation

- Converted all 51 real trained RAVEN conditioning projections. Streamed 26,142,079,488 conditioning bytes plus the source header; no full checkpoint saved. Output: 174,613,456 bytes. Time-curve relative residual: 1.3156818e-05. Across 65 probes per layer, relative L2 against original BF16 RAVEN ranges from 0.0017139807 to 0.0019677349. These differences include BF16 rounding.
- Nine packaged unit tests passed (conversion algebra, both timestep adapters, strict byte-range refusal, memory routing, downloader integrity).
- Tiny actual H3 full-vs-converted streaming rollout passed: six chunks and three historical retrievals. Video/audio output relative L2: 2.66e-7 and 6.03e-7. Synthetic small weights, not trained video quality.
- CUDA INT8 MLP test passed: both hooks ran, output matched the independent PEFT reference bitwise, weights stayed quantized.
- After the final socket revision, CPU structural checks passed: MODEL input/output, independent source branch, incoming LoRA patches retained, controls retained, and quantized adapter bytes visible to Comfy's memory accounting. No generation was run for this final revision, at the user's request.
- A broader inherited regression selection returned 167 passes and one strict BF16 bit-exact failure. Reproducing the same seeds against the original committed runtime produced the same failure/pass pattern; this is a pre-existing chunked-vs-whole GEMM rounding issue. It is not reported as a passing suite.
- An earlier real pruned-loader test reached historical retrieval (40 sections at chunk 3) before the server was closed. This is partial execution of the superseded loader, not successful completed generation or validation of the final node. No quality or final peak-memory claim follows from it.

The final conditioning was rebased into the original FL2VA table coordinates, preserving the table byte for byte. Coordinate-fit relative residual: 4.08e-5; worst projection change relative to the validated conversion: 3.52e-6. This avoids rotating the coordinates used by existing pruned-trained LoRAs. No additional model download was needed.


## Native offloading revision

Ten packaged unit tests pass, including an instrumented check that adapter casts occur twice per linear call (one A and one B), independent of activation row count. CPU structural checks exercise the native ModelPatcher clone dispatch with a dynamic-reporting test subclass: inherit preserves policy, static explicitly converts it, user patches remain, and the streaming contract accepts a prepared dynamic model. Ordinary dense sampling raises a clear error before doing model work. These are CPU checks; they do not validate AIMDO GPU execution or establish a speedup. The user runs the real generation test.

The changes are confined to this custom-node package and bundled examples. ComfyUI core and launch settings are untouched.

The offloading revision additionally passed 222 selected inherited CPU regressions covering sampling contracts, sampler integration, phase/memory planning and Comfy linear hooks (one warning). Native AIMDO GPU execution and revised generation timing remain unmeasured.


## Existing sampler controller revision

The MODEL-to-MODEL node now defers RAVEN preparation until sampling and installs an OUTER_SAMPLE wrapper. The original dense guard remains internal to prepared streaming models; ordinary sampler nodes now enter through the controller instead. First/last-frame and reference conditioning delegate to the original outer_sample without preparing a RAVEN copy.

CPU integration exercised the real installed CFGGuider.sample wrapper path with a tiny real H3 model and real MosaiChunk router/cache: six chunks, three history retrievals, 24 full-shape callbacks, exact tensor equality to the standalone RAVEN recipe with the same supplied noise, and cancellation cleanup. A simulated previous-run audio carry scale of four round-trips through the callback and process_latent_out without changing denoised audio. The native model preparation function is substituted in this controller probe; separate preparation/quantized tests cover that component.

The inherited CPU regression selection plus package tests passed 389 tests with one warning. The new status extension passed syntax and mocked event/widget checks; browser rendering remains untested. Standard-sampler workflow files are provided separately so existing user workflows are preserved. No trained GPU generation or speed/fan/peak-VRAM measurement was run for this revision; the user performs that test.

The prepared dynamic causal lane also propagates the native `prefetch_dynamic_vbars` switch normally supplied by BaseModel._apply_model. An additional CPU test verifies this switch reaches the block queue options without mutating incoming model options. Actual AIMDO GPU transfers remain unmeasured.


## Subgraph status display revision

A live reference-mode job used execution ID `105:1568`; its log explicitly reported memory inactive, but the frontend searched only the root graph for that compound ID. The status extension now resolves each subgraph instance in the execution path from app.rootGraph, including when the canvas is displaying a nested graph. `node web/tests/run_memory_status_checks.mjs` passes top-level, nested, duplicate-local-ID, invalid-path and root-fallback checks; JavaScript syntax checks pass. No browser-control tool was available to verify the live canvas. Sampling behavior is unchanged, and no running generation was interrupted. Reference and first/last-frame memory remain unimplemented.


## Image-conditioned memory revision

This revision supersedes earlier reference/keyframe passthrough behavior. The same MODEL controller now consumes native first/last-frame and reference image/video/audio conditioning in a permanent text-plus-condition prefix. Native packed positions and shifted reference target clocks are preserved; historical generated sections remain separate and are retrieved normally. Only off/unsupported non-pruned models use normal passthrough.

- The combined inherited CPU regression selection and packaged tests pass **399 tests**, with two warnings. New checks cover native reference/keyframe coordinates; independent first/last frame contributions to prefix KV; reference-video/audio prefix rows; six-chunk generation with three historical retrievals; output changes when references change; cancellation after prefix fill; input tensor preservation; rejection of varying condition timesteps; and modulation transfer across affine time bases.
- A CPU integration probe through the real installed CFGGuider.sample path exercises active text-to-video, first/last-frame and reference modes, native callback shape/audio-carry handling, explicit off passthrough and cleanup. It uses tiny real H3/cache/router components, substitutes the trained preparation function and GPU loader, and performs no trained GPU generation.
- Read only the 51 modulation projections and time tables of the actual local FL2VA INT8 baseline and Singularity REF2VA pruned v1.3 INT8 checkpoint. The transfer preserved the reference time table and added exactly the adapted-FL2VA-minus-original-FL2VA update to all 51 target projections: zero basis-fit error and zero coefficient error. Running the transfer onto the original FL2VA baseline reproduced the previous converted conditioning coefficients bit for bit. No backbone was loaded and no model was downloaded or overwritten.
- All three bundled API templates pass input/socket/choice checks against fresh native local ComfyUI schemas, including the reference node's expanded autogrow inputs. Example image filenames are placeholders and need replacement. The existing status routing test still passes nested-subgraph checks.

The released T2V router/adapter have not been retrained or benchmarked for REF2VA or first/last-frame output. These checks establish input handling, mathematical transfer and memory data flow, not trained identity quality, endpoint accuracy, GPU AIMDO execution, generation completion, peak VRAM or speed. The user performs the real video tests. Changes remain inside the downloadable custom-node package.

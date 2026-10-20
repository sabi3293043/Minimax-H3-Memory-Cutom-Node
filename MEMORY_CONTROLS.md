# MiniMax H3 Memory controls

Use the same MODEL-to-MODEL node and your existing pruned H3 checkpoint,
LoRAs, native guider and sampler. No full H3 download is required. Restart
ComfyUI and refresh the browser after updating. Saved explicit settings are
preserved; newly appended controls use their defaults.

## Presets and approximation

The preset selector updates the visible controls in the browser. Changing a
preset-controlled value marks it Custom. The server always honors the values
submitted by the workflow, including API workflows; a preset name alone never
silently overrides them.

- **Consistency** (new-node default): fixed full-resolution observations,
  eight captured frames, 64 retained descriptors, 256 anchors; Sol and residual
  reuse off. Shared current-evaluation feature descriptors are reused within
  guidance without skipping any model or guidance blocks.
- **Balanced**: the same observation coverage, Sol tau 0.50, acceleration window
  0.35–0.75, residual threshold 0.02, reuse cap one, 2 GiB RAM storage and 16 MiB
  transfers. Adaptive Sol stays off.
- **Speed**: keeps the same observation coverage, but Sol tau 1.0, window
  0.25–0.85 and residual threshold 0.05. This makes more approximate work eligible.
- **Custom**: preserves the explicit controls.

Sol and residual caching approximate calculations outside protected memory
layers. They can still change appearance, motion and audio indirectly. Neither
is described as having no effect on memory. Keep Spectrum and other generation
caches off for Balanced/Speed. Consistency with active guidance forces Spectrum
to actual evaluations as before. Sage attention and KJ preview remain supported.

## Inspector and region selection

Click **Memory inspector** on the node. After observations arrive it shows
source/target preview cards, proposal confidence, feature-match rejection
reasons and a coarse overlay of latent cells actually corrected during the run.
Cards are proposals for subsequent evaluations; a proposed pair is not proof
that clothing was corrected. The overlay is accumulated coverage, not a
measurement of visual improvement or semantic body-part detection.

Whole-frame matching stays active, including characters and scenery. Drawing a
box adds regional priority; it never switches the node to outfit-only memory.
Draw an optional box over a source preview and give it a name such as `face`, `coat` or
`trousers`. Up to four regions are supported. Boxes are stored as normalized
coordinates in `memory_regions`, for example:

```json
[{"name":"trousers","box":[0.25,0.50,0.75,0.98]}]
```

Without optional tracking the boxes stay in screen coordinates. Use them where
the subject remains within the selected region. They are not automatic anatomy
detection. Regional retrieval has its own budget so easy background matches
cannot consume every region's proposal slot. Overlapping regions never apply
multiple corrections to the same target token in a model evaluation.

Images used by the inspector are bounded thumbnails held in browser/process RAM
only. The browser retains at most 32 observation messages, with at most four
cards per message. It creates no persistent image cache. Disable
`memory_inspector` to avoid thumbnail encoding; closing the window alone does
not disable diagnostic collection. The old optional
`store_observation_images` remains separate and its run-owned cache is removed.

## Protected sources and stable appearance references

Use **Protect frame N** in a source card, or enter up to eight comma-separated,
zero-based final-video indices in `protected_frames`. A 2D Tiny VAE preview
contains fewer frames than the final video; the inspector converts its indices
before pinning. Choose frames where the entire desired outfit is visible.

Protected observation descriptors are not replaced after they become trusted
and cannot be evicted by unprotected records. The region bank waits for two
stable observations before freezing a small RGB reference. If the source later
drifts substantially, it withholds propagation and reports the reason.
Progressive previews do not reset these references. Each size has its own
appearance history, and later stages learn more detailed references. Current
detailed matches are preferred; retained earlier scales are queried using
additional current-step decodes at their original sizes. All scales share the
existing observation/image-record budget. Progressive resolution remains an
optional, default-off speed/quality trade-off; smaller early views can miss
detail, later history queries add work, and a net speedup is not guaranteed.
Normal run cleanup and invalid-observation rejection still apply.

The bank preserves a retrieval reference for this run. It does **not** inject
stored RGB images or stale attention tensors into the denoiser. Attention still
uses the source frame's current-evaluation K/V, with native positions. There is
no guarantee that a changed outfit can be restored. A trained persistent
appearance/KV backend would require a separate research and validation effort;
this implementation does not claim to reproduce the MosaiChunk router.

`confidence_guidance` optionally attenuates correction strength according to
current feature-match confidence and regional confidence. It never exceeds
`guidance_strength`. It defaults off because attenuation can also weaken useful
corrections; compare it on the same seed rather than assuming it is better.

## Optional occlusion-aware matching

`occlusion_aware_matching=false` is the default. Importing/installing the node,
using presets, or running ordinary memory guidance downloads nothing for SAM 2.
Enable the checkbox with or without manual regions. With no boxes, the existing
SAM 2 model automatically proposes up to four confident objects from a source
preview. These are generic object proposals, not semantic clothing labels.
Manual boxes remain optional priorities. The first enabled execution
automatically installs the optional packages and downloads the official
**SAM 2.1 Hiera tiny checkpoint (156 MB)**. Subsequent runs reuse it.

The installer pins official SAM 2 code and checkpoint revisions and verifies
the checkpoint SHA256. Dependencies go in this node's `.optional/sam2` directory.
It uses pip `--target --no-deps`; ComfyUI's torch, torchvision and numpy are not
upgraded. It needs an existing compatible torch >=2.5.1 and torchvision. Failed
installs do not publish a ready marker. If another process is installing, the
error identifies its lock file. Remove a stale lock only after that process exits.

For explicit setup before a generation, using ComfyUI's Python:

```bash
python -m mosaichunk_h3.optional_occlusion
```

Run that command from the node directory. Normal node installation does not run it.
Downloads and package building need internet access and a writable node folder.

The tracker runs in a separate CPU process, preserving GPU headroom for H3.
`occlusion_cpu_threads=0` uses all logical threads available to the process
(including detected affinity limits); a positive value caps the worker's
thread count. It never changes the main ComfyUI process's PyTorch or environment
settings. Tracking remains default-off and needs no packages while disabled.
More threads are not universally faster; benchmark a few limits on your CPU.
Status and optional reports now separate tracking time from total observation
time, and include the effective thread count and last refresh timings.
`occlusion_frame_selection=retrieved` selects frames from the **same current
decoded RGB observation** used by appearance memory and compatible previews.
It prioritizes current source/target indices selected by memory guidance, adds
intermediate frames and reserves coverage near the clip's end. Before retrieval
has usable matches, it spreads coverage across the observation. This does not
read stored old-step images, capture a second video or request another VAE decode.
The stored memory descriptors still identify candidates; masks always come from
the current step and resolution. No VAE weights or final output settings change.

`occlusion_frame_budget=8` is the starting budget (4–16). Protected sources are
always retained and may raise it. Missing/hidden or low-confidence masks request
broader coverage, up to 16 frames, on the **next fresh observation**. There is
only one segmentation pass per observation. Uncertain targets receive no
regional correction now; whole-frame memory continues. Only a coverage flag
survives between observations: unselected frames never borrow interpolated or
old-step masks/RGB. Broad coverage continues while uncertainty persists. This
is a speed/coverage trade-off, not a guarantee of unchanged tracking quality or
whole-video speedup.

Use `uniform` with a budget of `16` for broad fixed coverage without a second
adaptive pass. At most four selected regions are tracked. CPU tracking can be
substantially slower and use additional system RAM. Sparse frames can miss a
short occlusion. Appearance-memory observation/capture budgets are unchanged.
Optional report `tracking.last_refresh` includes selected indices, step/size,
retrieved-pair coverage, expansion, processed/encoded frames and worker timings.
Untracked or hidden targets receive no regional guidance; whole-frame matching
continues. No detections, installation/startup failure or a failed refresh
falls back to whole-frame guidance with an explicit notice. Old masks are not
reused after a failed refresh. Mask area/confidence tests are heuristic, not a
guarantee of identity or reliable reacquisition after a long absence.

SAM 2's public video API uses JPEG inputs. These are written to a parent-owned
temporary directory and removed after each refresh and on normal cancellation,
timeout or worker failure. The worker is terminated on sampler exit. Abrupt OS
or power failure can leave temporary files, as with any process-owned cleanup.
Only the reusable tracker packages and checkpoint remain installed.

Upstream implementation and license:
[Meta SAM 2](https://github.com/facebookresearch/sam2).

## Shared Tiny VAE frames and standalone playback

`share_tiny_vae_frames=true` reuses compatible uint8 RGB frames from KJ's
24-channel 2D `TinyVAEDecoder.decode_video` API within the same sampler callback.
This includes Model Preview Override and other nodes calling that API while
this memory node is in the model path. Both wrapper orders are supported. A
partial preview request fills only its frames; memory decodes only missing
frames. Memory-first decoding supplies later preview requests from the cache.

When KJ Preview Override selects the same Tiny VAE file, memory follows its
full-resolution view. This avoids adding smaller progressive decodes to a
preview that already requires full resolution. The status says `using KJ
preview resolution`. Progressive observation remains available when this node
owns the preview instead; earlier scale-specific references still persist.

Reuse checks exact decoder weights, latent values, dtype/device, layout and
resolution, including cloned inputs and modified weights. It uses bounded CPU
RGB storage, cleared after every callback and on error/cancellation. Consumers
receive independent CPU tensors. Temporary class hooks are execution-owned,
context-local in effect and restored afterwards; unrelated threads keep normal
decoding. No KJ or ComfyUI source files are edited. Weight/latent fingerprints
require CPU transfers and hashing, so reuse has some overhead. Decoded/reused
frame counts and peak cache bytes are exposed in status/reports.

Temporal/prefix decoders, float-image `decode()` calls and custom implementations
that bypass KJ's compatible `decode_video` API are not automatically shared.
Different weights or resolutions still require separate decoding. Cache budget
evictions can also require a later decode; the eviction count is reported.

`memory_preview=auto` provides an in-node silent observation player when KJ
Preview Override is absent. It supports play/pause and frame scrubbing, and
holds only the latest preview. It reuses observation RGB. Steps without a memory
observation decode only the selected preview frames; set `memory_preview=off`
to avoid this preview work. CPU JPEG encoding uses a bounded background queue.
`memory_preview_frames` (16), `memory_preview_resolution` (512) and
`memory_preview_fps` (8) control playback, not final video settings. The player
is diagnostic sampling playback, not final video/audio output. When KJ preview
is present this node's player hides and does no duplicate encoding.
When the memory player owns playback, the shared callback scope suppresses the
sampler's ordinary image preview to avoid another Tiny VAE pass for the same
display. Progress/x0 callbacks still run. Unrelated threads and previews outside
the callback scope retain their original behavior; the hooks are restored.

## Optional check skipping and second-pass repair

Both features are off by default, including when an older workflow is loaded.
They use the same MODEL node, checkpoint, Tiny VAE and existing sampler nodes.
They do not download another model. Optional SAM 2 tracking keeps its existing
explicit opt-in dependency installation.

**Check skipping:** turn on `skip_consistency_checks`, leave
`auto_check_skipping=false`, and set `checks_to_skip=1` to inspect every second
eligible step. A value of 2 inspects every third step. Every normal denoiser
evaluation and sampler callback still runs. Existing guidance plans remain in
use between checks; skipping can miss changes and is not a quality guarantee.

With `auto_check_skipping=true`, the count becomes the maximum skipped checks.
An inexpensive latent probe covers every temporal latent and an 8x8 spatial
grid. Warmup observations, detected change, resolution transitions and the
maximum gap force another check. This is a change heuristic, not a detector of
occlusion or semantic mistakes. It never skips the model's denoising work.

**Second-pass repair:** turn on `second_pass`, keep active `retrieved_frames`
or `manual_frames` guidance and CFG 1, and start with `second_pass_steps=6`,
`auto_second_pass_steps=false`. The first pass runs the normal sampler without
memory/tracking observations or memory corrections. Preview can still run.
The completed video is decoded with your selected Tiny VAE and inspected once;
optional tracking uses one bounded, uniform 16-frame pass from those same RGB
frames. Check-skipping controls do not change this deferred first-pass behavior.

The planner compares appearance in strongly aligned views. It proposes changes
only where its coarse RGB descriptors drift while structural agreement stays
high. It is not a semantic identity detector: pose/camera changes, similar
objects and low-detail frames can cause misses or false proposals. Regions and
trusted tracking restrict proposals; without boxes it uses conservative
whole-view comparisons and rejects broad exposure changes. Source frames,
protected frames and native keyframe guide spans cannot become repair targets.
Persistent changes include all qualifying target groups, rather than isolated
frame samples. If combined coverage exceeds `second_pass_max_area` (default
20% of video latents), or more than 128 region/frame candidates are needed,
the entire proposal is withheld to avoid a partial repair of a persistent change.

Repair restarts a low-noise tail of your sigma schedule with the same sampler.
`second_pass_strength=.25` sets the starting sigma to 25% of the first-pass
starting sigma; it is not a universal percentage of visual change. Manual steps
specify the repair schedule length; solvers can evaluate the denoiser more than
once per step. `auto_second_pass_steps` estimates a budget from coverage and
measured drift, bounded by `second_pass_max_steps` (12). This estimate is
uncalibrated, with no convergence guarantee or automatic early stop. It cannot
know how many steps are needed to correct clothing or identity.

Native video masks restrict changes to selected latents. Unselected video
latents and all audio latents are restored bit for bit. Decoded pixel boundaries
can still vary because the final VAE has a receptive field. The model continues
to process full context, so a small mask does not imply a proportional speedup.
Final inspection and extra denoising can make this slower; any savings come
from deferred memory/tracking checks, not from skipping model work.

A candidate is accepted only if its reference appearance metric improves while
structural agreement stays within tolerance. This is not proof of visual
quality. Otherwise the original result is retained; repair errors also retain
the original, while cancellation still cancels. No confident candidates means
zero extra sampler steps. Both normal and denoised sampler outputs agree with
the retained result. The inspector/report shows the proposal count, repair
budget, heuristic scores and acceptance. Correction counts/maps describe the
attempted repair, including when rejected. The node's final player identifies
the retained result; an external preview may have shown a rejected candidate
during repair.

This requires native H3 video/audio denoise-mask support in ComfyUI and a complete
descending sigma schedule ending at zero. Caller-supplied denoise masks remain
unsupported for memory guidance; the repair pass owns its masks internally.
Test manual skipping and manual second-pass repair separately against the same
seed with both features off before trying their automatic controls. Inspect the
video and soundtrack as well as timings. No trained-checkpoint end-to-end
speedup or quality improvement has been established for these features.

## Compatibility and reports

Preflight checks reject unsupported CFG, denoise masks, model geometry,
guidance layers, conflicting generation caches, before sampling. Manual regions are not required for tracking. Missing Tiny VAE files name the required decoder and folder.
The original sampler and model options are restored after success, interruption
or failure. No ComfyUI core, launcher, diffusion checkpoint or LoRA is edited.

**Download report** saves settings and current metrics through your browser.
`export_memory_report=true` additionally saves an atomic JSON report under
`ComfyUI/output/minimax_memory_reports`, including seed, solver/sigmas, latent
geometry, implementation hash, framework versions, timings and correction/cache
counts. It saves no prompt text or preview image content. Reports are opt-in and
are retained until you delete them. They do not hash checkpoint or LoRA files.

Select two reports in the inspector to compare them, or use:

```bash
python compare_memory_reports.py baseline.json guided.json
```

For a useful comparison, keep model/LoRAs, prompt, seed, sampler, steps, dimensions
and conditioning fixed. Generate guidance off/acceleration off, then guidance
on/acceleration off, then guidance on/Balanced. Compare several seeds, including
full-body reappearance, two similar characters crossing, viewpoint changes,
camera motion and scene cuts. Inspect face/outfit, motion and audio manually;
counts and timings do not establish identity quality.

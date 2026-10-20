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

Draw a box over a source preview and give it a name such as `face`, `coat` or
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
cards per message. It creates no persistent image cache. Close the panel or
disable `memory_inspector` to avoid thumbnail encoding. The old optional
`store_observation_images` remains separate and its run-owned cache is removed.

## Protected sources and stable appearance references

Use **Protect frame N** in a source card, or enter up to eight comma-separated,
zero-based final-video indices in `protected_frames`. A 2D Tiny VAE preview
contains fewer frames than the final video; the inspector converts its indices
before pinning. Choose frames where the entire desired outfit is visible.

Protected observation descriptors are not replaced after they become trusted
and cannot be evicted by unprotected records. The region bank waits for two
stable observations before freezing a small RGB reference. If the source later
drifts substantially, it withholds propagation and reports the reason. A size
change rebuilds the bank rather than comparing uncalibrated resolutions.

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
After selecting a region, enable the checkbox. The first enabled execution
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

The tracker runs in a separate CPU process with two threads, preserving GPU
headroom for H3. It uses a maximum of 16 sampled observation frames and four
selected regions per refresh. CPU tracking can be substantially slower and use
additional system RAM. Sparse preview frames can miss a short occlusion.
Untracked or hidden targets receive no regional guidance; old masks are not
reused after a failed refresh. Mask area/confidence tests are heuristic, not a
guarantee of identity or reliable reacquisition after a long absence.

SAM 2's public video API uses JPEG inputs. These are written to a parent-owned
temporary directory and removed after each refresh and on normal cancellation,
timeout or worker failure. The worker is terminated on sampler exit. Abrupt OS
or power failure can leave temporary files, as with any process-owned cleanup.
Only the reusable tracker packages and checkpoint remain installed.

Upstream implementation and license:
[Meta SAM 2](https://github.com/facebookresearch/sam2).

## Compatibility and reports

Preflight checks reject unsupported CFG, denoise masks, model geometry,
guidance layers, conflicting generation caches, and missing region controls
before sampling. Missing Tiny VAE files name the required decoder and folder.
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

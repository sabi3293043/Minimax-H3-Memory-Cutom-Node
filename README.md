# MiniMax H3 Memory Custom Node

**Experimental local ComfyUI port of MosaiChunk's T2V memory router. Full-size video generation and quality are not yet validated by this repository.**

**Existing pruned H3 checkpoint:** use **MosaiChunk H3 Conditioning**, the MODEL-to-MODEL controller, between your UNET/LoRA chain and your normal sampler or BasicGuider. Memory now accepts **text-to-video, first/last-frame and reference conditioning**. Image-conditioned routing and the transfer onto REF2VA are experimental; trained video quality has not been validated. Use four steps, CFG 1 and `generation_mode=auto`. The host sampler supplies seed/noise/step count; its numerical solver and sigma grid are replaced by the RAVEN consistency recipe. The original local FL2VA pruned checkpoint is used only as a modulation-update baseline, with `conditioning_base_unet=auto`; it is not loaded as a second generator. See [PRUNED.md](PRUNED.md). No full checkpoint download is required.

MosaiChunk retrieves details from earlier chunks when an object or scene returns. Older attention keys/values stay in system RAM. Only selected history for one transformer layer is brought onto the GPU at a time. Memory lasts for one sampler execution; it is released afterward.

This package extends [YanzuoLu's RAVEN ComfyUI backend](https://github.com/YanzuoLu/ComfyUI-MiniMax-H3-RAVEN-Streaming), using the [released MosaiChunk T2V router](https://github.com/mosaichunk/MosaiChunk). It is an independent integration, not an official release or a reproduction of the paper's multi-GPU benchmark.

## Installation

Clone into `ComfyUI/custom_nodes`:

```bash
git clone https://github.com/sabi3293043/Minimax-H3-Memory-Cutom-Node.git
```

Use ComfyUI's Python to download the required model files:

```bash
python custom_nodes/Minimax-H3-Memory-Cutom-Node/download_models.py --comfy-root . --plan
python custom_nodes/Minimax-H3-Memory-Cutom-Node/download_models.py --comfy-root .
```

The script uses `huggingface_hub`, already present in many ComfyUI environments. If it is absent, install it explicitly with your ComfyUI Python. Downloads happen only when you run this script; the nodes never download anything automatically. Restart ComfyUI after installing.

## Models

| Artifact | Folder | Download size |
|---|---|---:|
| Full **non-pruned** `minimax_h3_fl2va_bf16.safetensors` | `models/diffusion_models/` | 66.28 GB |
| `minimax_h3_raven_streaming_lora_4nfe_preview.safetensors` | `models/loras/` | 5.06 GB |
| MosaiChunk T2V `model.safetensors` and `config.json` | `models/mosaichunk/t2v/` | 54.6 MB |

### Singularity REF2VA pruned setup (two different files)

For the **Singularity REF2VA Pruned v1.3 INT8** workflow, obtain **both** of the following. They have different roles; the conditioning file does not replace the diffusion checkpoint.

| Required file | Purpose | Install to |
|---|---|---|
| [Singularity REF2VA Pruned v1.3 INT8](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity/blob/main/Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors) (`Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors`) | Main diffusion model / pruned H3 checkpoint | `ComfyUI/models/diffusion_models/` |
| [Published MosaiChunk H3 Conditioning](https://huggingface.co/sabi123456/MosaiChunk-H3-Conditioning) (`raven-fl2va-conditioning.safetensors`) | Converted RAVEN conditioning data consumed by the MosaiChunk controller | `ComfyUI/models/mosaichunk/` |

These are **separate from** the RAVEN LoRA adapter and MosaiChunk router listed above, which are also required for active RAVEN/MosaiChunk memory. The original FL2VA pruned baseline used by `conditioning_base_unet` may also be required for the controller's modulation transfer; the Singularity REF2VA checkpoint is not that baseline. See [PRUNED.md](PRUNED.md) for details.

**Experimental compatibility:** REF2VA image/reference conditioning and the transfer of the RAVEN modulation update onto REF2VA are implemented experimentally. They have not yet been validated for trained video quality or identity preservation. No compatibility guarantee is implied by these download links.

Also required, and reusable from an existing H3 setup:

- [Qwen3-VL-32B MiniMax NVFP4 AWQ text encoder](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders), in `models/text_encoders/`.
- [H3 video and audio VAEs](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae), in `models/vae/`.

The downloader pins revisions and checks SHA256 hashes. It preserves existing files. `--components router adapter` skips the base download; `--verify` checks existing model contents as well.

The original **MosaiChunk H3 RAVEN Model Loader** requires the full model. Pruned/quantized FL2VA checkpoints use the separate loader and converted conditioning in [PRUNED.md](PRUNED.md).

## Workflow

For the controller, open `example_workflows/mosaichunk_h3_controller_test.json` and retain your existing decoder chain. The node status reports active/inactive memory.

KJ's **Model Preview Override** can remain before or after the conditioning node. Its callback and cleanup run around the memory sampler; it is not silently bypassed. Other unimplemented sampling hooks remain rejected, with their wrapper keys included in the error.

The following is the legacy full-model workflow. Open `example_workflows/mosaichunk_h3_4090_test.json` in ComfyUI. API users can submit `api_workflows/mosaichunk_h3_4090_test_api.json` to `/prompt`.

1. **MosaiChunk H3 RAVEN Model Loader:** select the full H3 checkpoint and RAVEN LoRA.
2. Use the normal H3 text encoder and **MiniMax H3 Image to Video** in T2VA mode, with no first/last image connected.
3. Feed its conditioning and empty AV latent, plus both VAEs, into **MosaiChunk H3 Streaming Sampler**.
4. Save the sampler's IMAGE and AUDIO outputs, or combine them using ComfyUI's Create Video and Save Video nodes.

Re-select your installed VAE filenames if they differ from the examples.

Conservative initial settings: **512 × 320, 90 frames, 4 steps, 1 far-memory chunk, CPU storage, 8 GiB history budget**. This tests retrieval after history leaves the two-chunk recent window. It is a compatibility test, not a validated quality preset. Frame counts follow H3's `17k + 5` grid.

The released router partitions every full chunk into 40 equal sections. The visual-token count must therefore divide by 40. Examples of compatible canvases: 512×320, 640×384, 768×448, 1376×768.

`memory_mode=mosaichunk` uses text + two recent chunks + selected historical sections. `matched_window` uses text + three/four recent chunks with the same visual-token budget, for an A/B comparison. Keep seed and all other inputs identical.

## Memory and scope

The router weights occupy about 52 MiB at FP32. At 1376×768, one full media chunk's KV across all 50 layers occupies roughly **7 GiB of system RAM**, and history grows with clip length. The 66 GB model, 5 GB adapter, text encoder, VAEs, buffers, and OS memory are **additional**. The upstream full-model backend documents 128 GB host RAM as its baseline. A 96 GB machine is a constrained experimental target, not a promised supported configuration.

At 512×320, stored KV is roughly 1.15 GiB per full media chunk. The provided 90-frame example needs about 6 GiB for five stored chunks, plus router/descriptors/text. The explicit `history_ram_gib` limit prevents unlimited bank growth; it does not cap total process RAM or total VRAM.

GPU memory contains model working weights, activations, and one layer's assembled attention memory. It does **not** hold all 50 layers' historical KV simultaneously. Moving data over PCIe and CPU routing adds latency. Other GPU workloads should be paused before testing.

Current scope: text-to-video-and-audio plus experimental first/last-frame and reference memory, batch one, one GPU, one prompt per run, CPU or CPU-pinned history. The controller supports native pruned H3 models whose time basis can represent the RAVEN update. Persistent memory across runs, interactive prompt schedules, GGUF, PDD/VSA variants and paper-level reproducibility are not implemented. Image-conditioned modes reuse the released T2V router without retraining; correct data flow does not establish identity quality.

## Native H3 visual observer (experimental)

**Optional generation acceleration:** the same node now has
`generation_acceleration=sol`, `selective_cache`, or `sol_and_cache`.
Start with **Sol**, then compare the combined mode. Memory layers remain real
and dense; caching never skips them. Existing workflows default to `off`.
No new model download or Turbo LoRA is required. These are experimental
approximations with no measured full-model speedup in this repository yet.
See [GENERATION_ACCELERATION.md](GENERATION_ACCELERATION.md) for settings,
compatibility, memory limits and the comparison procedure.
VRAM-constrained setups can now choose bounded `cache_storage=ram` (or `auto`)
with small GPU transfers. Optional `adaptive_sol` varies sparsity using prediction
drift and final-detail protection. Both preserve the previous defaults; see the
same guide for the staged test procedure and limitations.

For **normal full-video MiniMax H3 sampling**, the separately registered
**MiniMax H3 Memory — Native Visual Observer (experimental)** MODEL-to-MODEL node
allows a visual-memory capture experiment without invoking RAVEN's
chunk-by-chunk sampler. Place it in your native MODEL chain upstream of the
normal H3 guider/sampler. Unlike `MosaiChunk H3 Conditioning`, it leaves the
sampler and selected solver untouched.

Its **tiny_vae_name** dropdown enumerates the files you have installed in
`ComfyUI/models/vae_approx` (plus `auto`). Any compatible **24-channel
MiniMax H3 Tiny VAE** can be selected; `auto` prefers the `taeh3` filenames.

Recommended Tiny VAE: **[madebyollin TAEH3 (`taeh3.safetensors`)](https://github.com/madebyollin/taehv/blob/main/safetensors/taeh3.safetensors)**.
Alternative: [Kijai MiniMax-H3-TAE](https://huggingface.co/Kijai/MiniMax-H3-TAE).

The observer uses selected decoded frames and a bounded temporary disk
cache that deletes its contents on completion/interruption/error. Selection
currently uses simple RGB thumbnail novelty and successive denoising-step
stability heuristics, *not* the trained MosaiChunk RAVEN router. New nodes default
to experimental `guidance_mode=retrieved_frames` at strength `0.10` and begin
observing at `min_progress=0.25`. This requires CFG 1 and disables Spectrum
forecasting during guided execution. Select `observe_only` (or zero strength)
to record observations without modifying predictions. Existing saved workflows
retain their explicit settings.
It decodes separately from KJ Model Preview Override and can slow sampling,
particularly because temporal TAEH3 must process the full sequence to examine
late video frames. This work remains experimental.

A new read-only cross-step retrieval interface can inspect references from
earlier denoising predictions and read their temporarily cached images. It
does **not** inject those images into H3 or alter subsequent predictions.
An opt-in experimental correction path is now available; visual consistency
improvements have not been established on a trained H3 generation.

### Experimental Tiny VAE appearance correction

The consistency-first starting preset uses `max_frames=8`, `max_memory_frames=64`,
`guidance_layers=8,16,24`, `guidance_similarity=0.85`, `max_anchor_tokens=256`,
`observation_interval=1`, `max_observation_megapixels=64` and `disk_gib=2`.
Observations use `observation_schedule=fixed` and `progressive_resolution=false`,
with `generation_acceleration=off`. These exploratory defaults favor observation
coverage over speed; they are not a measured optimal preset or an outfit guarantee.
Earlier observation gives cross-step retrieval more opportunities to find stable
references; it also performs more decoding than the old 50% start. Manual source
and target controls are ignored in retrieved mode. Short runs and large visual
changes may yield no confident matches and therefore zero corrections.

Use this **same Native Visual Observer node**, with the existing native H3
sampler and your own pruned INT8 generator/LoRAs. The RAVEN conditioning file,
RAVEN adapter and MosaiChunk router belong to the separate chunk backend;
they are not required by this native correction path. Do not chain the two
controllers together.

For an initial controlled A/B test, use `guidance_mode=manual_frames`,
`guidance_source_frame=0`, and a comma-separated list of reappearance frames
in `guidance_target_frames` (for example `68,85` in a 90-frame video).
Set `guidance_strength=0.10`, `guidance_layers=8,16,24`, CFG 1,
and `min_progress=0.25`. These layer/strength choices are exploratory, not a
validated quality preset. Set strength to zero for the baseline.
Bypass Spectrum for **both** comparison runs: active correction disables its
forecasting, whereas zero strength otherwise preserves it. Keep the seed,
sampler, schedule, conditioning, LoRAs and all other settings fixed. See
[the controlled test guide](NATIVE_GUIDANCE_TEST.md) for a prompt and settings.

Manual frame indices always refer to the **final video**, starting at zero.
The controller converts them to H3 latent positions. Internally it also
distinguishes temporal Tiny VAEs from 2D tiny decoders when mapping the
automatically retrieved observations; their frame indices are not equivalent.

After observing a prediction, the next native model evaluation can reinforce
selected target-video attention rows using current-evaluation K/V from the
selected source frame. Content matching uses unrotated value features,
similarity/uniqueness and mutual-match checks. The added residual is bounded;
source, text, conditioning and audio rows are not directly edited. Audio can
still change later through H3's joint video/audio computation. No previous
step's K/V is reused, and no all-layer history bank is retained.

`retrieved_frames` instead uses earlier Tiny VAE observations to propose
frame pairs. This is especially experimental: thumbnail matching is not
identity recognition, and a short four-step run may provide too little
stable observation history to produce any corrections. A log reports actual
attention corrections, not just recorded images. Neither mode guarantees
correct identity or occlusion handling; blending can bias poses or motion.

Automatic guidance filters proposals **before** spending its frame-pair budget:
the source must precede the target, both must map to different H3 latent frames,
and selected source anchors cannot also be targets. Repeated decoded frames
within one latent position count as one target. This avoids wasting the budget
on pairs the attention controller would discard later; it does not establish
semantic identity. Status reports frame-pair proposals separately from actual
attention corrections, so a zero-correction run can be diagnosed more clearly.

The existing Sage attention override is composed with the additional attention
branch. Spectrum, when present, is explicitly configured for **actual model
evaluations** for this execution; forecasting, offline replay and model-aware
correction are disabled because they can bypass or invalidate the new guidance.
Its speed benefit is therefore not retained during this prototype. Source
MODEL settings and subsequent unguided runs are preserved.

Full Tiny VAE decoding remains separate from KJ Preview and can be expensive.
The node checks `max_observation_megapixels` before observation (default
64 MP across the decoded clip); this is an observation-buffer guard, not a
total process RAM/VRAM limit. If over budget, active guidance stops with an
explicit error; observation-only generation continues with capture disabled.
Start with 512x320 / 90 frames. Temporary image
files are removed after success, failure or cancellation. Real trained-model
quality, offloading performance and GPU interoperability still require testing.

The node's live `memory_status` distinguishes **observation only / guidance
inactive** from enabled guidance and reports actual correction calls. It also
reports cumulative observation time and peak encoded-image cache bytes, which
are not total generation time or process RAM/VRAM. The console has the same
start/end notices when running without a browser. A stopped run reports its
error; normal completion reports whether cache removal succeeded or was deferred.

`observation_interval=1` preserves the original observation frequency. A value
of `2` or `3` reduces Tiny VAE decoding and indexing work, but provides fewer
opportunities to select/update references. It does not change sampler callbacks
or denoising steps. Active guidance skips observations too late to influence
another evaluation. An observation CUDA OOM or disk-memory limit clears the
guidance plan and disables further capture for that run; sampler errors and
cancellation still propagate. Full Tiny VAE decoding is still separate from KJ
Preview; duplicate decode has not yet been eliminated.

### Experimental observation speed controls

New nodes default to fixed observation at full resolution and descriptor-only
memory. Adaptive scheduling and progressive resolution remain optional speed
controls. Existing saved settings are preserved; recreate the node or edit its
widgets to adopt the new defaults. The defaults are:

| Control | Starting value |
|---|---|
| `observation_schedule` | `fixed` |
| `progressive_resolution` | `false` |
| `observation_start_scale` | `0.50` |
| `observation_full_progress` | `0.75` |
| `adaptive_change_threshold` | `0.04` |
| `adaptive_max_interval` | `3` |
| `observation_interval` | `1` |
| `min_progress` | `0.25` |
| `store_observation_images` | `false` |

Progressive mode spatially downsamples **only the Tiny VAE's observation
latents**, before decoding. When enabled, it uses half-sized observations until
50% of the schedule, three-quarter size until 75%, then full size. These
are per-dimension scales: half width/height has approximately a quarter of the
pixels. Temporal length, sampler latents, model attention and final output
resolution are unchanged. Spatial rounding and minimum sizes can merge stages
for very small videos. Smaller previews are approximate and can lose details.

Adaptive mode compares bounded thumbnails of the current latent prediction to
the last observed prediction. Changes can request an earlier refresh, while
`adaptive_max_interval` forces periodic refreshes. Each new observation size
starts with three warmup observations to seed retrieval. This is a prediction-
drift heuristic, not semantic object/occlusion detection; some changes can be
missed. The threshold is exploratory and not a trained quality calibration.

At a resolution change, the RGB retrieval index is rebuilt and stale frame-pair
plans are cleared. Cross-resolution descriptors are not assumed comparable.
Automatic guidance may pause while the new stage gathers observations. Short
four-step schedules can have too little history; begin with your usual longer
schedule and compare against fixed/full mode. The full-resolution pixel budget
is still checked before sampling, even when early previews are smaller.

Disabling image storage keeps bounded descriptors in RAM and avoids WebP
encoding, image writes and cache-directory creation. It does **not** disable
retrieval or guidance. Diagnostic hooks cannot read image files in this mode.
Synthetic native sampling tests show bit-exact output when toggling storage
alone; adaptive cadence and resolution can alter guidance and quality.

Status reports skipped observations and the latest preview dimensions. The
observation timer includes scheduling overhead. These options reduce observation
work, not the number of H3 evaluations; Spectrum forecasting remains disabled
with active guidance. End-to-end GPU speed and trained-video quality need A/B
testing. Set `observation_schedule=fixed`, `progressive_resolution=false` and
`store_observation_images=true` to return to the previous observation behavior.

See [NATIVE_MEMORY_STATUS.md](NATIVE_MEMORY_STATUS.md) for milestones and
remaining native memory-injection work.

## Validation

See [VALIDATION.md](VALIDATION.md) for the exact checks run and their limits. Synthetic tests and API compatibility checks do not establish full-model image quality or a successful generation on a 24 GB card.

```bash
python -m unittest discover -s tests -v
```

## Sources and licenses

- RAVEN ComfyUI backend: MIT, retained in `LICENSES/RAVEN-MIT.txt`, with original notices in `NOTICE`.
- MosaiChunk routing modules: copied from the pinned upstream source listed in `PROVENANCE.json`; they retain upstream's H3/RAVEN-related terms. See `LICENSES/MosaiChunk-NOTICE.md` and the [upstream license section](https://github.com/mosaichunk/MosaiChunk#license).
- New integration code: CC BY-NC 4.0. Model weights retain their own licenses and are not distributed here.

This package is not affiliated with or endorsed by MiniMax, ComfyUI, RAVEN, or MosaiChunk's authors.

# Pruned and quantized FL2VA models

The **MosaiChunk H3 Conditioning** node (MODEL input and MODEL output) adds an experimental path for existing curve-form FL2VA H3 checkpoints. The separate original loader still accepts full BF16 models. GGUF is not implemented by this loader.

The conditioning node keeps the checkpoint's quantized attention and MLP weights, attaches all 213 non-conditioning RAVEN targets, and replaces the timestep table and 51 modulation projections with converted RAVEN conditioning. Both timestep-encoder adapters and all 51 modulation adapters are included in that conversion. No adapter families are silently discarded. The MLP calls its down-projection directly to ensure the activation-side RAVEN hook runs with quantized weights.

Use the stock `minimax_h3_fl2va_pruned_int8_convrot.safetensors` first. W4A8 shares its time table but needs a separate runtime check. Ref2VA and FastVideo/VSA checkpoints are different trained generators and are not substitutes for the RAVEN FL2VA base. A matching time table verifies coordinates, not that a third-party finetune is the original base. Applying the converted conditioning replaces that finetune's modulation layers.

## One-time conditioning conversion

Use the ComfyUI Python environment, with `torch`, `safetensors`, and `requests` installed. The source adapter must be the published pinned RAVEN preview file. The command below reads only conditioning tensors from a pinned full BF16 source, through exact HTTP byte ranges. It transfers **26.142 GB**, holds one full-width projection in RAM at a time, and never writes the full checkpoint or full-width projections to disk. It refuses servers that return a full-file response instead of the requested range.

```bash
python convert_pruned_raven.py \
  --base /path/to/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --adapter /path/to/minimax_h3_raven_streaming_lora_4nfe_preview.safetensors \
  --output /path/to/ComfyUI/models/mosaichunk/raven-fl2va-conditioning.safetensors \
  --device cuda
```

`--inspect-only` reads just the source header and reports the transfer size. `--device cpu` avoids GPU use. Converted layer files in the adjacent `.conversion` directory allow an interrupted run to resume. They contain only small converted projections. Do not run two conversions into the same output directory simultaneously.

The conversion evaluates RAVEN's adapted timestep encoder over 1025 points and fits an affine representation in the existing base table's eight coordinates, preserving their exact convention for your LoRAs. Its mean is folded into each projection's bias. Every projection is checked at 65 probe timesteps against the original BF16 RAVEN calculation; relative L2 error above 1% refuses output. The comparison includes BF16 rounding differences, so it is not bitwise equivalence. A report is saved beside the converted file. The node checks the original base-table hash and adapter SHA256 before applying the result.

The conditioning file is **174,613,456 bytes**. It contains model-derived weights and remains subject to the MiniMax H3 model license and the RAVEN adapter's applicable terms. It is not a replacement generator checkpoint.

## ComfyUI workflow

The same **MosaiChunk H3 Conditioning** node now contains the sampler controller. Recreate this node after restarting ComfyUI if its new controls do not appear.

1. Connect **UNET Loader → your own LoRA chain → MosaiChunk H3 Conditioning**.
2. Connect its MODEL output to your existing **KSampler**, or to **BasicGuider / CFGGuider → SamplerCustomAdvanced**. CFG must be 1 for active memory.
3. Keep the normal H3 conditioning and empty AV latent connected to the existing sampler; keep your existing video/audio VAE decode and save chain.
4. Start with 512 x 320, 90 frames, four sampler steps, `generation_mode=auto`, `offloading=inherit`, CPU history and an 8 GiB history budget.

Example: `example_workflows/mosaichunk_h3_controller_test.json`, with an API equivalent in `api_workflows/`. This package update does not overwrite your saved workflows. Old dedicated sampler workflows remain available for compatibility.

## Mode routing

| Controller mode | Behavior |
|---|---|
| auto | Detect image/reference conditioning and checkpoint time table at sampling time |
| text_to_video | Request T2V memory; image/reference inputs still take priority and pass through |
| first_last_frame | Preserve normal generation; memory inactive |
| reference | Preserve normal generation; memory inactive |
| off | Preserve normal generation; memory inactive |

Only the supported FL2VA time table with text-only conditioning activates RAVEN/MosaiChunk. Other time tables, including REF2VA, pass through with an unsupported-checkpoint notice. A matching time table alone cannot establish compatibility with every third-party finetune. First/last-frame and reference memory are **not implemented**. The status widget and ComfyUI log state whether memory is active. Explicit modes label routing; your existing conditioning nodes still provide frames/references.

## Sampling controls

Active memory runs the trained **RAVEN consistency update**, chunk by chunk, inside the controller. The existing sampler's seed, supplied initial noise and **number of schedule intervals** are used. Its selected numerical solver, solver-specific settings, and individual sigma values are replaced by shifted RAVEN grids controlled by `video_shift` and `audio_shift`. Four steps is the released preview recipe. This is not a claim that ordinary Euler/res_multistep integration reproduces RAVEN. Partial denoise, nonempty input latents, noise masks, custom guiders, CFG other than 1, conditioning schedules and incompatible model wrappers are rejected before model preparation.

In inactive modes the original sampler, schedule, guider, LoRAs, conditioning and model run normally, with no RAVEN adapter or converted conditioning applied. Artifact files are not opened for those modes.

The node also exposes router selection, RAVEN adapter/converted conditioning selection, CPU or page-locked CPU KV storage, one/two retrieved chunks, history RAM limit, MosaiChunk/matched-window retrieval and offloading policy. The RAVEN adapter strength is fixed at 1 because its timestep/modulation updates are baked into the converted file. Style/character LoRAs remain on the incoming MODEL chain; quality when stacking them with RAVEN is experimental.

## Offloading and scope

`offloading=inherit` retains native ComfyUI dynamic/static policy for the prepared memory model. `static` is an explicit fallback for active memory only. The controller reserves chunk working memory through native `load_models_gpu` and releases its history on completion, failure or cancellation. It does not decode between chunks; your existing decoder runs after sampling. An independent local model copy isolates the incoming branch and can increase host RAM while both copies remain cached.

All changes are in this custom-node package. ComfyUI core, launcher, OS shared-memory settings, global VRAM policy and fan controls are not changed. CPU history, adapter computation, routing and clean-context fills still add overhead. Speed, GPU fan behavior and trained output quality require real generation testing; no improvement is promised from CPU checks.

The history limit covers stored KV, not all process RAM or VRAM. Default downloads fetch only router and adapter; full BF16 requires explicit `--components base` and is unnecessary here. No new download is required if you already have the adapter, router and converted conditioning.

References inspected during development: [H3 pruning and affine LoRA projection](https://huggingface.co/multimodalart/MiniMax-H3-Pruned), [conditioning rebase measurements](https://github.com/matsuo-koya/minimax-h3-notes), and [H3 pruned AdaLN builder](https://github.com/xiaolibai-sys/MiniMax-H3-Pruned-Lora-Adapter). This implementation folds the RAVEN-adapted time curve as well as the modulation adapters.

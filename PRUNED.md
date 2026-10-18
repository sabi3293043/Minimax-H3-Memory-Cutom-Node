# Pruned and quantized H3 models

The **MosaiChunk H3 Conditioning** node (MODEL input and MODEL output) adds experimental memory support for existing curve-form FL2VA and REF2VA H3 checkpoints. The separate original loader still accepts full BF16 models. GGUF is not implemented by this loader.

The conditioning node keeps the checkpoint's quantized attention and MLP weights, attaches all 213 non-conditioning RAVEN targets, and transfers the converted time/modulation update onto its 51 modulation projections while retaining the incoming time-table coordinates. Both timestep-encoder adapters and all 51 modulation adapters are included in that conversion. No adapter families are silently discarded. The MLP calls its down-projection directly to ensure the activation-side RAVEN hook runs with quantized weights.

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

The same **MosaiChunk H3 Conditioning** node contains the sampler controller for text-to-video, first/last-frame and reference generation. Restart ComfyUI and refresh/recreate this node to show the new controls.

1. Connect **UNET Loader → your own LoRA chain → MosaiChunk H3 Conditioning → your existing sampler / BasicGuider**. Keep your existing decoder/save chain.
2. Connect normal **MiniMax H3 Image to Video** conditioning for text-to-video or first/last-frame generation. Connect its first and/or last images as usual. For references, use **MiniMax H3 Reference to Video** with its VAE and reference inputs connected so it supplies encoded reference latents.
3. Leave `generation_mode=auto`, `offloading=inherit`, and `conditioning_base_unet=auto`. Auto finds the original local `minimax_h3_fl2va_pruned_int8_convrot.safetensors`. If it has a different name, explicitly select the original pruned FL2VA baseline used by your conditioning conversion. The selected baseline's small time table and 51 modulation projections are read; its backbone is never loaded as another generator.
4. Start at 512 x 320, 90 frames, **four sampler steps and CFG 1**, CPU history and an 8 GiB history budget. Bypass Spectrum and attention replacement patches for active memory. Keep the converted conditioning file and mandatory RAVEN adapter selected; do not apply RAVEN again through an ordinary LoRA loader.

The status should say **Memory active: text_to_video / first_last_frame / reference** once the prepared model loads. Later retrieval messages in the log report selected historical sections. `generation_mode=off` restores normal generation for comparison. Old reference/keyframe modes no longer silently bypass memory. Missing or incompatible image conditioning raises an error.

`example_workflows/mosaichunk_h3_controller_test.json` uses a native sampler with normal video/audio decoders. The matching API example is in `api_workflows/`. Image-mode API templates are also provided; replace their example image filenames with your uploaded images. These are schema examples, not trained quality presets. Existing saved user workflows are not overwritten.

## Mode routing

| Mode | Memory behavior |
|---|---|
| auto | Detect actual positive image/reference inputs; otherwise use text-to-video |
| text_to_video | Use text memory; supplied image/reference inputs still take priority |
| first_last_frame | Require encoded keyframes and retain them in the permanent prefix |
| reference | Require encoded references and retain them in the permanent prefix |
| off | Delegate unchanged to the original sampler and model; memory inactive |

Native keyframe/guide positions, mixed text modality tags, reference-image grids, reference-video clocks and stereo reference audio are retained. Image/reference rows join text in permanent cache record zero; they are never evicted or treated as generated-history candidates. Each completed media chunk enters the normal history bank. The router retrieves generated visual sections beyond the two recent chunks. Reference target clocks and retrieved-key re-anchoring use H3's native packed layout.

## Preserve the input model

The converted file describes RAVEN-adapted FL2VA modulation. Applying those absolute projections to a REF2VA finetune would erase its own modulation. The controller instead subtracts the original FL2VA baseline and transfers **only the RAVEN update** onto the incoming model's projections. The incoming time table and LoRA coordinates remain unchanged. A least-squares mapping handles compatible alternative pruned bases; a projection with more than 1% relative update-fit error is rejected. On the original FL2VA baseline this reproduces the converted file itself.

The baseline must be the original FL2VA checkpoint, not an arbitrary finetune with a similar name. REF2VA and image-conditioned use of the released T2V router/adapter is **experimental**, without new training or quality validation. Quantization, model finetunes and stacked style/character LoRAs can change output quality. CPU execution tests show conditioning is consumed and history retrieval runs; they do not prove identity preservation or endpoint accuracy in trained videos.

## Sampling and memory controls

Active memory uses RAVEN's chunk-by-chunk consistency update. The existing sampler's seed, supplied initial noise and number of schedule intervals are used. The selected solver, solver-specific settings and individual sigma values are replaced by node-controlled video/audio shifted RAVEN grids. Four steps is the released preview recipe. Partial denoise, nonempty input latents, noise masks, custom guiders, CFG other than 1, conditioning schedules and incompatible wrappers are rejected. Image modes also reject augmentation/schedule combinations that would change a reference row's timestep while its KV is cached.

Controls include router and adapter/conditioning selection, local conditioning baseline selection, CPU or CPU-pinned history, one/two retrieved chunk-equivalents, history RAM limit, MosaiChunk/matched-window comparison and native offloading policy. The RAVEN strength stays fixed at 1 because the converted time/modulation updates are tied to that strength. Extra references increase permanent-context RAM and attention workspace; the history limit includes their KV records and is not a total process RAM/VRAM limit.

`offloading=inherit` retains native ComfyUI dynamic/static policy. `static` is an explicit fallback. A fresh local copy isolates your original MODEL and can increase host RAM while both remain cached. History is released on success, failure or cancellation. Existing VAE decoders run after sampling. Node installation changes no ComfyUI core, model, launcher or saved workflow files. No speed, fan-noise, peak-VRAM or trained-quality claim follows from CPU checks.

No new model download is needed when you already have the original FL2VA pruned baseline, your target generator, converted conditioning, RAVEN adapter and T2V router. Default downloads fetch router/adapter only; the full BF16 checkpoint is unnecessary.

References inspected during development: [H3 pruning and affine LoRA projection](https://huggingface.co/multimodalart/MiniMax-H3-Pruned), [conditioning rebase measurements](https://github.com/matsuo-koya/minimax-h3-notes), and [H3 pruned AdaLN builder](https://github.com/xiaolibai-sys/MiniMax-H3-Pruned-Lora-Adapter). This implementation folds the RAVEN-adapted time curve as well as the modulation adapters.

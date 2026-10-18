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

Load `example_workflows/mosaichunk_h3_pruned_test.json`, then reselect your local filenames:

1. **Load Diffusion Model / UNET Loader**: select your existing FL2VA INT8 checkpoint. Add your own LoRA loaders on its MODEL chain if desired.
2. **MosaiChunk H3 Conditioning**: connect that MODEL, select the original RAVEN adapter and `raven-fl2va-conditioning.safetensors`, then connect its MODEL output to the streaming sampler. Do not apply the RAVEN adapter again in an ordinary LoRA loader.
3. Existing H3 text encoder and both VAEs.
4. **MosaiChunk H3 Streaming Sampler**: start at 512 x 320, 90 frames, four steps, one far-memory chunk, CPU KV storage, and 8 GiB history budget.

The history budget limits stored KV only; model weights, text encoder, VAEs, and activation buffers need additional RAM/VRAM. The default model downloader fetches only router and adapter. A full BF16 download requires explicit `--components base` and is unnecessary for this path.

See `VALIDATION.md` for the observed tests and remaining limits. Numerical conversion and successful execution alone do not establish long-video memory quality or equivalence to the research benchmark.

References inspected during development: [H3 pruning and affine LoRA projection](https://huggingface.co/multimodalart/MiniMax-H3-Pruned), [conditioning rebase measurements](https://github.com/matsuo-koya/minimax-h3-notes), and [H3 pruned AdaLN builder](https://github.com/xiaolibai-sys/MiniMax-H3-Pruned-Lora-Adapter). This implementation folds the RAVEN-adapted time curve as well as the modulation adapters.

## Controls and existing LoRAs

The conditioning node exposes RAVEN adapter selection, video/audio sigma shifts, CPU/page-locked CPU KV storage, one/two retrieved chunks, history RAM limit, and MosaiChunk/matched-window mode. These settings take precedence over the sampler's matching widgets. Seed and four-step sampling remain on the streaming sampler. The converted adapter strength is fixed at 1.0; changing only its remaining linear adapters would disagree with the baked timestep/modulation branch. Non-default shifts are experimental.

The node uses ComfyUI's native reload/clone mechanism to prepare an independent static model from the same local checkpoint and retains the incoming LoRA patches and model options. It does not alter the original MODEL branch or merge/overwrite the generator on disk. The independent copy can increase host RAM usage while the input MODEL remains cached. Image/reference conditioning and arbitrary sampler wrappers are still outside the streaming sampler's supported contract. Adding style/character LoRAs on top of RAVEN is experimental; retained patches do not guarantee quality.

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

Also required, and reusable from an existing H3 setup:

- [Qwen3-VL-32B MiniMax NVFP4 AWQ text encoder](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders), in `models/text_encoders/`.
- [H3 video and audio VAEs](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae), in `models/vae/`.

The downloader pins revisions and checks SHA256 hashes. It preserves existing files. `--components router adapter` skips the base download; `--verify` checks existing model contents as well.

The original **MosaiChunk H3 RAVEN Model Loader** requires the full model. Pruned/quantized FL2VA checkpoints use the separate loader and converted conditioning in [PRUNED.md](PRUNED.md).

## Workflow

For the controller, open `example_workflows/mosaichunk_h3_controller_test.json` and retain your existing decoder chain. The node status reports active/inactive memory.

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

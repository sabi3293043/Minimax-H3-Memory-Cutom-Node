# Validation status

This is an experimental local port. The following results were collected on 2026-10-18. Full-model video generation, output quality, and peak full-model memory have not yet been established by this repository.

| Check | Observed result | Limit |
|---|---|---|
| Installed ComfyUI feature probe | All required features passed on ComfyUI 0.37.0, Python 3.12.13, Torch 2.10.0+cu130 | Checks API availability, not generation |
| New memory/downloader tests | 5 passed | Synthetic keys, small CPU tensors and artifact-integrity checks |
| Inherited cache/storage/rollout/schedule regression tests | 340 passed | Tested against this fork's cache factory/hook changes; two expectations updated to require complete cleanup after failure; fake model used for rollout tests |
| Preview JavaScript protocol/state/identity tests | 32/32 passed | No browser rendering claim |
| Preview controller lifecycle tests | 29/29 passed | Mock DOM and WebSocket events, not browser playback |
| Tiny actual H3 integration | Six chunks, 24 denoising forwards, five clean fills, three historical retrievals, finite AV latents | Real Comfy H3 blocks + stock ModelPatcher, but tiny random weights on CPU; no VAE decode |
| Tiny CUDA H3 integration | Six chunks and three historical retrievals passed on RTX 4090 using BF16 and CPU-backed KV | Tiny random model; exercises the GPU/CPU transfer and attention paths, not trained full-model generation |
| Released T2V router | SHA256 verified; all 13,648,896 parameters loaded strictly | Tests checkpoint compatibility |
| Router parity against pinned source | Query/key descriptors bitwise equal; 40 unique selected sections; mean gate ~1 | Deterministic synthetic descriptors, not generated-video evaluation |

The inherited backend's upstream real-weight validation belongs to that upstream project. Its H200 experiments with a simulated 24 GiB VRAM envelope do not establish that this port has generated video on an RTX 4090. See [upstream validation](https://github.com/YanzuoLu/ComfyUI-MiniMax-H3-RAVEN-Streaming/blob/main/docs/validation.md).

The full non-pruned BF16 checkpoint is required. The original research benchmark uses different kernels, hardware, prompt schedules, and runtime versions. No pixel-exact reproduction or quality equivalence is claimed.

To run the included tests, use a Python environment with torch and numpy:

```bash
python -m unittest discover -s tests -v
node web/tests/run_checks.mjs
node web/tests/run_controller_checks.mjs
```

The conservative workflow is intended for the first real-weight smoke test. Inspect logs for `MosaiChunk chunk ... retrieved ... sections` and check the produced video/audio, rather than treating a successful import as successful generation.

# Validation status

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

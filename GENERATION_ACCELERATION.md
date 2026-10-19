# Optional native H3 generation acceleration

The **MosaiChunk H3 Native Visual Observer (experimental)** keeps its MODEL
input/output and existing sampler connection. Its optional accelerator is
independent of Spectrum, with no additional model weights, Turbo LoRA or disk
cache. Observation defaults and memory-guidance settings are unchanged.

These methods approximate model computations and can change detail, motion,
reference adherence and audio. Full trained-model GPU speed and quality remain
to be validated. Off is the default, including for older saved workflows.

## Test in this order

1. Keep your checkpoint, LoRAs, prompt, seed, sampler, steps, dimensions and
   observation settings fixed. Save a baseline with `generation_acceleration=off`.
2. Bypass **Spectrum**, EasyCache, other residual caches and other block-patching
   acceleration nodes. Keep **Patch Sage Attention KJ** and **Model Preview
   Override** if desired. Our node uses the existing backend on dense paths.
3. Select `generation_acceleration=sol`. Leave its starting controls below.
4. Compare `generation_acceleration=sol_and_cache` against both the baseline
   and Sol alone. `selective_cache` is available to isolate the cache effect.
5. Compare sampling time and complete-video time separately; check occlusion,
   reappearance, fast motion, reference likeness, fine details, speech and lip
   synchronization across several seeds. One faster clip is not a quality test.

| Control | Starting value | Meaning |
| --- | --- | --- |
| `generation_acceleration` | `off` | Select Sol first, then the combined mode |
| `sol_tau` | `1.0` | Higher is sparser with greater approximation |
| `acceleration_start` | `0.25` | Fraction of the noise schedule before acceleration starts |
| `acceleration_end` | `0.85` | Return to full computation near the end |
| `cache_threshold` | `0.05` | Lower tolerates less sampled residual drift |
| `cache_max_reuse` | `1` | At most one consecutive reuse before refreshing |
| `cache_vram_gib` | `0.5` | Maximum retained GPU cache tensors, additional to model memory |

The window uses the model's percent-to-sigma mapping; it is not a fraction of
elapsed wall time. Start must be less than end. Cache/guidance currently require
CFG 1 and no denoise mask. Euler is the initial runtime-tested sampler; repeated
or reversed evaluation sigmas invalidate caches, so other solvers may reuse less.

## Sol attention

Uses the installed ComfyUI `comfy_extras.nodes_sparse_attention` producer and
Comfy Kitchen Sol kernels. It does not copy or download kernels, install Python
packages, update ComfyUI or change your checkpoint. An older installation may
need a compatible Sol implementation before this path can accelerate anything.

First/last model blocks and the selected memory-guidance layers stay on the
normal attention path. With combined caching, section probe blocks also stay
dense. Eligible remaining blocks use the chunked QKV producer and Sol attention.
Packed text/reference conditioning stays available through exact KV access;
audio query rows stay dense. This protects those calculations, not necessarily
the final audio: later joint audio/video processing can still change it.

Missing APIs, unsupported GPU/dtype/head shape, short sequences (under 12288
tokens), and steps outside the activation window use dense computation. Watch
the actual **sparse calls** count. Zero means no sparse acceleration happened.
Existing arbitrary block patches are retained on their blocks rather than
bypassed; a patch on a memory layer is rejected because it might hide guidance.

Source design and reusable implementation:
[NVIDIA Sol-Attn](https://nvlabs.github.io/Sana/Sol-Attn/),
[ComfyUI sparse attention](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_sparse_attention.py).

## Selective residual cache

This is an independent experimental implementation, not Spectrum, and not a
drop-in FirstBlockCache. It divides the model around the protected memory
layers. A section must have one real probe block and at least two reusable
blocks. On a hit it adds the previous section update to **current** features;
it does not restore old full hidden states. Guidance layers always run with
current Q/K/V. Surrounding cached updates remain approximations.

A bounded sampled probe checks each video latent frame and each packed
conditioning/audio segment separately. Accumulated change must pass the
threshold, and the reuse cap forces regular refreshes. Memory frame-pair changes,
guidance expiry, branch/geometry changes, and repeated/reversed sigmas invalidate
the cache. The start/end window leaves early and final denoising uncached.
Sampling can miss changes between selected tokens/channels; it is not an error
bound or a semantic motion detector.

Only retained section residuals and probe descriptors occupy the explicit
cache budget. Long sections get admission priority. The cache also checks
current GPU headroom and retains at least 1 GiB free at admission. This is not
a cap on all model/activation memory, and does not guarantee an OOM-free render.
Sections that do not fit execute normally. Optional cache-allocation OOM disables
that section; underlying model errors still propagate. Buffers are released on
success, failure and cancellation; nothing persists between runs.

There is no all-layer historical KV bank and no SSD cache. With CPU/offloaded
weights, ComfyUI can still prefetch weights for skipped blocks. Consequently,
compute saved is not a promise of equal wall-time savings. No core offloading
or prefetch code is changed.

For the cache design space, see
[First Block Cache](https://github.com/chengzeyi/ParaAttention#first-block-cache-our-dynamic-caching)
and [section caching](https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit).
Their published performance figures do not measure this implementation.

## Status and compatibility

The existing status message and console report mode, actual sparse calls,
reused blocks, section-cache hits and peak retained cache bytes. Reasons for
unavailable paths are logged. A reused-block count is not a skipped sampler step
and does not imply a measured speedup. Memory correction calls remain visible.

Sol uses the native H3 packed layout for text, first-frame, first/last and
reference conditioning. Cache/guidance CPU integration tests exercise all four.
The original MODEL/options and sampler callbacks are restored after execution.
Spectrum and other generation caches must be turned off explicitly: combining
uncoordinated caches can bypass corrections or reuse invalid state. Selective
caching rejects existing double_block replacements rather than skipping them.

No frontend or core changes are required. If optional controls do not appear
after updating, restart ComfyUI and refresh the browser's node definitions.

## RAM-backed storage and adaptive Sol

Existing defaults remain `cache_storage=gpu` and `adaptive_sol=false`, preserving
the previously working Sol behavior. On VRAM-constrained systems, test these
additional controls in order:

1. Select `sol_and_cache`, set `cache_storage=ram`, and keep `cache_ram_gib=2`
   and `cache_transfer_mib=16`. Keep your cache threshold and reuse cap unchanged.
   Check **section hits/reused blocks**: RAM allocation alone does not mean reuse.
2. Compare against `cache_storage=auto` if desired. Auto admits GPU residuals
   when they fit and otherwise uses RAM. A storage migration refreshes the section.
3. Enable `adaptive_sol=true` and compare the same prompts and seeds. Leave
   `sol_stable_boost=0.15` and `sol_drift_threshold=0.04` initially.

RAM storage preserves the activation dtype without additional quantization. It
retains one full residual and a bounded CPU scratch buffer per admitted section,
both counted against `cache_ram_gib`. CUDA captures use pinned CPU buffers.
Transfers and GPU staging are chunked by `cache_transfer_mib`, including capture
and refresh; a full cached residual need not reside on the GPU. Replay still
allocates the normal full block output. The VRAM limit covers retained tensors,
not all model outputs or temporary allocations.

RAM admission retains 512 MiB physical host headroom when psutil is available;
GPU-backed residual admission retains 1 GiB GPU headroom. The RAM path instead
checks its small staging allocation with 256 MiB GPU headroom. These admission
checks cannot guarantee an OOM-free render. Optional capture/replay allocation
failure disables the affected section and continues with real computation;
underlying model failures still propagate.

Copies are deliberately synchronous and ordered, avoiding a race between CPU
buffer reuse and outstanding CUDA transfers. PCIe traffic and CPU arithmetic can
outweigh saved computation; there is no promised end-to-end gain. No disk writes
are added. All RAM/VRAM cache tensors are released on exit. Status distinguishes
peak GPU and RAM cache storage; structured metrics also count transferred bytes.

Adaptive Sol uses the existing original-latent prediction fingerprints when
available. Fixed observation schedules may need an additional bounded thumbnail
calculation, never another Tiny VAE decode. Two consecutive stable prediction
comparisons allow the configured tau boost. A large change immediately selects
a denser threshold (base minus 0.35); the last 10 percentage points before the
configured acceleration end cap tau at base minus 0.2. Thresholds stay in [0,2].
Missing/failed observations remove the stability boost. This heuristic is not a
semantic motion detector or a quality guarantee.

Tau changes invalidate residual caches before another reuse; otherwise a cache
could replay calculations made under different attention behavior. This may
reduce cache hits. Protected memory/boundary/probe blocks and conditioning/audio
sink behavior are unchanged. Adaptive Sol can be slower on changing clips; keep
it optional and compare against the fixed Sol setting that already works.

Validation includes CPU native H3 sampling with real Tiny VAE decoding across
text, first-frame, first/last-frame and reference conditioning, RAM/GPU-storage
arithmetic equivalence, budget guards, cache invalidation, and cancellation.
A small RTX 4090 kernel probe also checked pinned RAM replay in FP32/FP16/BF16
and real Sol calls at changing tau. These tests do not establish trained-video
quality or end-to-end acceleration for these new options.

"""Native H3 image/reference rows retained as an immutable causal prefix.

The prefix is cache record zero, alongside text. Its rows are never selected
as generated-history candidates or evicted from the recent-media window.
"""

from dataclasses import dataclass
import math

import torch

from . import layout as layout_mod
from .contracts import ContractError

IMAGE_KEYS = frozenset(("minimax_keyframes", "minimax_refs", "minimax_visual_cond_noise_aug",
                        "minimax_audio_cond_noise_aug"))


def _latent(value, name, *, audio=False):
    ndim, channels = (4, 32) if audio else (5, 24)
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ContractError(f"{name} must be a native H3 {'audio' if audio else 'video'} latent")
    if value.shape[0] != 1 or value.shape[1] != channels or min(value.shape) < 1:
        raise ContractError(f"Invalid {name} shape: {tuple(value.shape)}")
    if audio and value.shape[2] != 2:
        raise ContractError(f"{name} must contain stereo H3 audio")
    if not audio and (value.shape[-2] % 2 or value.shape[-1] % 2):
        raise ContractError(f"{name} spatial dimensions must be divisible by the H3 2x2 patch")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ContractError(f"{name} must contain finite floating-point latents")
    return value


@dataclass(frozen=True)
class ImageContext:
    keyframes: tuple = ()
    refs: tuple = ()
    visual_aug: float = .999
    audio_aug: float = 1.

    @property
    def active(self):
        return bool(self.keyframes or self.refs)

    def payload(self, seed):
        items = self.keyframes + self.refs
        return dict(keyframes=self.keyframes, refs=self.refs,
                    cond_video_latents=[r['latent'] for r in items if r.get('latent') is not None],
                    cond_audio_latents=[r['audio_latent'] for r in items if r.get('audio_latent') is not None],
                    visual_cond_noise_aug=self.visual_aug, audio_cond_noise_aug=self.audio_aug, seed=int(seed))

    def validate_schedule(self, config):
        # Native H3 labels reference rows max(target_t, augmentation). Caching
        # those rows is valid only while that label stays fixed across every
        # noisy step and the .999 clean-context fills.
        for name, augmentation, sigmas in (('video',self.visual_aug,config.video_sigmas),
                                           ('audio',self.audio_aug,config.audio_sigmas)):
            if max(.999,1.-min(sigmas)) > augmentation + 1e-7:
                raise ContractError(f"{name} conditioning timestep changes over this schedule; use native condition augmentation (.999 video / 1 audio) and the default four-step shifts")


def split_conditioning(positive, request):
    """Validate condition rows separately, retaining strict text-extra checks."""
    if not isinstance(positive, (list, tuple)) or len(positive) != 1:
        raise ContractError("Image memory requires one positive conditioning entry")
    cross, extras = positive[0]
    extras = dict(extras or {})
    keyframes, refs = [], []
    for item in extras.get('minimax_keyframes') or ():
        if not isinstance(item, dict) or set(item) - {'resolved_frame_index', 'latent', 'audio_latent'}:
            raise ContractError("Unsupported keyframe fields in H3 memory conditioning")
        item = dict(item)
        frame = item.get('resolved_frame_index')
        if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < request.frames:
            raise ContractError("Keyframe index must be an integer inside the target clip")
        if item.get('latent') is None and item.get('audio_latent') is None:
            raise ContractError("A keyframe must contain video or audio latents")
        if item.get('latent') is not None:
            z = _latent(item['latent'], 'keyframe')
            if tuple(z.shape[-2:]) != (request.latent_h, request.latent_w):
                raise ContractError("Keyframe latent canvas must match the target")
            # Native first/last frames have one temporal latent. Guide clips
            # remain supported if their decoded temporal span fits the target.
            pixels = sum(layout_mod.FRAME_PER_TOKEN[k % 5] for k in range(z.shape[2]))
            if frame + pixels > request.frames:
                raise ContractError("Keyframe/guide clip extends beyond the target video")
        if item.get('audio_latent') is not None:
            z = _latent(item['audio_latent'], 'keyframe audio', audio=True)
            if layout_mod.FRAME_RESCALE * frame + z.shape[-1] > request.audio_t + 1e-6:
                raise ContractError("Keyframe audio extends beyond the target soundtrack")
        keyframes.append(item)
    for item in extras.get('minimax_refs') or ():
        if not isinstance(item, dict) or set(item) - {'kind', 'latent', 'audio_latent', 'latent_t',
                                                     'latent_h', 'latent_w', 'ref_audio_t'}:
            raise ContractError("Unsupported reference fields in H3 memory conditioning")
        item = dict(item)
        kind = item.get('kind')
        if kind not in ('image', 'audio', 'video', 'video_audio'):
            raise ContractError(f"Unsupported H3 reference kind: {kind}")
        if kind != 'audio':
            z = _latent(item.get('latent'), 'reference')
            if (item.get('latent_h'), item.get('latent_w')) != tuple(z.shape[-2:]):
                raise ContractError("Reference latent canvas metadata does not match its tensor")
            if kind == 'image' and z.shape[2] != 1:
                raise ContractError("An image reference must have one temporal latent")
            if kind in ('video', 'video_audio') and item.get('latent_t') != z.shape[2]:
                raise ContractError("Reference video length metadata does not match its tensor")
        if kind != 'image':
            rt = item.get('ref_audio_t', 0)
            if not isinstance(rt, int) or isinstance(rt, bool) or rt < 0:
                raise ContractError("Invalid reference audio length")
            z = item.get('audio_latent')
            if rt:
                z = _latent(z, 'reference audio', audio=True)
                if z.shape[-1] != rt:
                    raise ContractError("Reference audio length metadata does not match its tensor")
            elif z is not None:
                raise ContractError("Reference audio tensor supplied with zero declared length")
            if kind in ('audio', 'video_audio') and not rt:
                raise ContractError("Audio reference must contain audio latents")
        elif item.get('audio_latent') is not None:
            raise ContractError("Use a video_audio or audio reference for audio conditioning")
        refs.append(item)
    aug = []
    for key, default in [('minimax_visual_cond_noise_aug', .999), ('minimax_audio_cond_noise_aug', 1.)]:
        value = float(extras.get(key, default))
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ContractError(f"{key} must be finite and in (0, 1]")
        aug.append(value)
    return [[cross, {k: v for k, v in extras.items() if k not in IMAGE_KEYS}]], ImageContext(
        tuple(keyframes), tuple(refs), *aug)


class ConditionedLayout:
    """Keep native reference clocks distinct from the number of prefix rows."""
    def __init__(self, base, context):
        from comfy.ldm.minimax.model import PackedLayout
        self.base = base
        native = PackedLayout(base.text_len, base.latent_t, base.latent_h, base.latent_w, base.audio_t,
                              keyframes=context.keyframes, refs=context.refs)
        self.prefix_segments = tuple(s for s in native.segments if s[2] not in ('video', 'audio'))
        self.prefix_rows = self.prefix_segments[-1][1]
        self.prefix_positions = native.position_ids[:self.prefix_rows].clone()
        target = next(a for a, b, kind in native.segments if kind == 'audio')
        self.target_origin = float(native.position_ids[target, 0])

    def __getattr__(self, name):
        return getattr(self.base, name)

    def chunk_position_ids(self, index):
        positions = self.base.chunk_position_ids(index).clone()
        positions[:, 0] += self.target_origin - self.base.text_len
        return positions


@torch.no_grad()
def prefill_conditioned(model, context, conditioning, layout, cache, options, dtype, device, seed):
    from .causal_model import _fp32_scalar, _text_mod_segments, CLEAN_TIMESTEP_TEXT, CausalModelError
    if cache.committed_chunks or cache.has_pending:
        raise CausalModelError("Conditioned prefix requires an empty cache")
    payload = context.payload(seed)
    video = model._cond_video_rows(payload, device)
    audio = model._cond_audio_rows(payload, device)
    text = conditioning.cross_attn[0].to(device=device, dtype=dtype)
    if text.shape[-1] != model.hidden_size:
        text = model._causal_refine_text(text, options)
    times = sorted({_fp32_scalar(CLEAN_TIMESTEP_TEXT), _fp32_scalar(context.visual_aug),
                    _fp32_scalar(context.audio_aug)})
    labels = {t: i for i, t in enumerate(times)}
    h, mods = [], []
    vi = ai = 0
    for start, stop, kind in layout.prefix_segments:
        n = stop - start
        if kind == 'text':
            h.append(text.clone())
            base = labels[_fp32_scalar(CLEAN_TIMESTEP_TEXT)] * 3
            mods.extend((a, b, row + base) for a, b, row in _text_mod_segments(n, conditioning.token_tags))
        elif kind in ('cond', 'ref_img'):
            h.append(model.video_patch_proj(video[vi:vi+n]).to(dtype))
            mods.append((start, stop, labels[_fp32_scalar(context.visual_aug)] * 3 + layout_mod.VIDEO_TAG))
            vi += n
        else:
            h.append(model.audio_patch_proj(audio[ai:ai+n]).to(dtype))
            mods.append((start, stop, labels[_fp32_scalar(context.audio_aug)] * 3 + layout_mod.AUDIO_TAG))
            ai += n
    if vi != (0 if video is None else len(video)) or ai != (0 if audio is None else len(audio)):
        raise CausalModelError("Conditioned prefix rows do not match their native packed layout")
    t_emb, adaln = model._causal_time_embeddings(times, device, dtype)
    rope = model._rope_table(layout.prefix_positions, device, dtype)
    model._run_blocks(torch.cat(h), t_emb, mods, rope, options, cache, True, adaln_input=adaln)
    cache.commit(role='text')

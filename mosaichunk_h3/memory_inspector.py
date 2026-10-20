"""Bounded, ephemeral diagnostic images and opt-in reproducibility reports."""
from __future__ import annotations

from collections import deque
import base64
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import platform
import uuid


class MemoryInspector:
    def __init__(self, *, enabled=False):
        self.enabled = enabled
        self.history = deque(maxlen=32)
        self.run_id = uuid.uuid4().hex

    def observation(self, frames, plans, *, step, events=(), video_frames=None, frame_groups=None):
        if not self.enabled:
            return None
        from PIL import Image, ImageDraw
        import torch
        if not plans:
            from .region_memory import RegionPlan
            plans = (RegionPlan(0,0,'preview only; no accepted pair',(0,0,1,1),
                                (0,0,1,1),0.,0,0),)
        cards = []
        for plan in plans[:4]:
            images = []
            for frame_id, box, label in ((plan.source_frame, plan.source_box, 'source'),
                                         (plan.target_frame, plan.target_box, 'target')):
                f = frames[frame_id].detach().cpu()
                if f.dtype != torch.uint8:
                    f = f.float().clamp(0, 1).mul(255).byte()
                image = Image.fromarray(f.permute(1, 2, 0).numpy())
                image.thumbnail((192, 192))
                draw = ImageDraw.Draw(image)
                draw.rectangle(tuple(v*s for v,s in zip(box, (image.width,image.height)*2)),
                               outline='#51d88a', width=2)
                stream = io.BytesIO()
                image.save(stream, format='JPEG', quality=70)
                images.append(dict(role=label, frame=frame_id,
                    video_frame=video_frames[frame_id] if video_frames is not None else frame_id,
                    latent_frame=frame_groups[frame_id] if frame_groups is not None else frame_id,
                    image=base64.b64encode(stream.getvalue()).decode()))
            cards.append(dict(region=plan.name, confidence=round(plan.confidence, 3), images=images))
        entry = dict(step=step, plans=cards, events=list(events)[:16])
        self.history.append(entry)
        return entry

    def snapshot(self):
        return dict(run_id=self.run_id, observations=list(self.history))


def write_report(directory, *, settings, seed, metrics, inspector=None, error=None):
    """No prompts, images or arbitrary client-selected filesystem paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(schema=1, run_id=inspector.run_id if inspector else uuid.uuid4().hex,
                  timestamp=datetime.now(timezone.utc).isoformat(), seed=seed,
                  settings=settings, metrics=metrics, error=error, python=platform.python_version())
    import torch
    report['torch'] = torch.__version__
    try:
        import comfyui_version
        report['comfyui'] = comfyui_version.__version__
    except ImportError:
        report['comfyui'] = 'unknown'
    source = Path(__file__).resolve().parent
    import hashlib
    report['implementation_sha256'] = hashlib.sha256(b''.join(
        (source/name).read_bytes() for name in ('native_guidance.py','region_memory.py',
                                              'generation_acceleration.py'))).hexdigest()
    name = f'minimax-memory-{report["run_id"]}.json'
    pending = directory / (name+'.partial')
    pending.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    pending.replace(directory/name)
    return str(directory/name)

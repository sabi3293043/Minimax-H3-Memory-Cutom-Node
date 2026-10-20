"""Bounded region retrieval and immutable RGB references, independent of torch KV."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math


@dataclass(frozen=True)
class Region:
    name: str
    box: tuple


def parse_regions(value):
    try:
        items = json.loads(value or '[]')
    except (ValueError, TypeError) as exc:
        raise ValueError('memory_regions must be JSON: [{"name":"outfit","box":[0.2,0.1,0.8,1]}]') from exc
    if not isinstance(items, list) or len(items) > 4:
        raise ValueError('Select at most four memory regions')
    result = []
    for item in items:
        if not isinstance(item, dict) or set(item) != {'name', 'box'}:
            raise ValueError('Each memory region needs exactly name and box')
        name, box = item['name'], item['box']
        if not isinstance(name, str) or not name.strip() or len(name) > 48:
            raise ValueError('Region names must contain 1-48 characters')
        if name in [r.name for r in result]:
            raise ValueError('Region names must be unique')
        if (not isinstance(box, list) or len(box) != 4
                or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in box)
                or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
            raise ValueError('Region box must be normalized [left,top,right,bottom] within 0..1')
        result.append(Region(name, tuple(float(v) for v in box)))
    return tuple(result)


def crop(frame, box):
    h, w = frame.shape[-2:]
    x0, y0, x1, y1 = box
    left, top = int(x0*w), int(y0*h)
    return frame[:, top:max(top+1, math.ceil(y1*h)), left:max(left+1, math.ceil(x1*w))]


def descriptor(frame, box):
    import torch
    import torch.nn.functional as F
    value = crop(frame, box).detach().float()
    if frame.dtype == torch.uint8:
        value = value / 255.
    return F.adaptive_avg_pool2d(value[None], (6, 6)).flatten().clamp(0, 1).cpu()


def grid_ids(box, height, width, *, device=None, mask=None):
    """Latent patch centers within the box/mask; never select text/audio tokens."""
    import torch
    y = (torch.arange(height, device=device) + .5) / height
    x = (torch.arange(width, device=device) + .5) / width
    valid = ((y[:, None] >= box[1]) & (y[:, None] < box[3])
             & (x[None, :] >= box[0]) & (x[None, :] < box[2]))
    if mask is not None:
        import torch.nn.functional as F
        m = torch.as_tensor(mask, dtype=torch.float32, device=device)
        m = F.interpolate(m[None, None], (height, width), mode='nearest')[0, 0] > .5
        valid &= m
    return valid.flatten().nonzero().flatten()


def move_region(box, anchor, target):
    """Preserve the user's subregion when SAM returns a whole-object mask."""
    ax,ay,ar,ab = anchor
    tx,ty,tr,tb = target
    relative = ((box[0]-ax)/max(ar-ax,1e-6), (box[1]-ay)/max(ab-ay,1e-6),
                (box[2]-ax)/max(ar-ax,1e-6), (box[3]-ay)/max(ab-ay,1e-6))
    relative = tuple(max(0.,min(1.,v)) for v in relative)
    moved = (tx+relative[0]*(tr-tx),ty+relative[1]*(tb-ty),
             tx+relative[2]*(tr-tx),ty+relative[3]*(tb-ty))
    return moved if moved[0]<moved[2] and moved[1]<moved[3] else None


@dataclass(frozen=True)
class RegionPlan:
    source: int
    target: int
    name: str
    source_box: tuple
    target_box: tuple
    confidence: float
    source_frame: int
    target_frame: int
    source_mask: object = None
    target_mask: object = None


class RegionMemory:
    """Frozen descriptors detect bad anchors; current-call KV still provides guidance.

    A bank holds at most four regions times eight sources, each 108 CPU floats.
    No old-step attention tensors are retained or reinterpreted at another sigma.
    """
    def __init__(self, regions, protected=(), *, drift_limit=.15):
        self.regions = regions
        self.protected = tuple(protected)
        self.drift_limit = drift_limit
        self.bank = {}
        self.events = []

    def plan(self, frames, *, groups, step, source_frames=(), tracked=None):
        self.events = []
        proposals = []
        # Explicit source frames take precedence, otherwise use the first frame.
        sources = tuple(dict.fromkeys(source_frames or self.protected))[:8] or (0,)
        for source in sources:
            if not 0 <= source < len(frames):
                continue
            for region in self.regions:
                key = (source, region.name)
                source_box = region.box
                anchor_record = source_record = None
                if tracked is not None:
                    anchor_record = tracked.get((min(sources),region.name))
                    source_record = tracked.get((source,region.name))
                    if (not anchor_record or not source_record or not anchor_record.get('visible')
                            or not source_record.get('visible')):
                        continue
                    source_box = move_region(region.box,anchor_record['box'],source_record['box'])
                    if source_box is None:
                        continue
                current = descriptor(frames[source], source_box)
                prior = self.bank.get(key)
                if prior is None:
                    # Wait for a second stable observation before trusting a reference.
                    self.bank[key] = (current, False, step)
                    self.events.append(dict(region=region.name, source=source, reason='warming reference'))
                    continue
                frozen, trusted, previous_step = prior
                drift = float((frozen-current).abs().mean())
                if step <= previous_step:
                    continue
                if not trusted:
                    if drift <= .075:
                        self.bank[key] = (current.clone(), True, step)
                        frozen = current
                    else:
                        self.bank[key] = (current, False, step)
                        continue
                elif drift > self.drift_limit:
                    self.events.append(dict(region=region.name, source=source,
                                            reason='source appearance drift; correction withheld', drift=drift))
                    continue
                for target in range(source+1, len(frames)):
                    if groups[target] == groups[source] or target in sources:
                        continue
                    box = region.box
                    visibility = 1.
                    if tracked is not None:
                        record = tracked.get((target, region.name))
                        if record is None or not record.get('visible', False):
                            continue  # Untracked frames never borrow stale masks.
                        box = move_region(region.box,anchor_record['box'],record['box'])
                        if box is None:
                            continue
                        visibility = float(record['confidence'])
                    distance = float((frozen-descriptor(frames[target], box)).abs().mean())
                    if distance > .15:
                        continue
                    confidence = max(0., 1.-distance/.15) * visibility
                    if confidence >= .5:
                        source_mask = target_mask = None
                        if tracked is not None:
                            source_mask, target_mask = source_record['mask'], record['mask']
                        proposals.append(RegionPlan(groups[source], groups[target], region.name,
                            source_box, box, confidence, source, target, source_mask, target_mask))
        # Select a diverse set; source frames are immutable within this evaluation.
        proposals.sort(key=lambda item: (-item.confidence, item.target, item.name))
        result, seen = [], set()
        by_region = {region.name: [] for region in self.regions}
        for proposal in proposals:
            key = (proposal.target, proposal.name)
            if key not in seen:
                by_region[proposal.name].append(proposal)
                seen.add(key)
        # Round-robin prevents easy background matches consuming the whole budget.
        while len(result) < 8:
            for candidates in by_region.values():
                if candidates:
                    result.append(candidates.pop(0))
                if len(result) == 8:
                    break
            if not any(by_region.values()):
                break
        return tuple(result)

    def clear(self):
        self.bank.clear()
        self.events.clear()

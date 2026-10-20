"""Bounded SAM mask selection; no imports, downloads or model loading here."""
import math


def _spread_frames(count, selected, limit):
    """Fill the largest temporal gaps without allocating image tensors."""
    selected = set(selected)
    start = min(selected)
    selected.add(count - 1)
    while len(selected) < min(limit, count - start):
        ordered = sorted(selected)
        left, right = max(zip(ordered, ordered[1:]), key=lambda pair: pair[1] - pair[0])
        if right - left <= 1:
            break
        selected.add((left + right) // 2)
    return tuple(sorted(selected))


def select_tracking_frames(count, *, sources=(0,), pairs=(), budget=8, mode='retrieved'):
    """Select indices in ONE current decoded observation, never old-step RGB.

    Protected sources cannot be evicted. Retrieved pair endpoints take priority;
    two remaining slots provide a bridge and discovery near the clip's end.
    Before retrieval warms up, use bounded temporal coverage instead.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('Tracking needs a nonempty current observation')
    if isinstance(budget, bool) or not isinstance(budget, int) or not 4 <= budget <= 16:
        raise ValueError('Tracking frame budget must be 4..16')
    if mode not in ('retrieved', 'uniform'):
        raise ValueError('Unknown tracking frame selection')
    sources = tuple(sorted(set(sources or (0,))))
    if len(sources) > 8 or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < count for i in sources):
        raise ValueError('Tracking sources must be at most eight valid observation indices')
    pairs = tuple(pairs)
    if any(len(pair) != 2 or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < count
                               for i in pair) for pair in pairs):
        raise ValueError('Tracking pairs must refer to current observation indices')
    limit = min(count, max(budget, len(sources)))
    # Reserve discovery even when protected sources fill the requested budget.
    if count - 1 not in sources and len(sources) >= limit:
        limit = min(count, limit + 1)
    selected = set(sources)
    if mode == 'retrieved':
        endpoint_limit = max(len(selected), limit - 2, min(limit - 1, len(selected) + 2))
        for source, target in pairs:
            if not sources[0] <= source < target:
                continue
            proposal = selected | {source, target}
            if len(proposal) <= endpoint_limit:
                selected = proposal
        # One midpoint within a retrieved interval, then global discovery/coverage.
        if len(selected) > len(sources) and len(selected) < limit - 1:
            ordered = sorted(selected)
            left, right = max(zip(ordered, ordered[1:]), key=lambda pair: pair[1] - pair[0])
            if right - left > 1:
                selected.add((left + right) // 2)
    return _spread_frames(count, selected, limit)


def expand_tracking_frames(count, selected):
    """Keep every initial index and add coverage, bounded to 16 shared frames."""
    return _spread_frames(count, selected, 16)


def tracking_record_trusted(record):
    return bool(record and record.get('visible', False)
                and math.isfinite(float(record.get('confidence', 0.)))
                and float(record.get('confidence', 0.)) >= .65)


def tracking_uncertain(records, ids, regions):
    """Conservative expansion trigger; absence is never treated as visibility."""
    if not regions:
        return False
    for frame in ids:
        for region in regions:
            record = records.get((frame, region.name))
            if not tracking_record_trusted(record):
                return True
    return False


def automatic_regions(masks, width, height, *, limit=4):
    """Choose diverse confident object-sized masks, never a whole-image prompt."""
    proposals = []
    for item in masks:
        quality = float(item.get('predicted_iou',0.)) * float(item.get('stability_score',0.))
        area = float(item.get('area',0.)) / max(1,width*height)
        if quality < .65 or not .01 <= area <= .8:
            continue
        x,y,w,h = (float(v) for v in item['bbox'])
        box = (max(0.,x/width),max(0.,y/height),min(1.,(x+w)/width),min(1.,(y+h)/height))
        if not box[0] < box[2] or not box[1] < box[3]:
            continue
        if (box[2]-box[0])*(box[3]-box[1]) > .95:
            continue
        proposals.append((quality*math.sqrt(area),box,item['segmentation']))
    proposals.sort(key=lambda entry: entry[0],reverse=True)
    selected = []
    for _,box,mask in proposals:
        duplicate = False
        area = (box[2]-box[0])*(box[3]-box[1])
        for prior in selected:
            b = prior['box']
            overlap = max(0.,min(box[2],b[2])-max(box[0],b[0])) * max(0.,min(box[3],b[3])-max(box[1],b[1]))
            union = area+(b[2]-b[0])*(b[3]-b[1])-overlap
            if overlap/max(union,1e-8) > .65:
                duplicate = True
                break
        if not duplicate:
            selected.append(dict(name=f'auto object {len(selected)+1}',box=box,segmentation=mask))
        if len(selected) == limit:
            break
    return selected

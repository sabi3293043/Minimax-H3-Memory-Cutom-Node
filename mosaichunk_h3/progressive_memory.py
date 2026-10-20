"""Bounded resolution-specific RGB history; no cross-scale descriptor comparisons."""
from dataclasses import dataclass


@dataclass
class ReferenceView:
    size: tuple
    index: object
    regions: object


class ProgressiveReferenceMemory:
    """Share the existing observation budget across at most three decode scales.

    Earlier views remain usable while a larger view learns more detailed RGB
    references. Each view freezes its own regional descriptors. Normal eviction
    is shared across scales; changing resolution alone never evicts a record.
    """
    def __init__(self, *, capacity, captures_per_step, protected_frames, regions,
                 index_factory, region_factory):
        self.capacity = capacity
        self.captures_per_step = captures_per_step
        self.protected = tuple(protected_frames)
        self.regions = regions
        self.index_factory = index_factory
        self.region_factory = region_factory
        self.views = {}

    def view(self, size):
        size = tuple(size)
        if size not in self.views:
            if len(self.views) == 3:
                raise ValueError('Progressive reference memory supports at most three decode sizes')
            self.views[size] = ReferenceView(size,
                self.index_factory(capacity=self.capacity, captures_per_step=self.captures_per_step,
                                   protected_frames=self.protected),
                self.region_factory(self.regions, self.protected))
        return self.views[size]

    def put(self, view, step, frame, descriptor, confidence, record, *, on_evict):
        if not any(e.frame == frame for e in view.index.entries):
            entries = [(v, e) for v in self.views.values() for e in v.index.entries]
            if len(entries) >= self.capacity:
                unprotected = [(v,e) for v,e in entries if e.frame not in self.protected]
                if not unprotected:
                    return False
                owner, weakest = min(unprotected, key=lambda pair: (pair[1].confidence, pair[1].step))
                if confidence < weakest.confidence:
                    return False
                on_evict(weakest.record)
                owner.index.entries.remove(weakest)
        return view.index.put(step, frame, descriptor, confidence, record, on_evict=on_evict)

    def close(self):
        for view in self.views.values():
            view.regions.clear()


def merge_references(batches, *, groups, count, forward_only):
    """Prefer current-detail proposals, then older compatible-scale matches."""
    if count <= 0:
        return []
    chosen, targets, anchors = [], set(), set()
    for batch in batches:
        for reference in batch:
            source, target = groups[reference.source_frame], groups[reference.query_frame]
            if target in targets:
                continue
            if forward_only and (source >= target or target in anchors or source in targets):
                continue
            chosen.append(reference)
            targets.add(target)
            anchors.add(source)
            if len(chosen) >= count:
                return chosen
    return chosen

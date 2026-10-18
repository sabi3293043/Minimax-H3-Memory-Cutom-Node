"""Read-only, cross-step visual-memory retrieval for native H3.

This identifies candidate historical observations that MAY be useful for later
denoising. Thumbnail similarity is not proof of identity. This module never
modifies sampler latents, attention weights, or model output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MemoryReference:
    query_frame: int
    source_frame: int
    source_step: int
    source_record: Any
    distance: float
    confidence: float


def retrieve_step_references(
    descriptors,
    observations,
    *,
    step: int,
    max_references: int = 8,
    min_temporal_gap: int = 2,
    max_distance: float = 0.075,
    min_confidence: float = 0.50,
):
    """Rank cross-frame references retained from earlier denoising steps.

    Inputs are compact CPU [T,D] descriptors and StoredObservation entries.
    This is a retrieval/proposal boundary, NOT a correction signal.
    Rejects sources from the same/future step and nearby identical frames.
    """
    import torch

    if not isinstance(descriptors, torch.Tensor) or descriptors.ndim != 2:
        raise ValueError("descriptors must be a [frames,features] tensor")
    if step < 0 or max_references < 0 or min_temporal_gap < 1:
        raise ValueError("invalid step, budget, or temporal gap")
    if not 0 <= max_distance <= 1 or not 0 <= min_confidence <= 1:
        raise ValueError("distance/confidence thresholds must be within [0,1]")
    if max_references == 0 or descriptors.shape[0] == 0:
        return []
    queries = descriptors.detach().float().cpu()
    refs = [item for item in observations
            if item.step < step and item.confidence >= min_confidence]
    if not refs:
        return []
    if any(item.descriptor.shape != queries.shape[1:] for item in refs):
        raise ValueError("reference descriptor dimensions do not match query")
    pool = torch.stack([item.descriptor.detach().float().cpu() for item in refs], 0)
    distances = (queries[:, None, :] - pool[None, :, :]).abs().mean(dim=2)
    proposals = []
    for query_frame in range(queries.shape[0]):
        for j, item in enumerate(refs):
            if abs(query_frame - item.frame) < min_temporal_gap:
                continue
            distance = float(distances[query_frame, j])
            if distance <= max_distance:
                proposals.append(MemoryReference(query_frame=query_frame,
                    source_frame=item.frame, source_step=item.step,
                    source_record=item.record, distance=distance,
                    confidence=float(item.confidence)))
    # Prefer confident, nearby references; only one source per queried frame.
    proposals.sort(key=lambda r: (r.distance - 0.025 * r.confidence, r.query_frame))
    chosen = []
    seen = set()
    for proposal in proposals:
        if proposal.query_frame in seen:
            continue
        chosen.append(proposal)
        seen.add(proposal.query_frame)
        if len(chosen) == max_references:
            break
    return chosen

// Presets update visible widgets; the server always honors explicit widget values.
const memory = {
  guidance_mode: 'retrieved_frames', guidance_strength: .10,
  guidance_similarity: .85, guidance_layers: '8,16,24',
  max_frames: 8, max_memory_frames: 64, max_anchor_tokens: 256,
  observation_schedule: 'fixed', observation_interval: 1,
  progressive_resolution: false, min_progress: .25,
  store_observation_images: false, adaptive_sol: false,
  cache_storage: 'ram', cache_ram_gib: 2, cache_transfer_mib: 16,
  cache_max_reuse: 1,
}
export const PRESETS = {
  Consistency: { ...memory, generation_acceleration: 'off', sol_tau: .5,
    acceleration_start: .35, acceleration_end: .75, cache_threshold: .02 },
  Balanced: { ...memory, generation_acceleration: 'sol_and_cache', sol_tau: .5,
    acceleration_start: .35, acceleration_end: .75, cache_threshold: .02 },
  Speed: { ...memory, generation_acceleration: 'sol_and_cache', sol_tau: 1,
    acceleration_start: .25, acceleration_end: .85, cache_threshold: .05 },
}

export function matchesPreset(node, name) {
  const values = PRESETS[name]
  return !!values && Object.entries(values).every(([key, value]) =>
    node.widgets?.find(widget => widget.name === key)?.value === value)
}

export function applyPreset(node, name) {
  const values = PRESETS[name]
  if (!values) return
  for (const [key, value] of Object.entries(values)) {
    const widget = node.widgets?.find(widget => widget.name === key)
    if (widget) widget.value = value
  }
  node.setDirtyCanvas?.(true, true)
}

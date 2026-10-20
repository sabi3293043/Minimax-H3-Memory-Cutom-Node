import { app } from '../../scripts/app.js'
import { api } from '../../scripts/api.js'
import { PRESETS, matchesPreset, applyPreset } from './memory_presets.js'

const TYPE = 'MosaiChunkH3NativeVisualObserver'
const states = new WeakMap()
const advanced = new Set(['memory_regions', 'sol_tau', 'acceleration_start', 'acceleration_end',
  'max_frames','max_memory_frames','min_progress','guidance_source_frame','guidance_target_frames',
  'guidance_layers','guidance_similarity','max_anchor_tokens','observation_interval',
  'observation_schedule','progressive_resolution','store_observation_images','export_memory_report',
  'cache_threshold', 'cache_max_reuse', 'cache_vram_gib', 'cache_storage', 'cache_ram_gib',
  'cache_transfer_mib', 'adaptive_sol', 'sol_stable_boost', 'sol_drift_threshold',
  'adaptive_change_threshold', 'adaptive_max_interval', 'observation_start_scale',
  'observation_full_progress', 'max_observation_megapixels', 'disk_gib'])

function widget(node, name) { return node.widgets?.find(w => w.name === name) }
function state(node) {
  if (!states.has(node)) states.set(node, { history: [], detail: null, panel: null, expanded: false })
  return states.get(node)
}
function element(tag, text, parent) {
  const el = document.createElement(tag)
  if (text !== undefined) el.textContent = text
  parent?.append(el)
  return el
}
function button(parent, text, action) {
  const el = element('button', text, parent)
  el.type = 'button'
  el.onclick = action
  return el
}
function changed(node) {
  const preset = widget(node, 'memory_preset')
  if (preset && !matchesPreset(node, preset.value)) preset.value = 'Custom'
  node.setDirtyCanvas?.(true, true)
}
function download(value) {
  const blob = new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const anchor = element('a')
  anchor.href = url
  anchor.download = `minimax-memory-${value.run_id ?? 'report'}.json`
  anchor.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
function toggleAdvanced(node) {
  const data = state(node)
  data.expanded = !data.expanded
  for (const control of node.widgets ?? []) {
    if (!advanced.has(control.name)) continue
    if (!control._memoryOriginal) control._memoryOriginal = { type: control.type, computeSize: control.computeSize }
    control.type = data.expanded ? control._memoryOriginal.type : 'hidden'
    control.computeSize = data.expanded ? control._memoryOriginal.computeSize : () => [0, -4]
    control.hidden = !data.expanded
    if (control.inputEl) control.inputEl.style.display = data.expanded ? '' : 'none'
  }
  node.setSize?.([node.size?.[0] ?? 440, node.computeSize?.()[1] ?? 600])
  node.setDirtyCanvas?.(true, true)
}

export function openInspector(node) {
  const data = state(node)
  if (data.panel) { data.panel.focus(); return }
  const panel = element('dialog')
  data.panel = panel
  panel.className = 'minimax-memory-inspector'
  panel.setAttribute('aria-label', 'MiniMax H3 Memory inspector')
  const style = element('style', undefined, panel)
  style.textContent = `.minimax-memory-inspector {color:#eee;background:#222;border:1px solid #666;border-radius:12px;width:min(900px,90vw);max-height:85vh;overflow:auto;padding:20px;font:14px system-ui}
    .minimax-memory-inspector button,.minimax-memory-inspector select,.minimax-memory-inspector input {color:#eee;background:#383838;border:1px solid #777;border-radius:5px;padding:7px;margin:4px}
    .minimax-memory-inspector canvas{display:block;margin:8px 0;max-width:100%;touch-action:none;border:1px solid #555;cursor:crosshair}
    .minimax-memory-inspector .cards{display:flex;flex-wrap:wrap;gap:12px}
    .minimax-memory-inspector .card{background:#303030;border-radius:8px;padding:10px;max-width:420px}
    .minimax-memory-inspector img{max-width:180px;margin:4px}
    .minimax-memory-inspector pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
    .minimax-memory-inspector .toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:6px}`
  const titlebar = element('div', undefined, panel)
  titlebar.className = 'toolbar'
  element('h2', 'MiniMax H3 Memory inspector', titlebar)
  button(titlebar, 'Close', () => panel.close())
  element('p', 'Whole-frame memory stays active. Boxes add optional regional priorities. With tracking enabled, empty regions trigger automatic object proposals. Changes apply to the next queued run.', panel)
  const tools = element('div', undefined, panel)
  tools.className = 'toolbar'
  const preset = element('select', undefined, tools)
  preset.setAttribute('aria-label', 'Memory preset')
  for (const name of [...Object.keys(PRESETS), 'Custom']) {
    const option = element('option', name, preset); option.value = name
  }
  preset.value = widget(node, 'memory_preset')?.value ?? 'Custom'
  data.presetSelect = preset
  preset.onchange = () => {
    if (preset.value !== 'Custom') applyPreset(node, preset.value)
    widget(node, 'memory_preset').value = preset.value
    render(node)
  }
  button(tools, 'Download report', () => {
    const detail = data.detail ?? {}
    download({ schema: 1, run_id: detail.run_id, seed: detail.seed,
      settings: detail.settings ?? {}, metrics: { sampling_seconds: detail.sampling_seconds, observations: detail.observations,
        tracking: detail.tracking,
        observation_seconds: detail.observation_seconds, corrections: detail.corrections,
        guidance: detail.guidance_details, acceleration: detail.acceleration },
      node_values: Object.fromEntries((node.widgets ?? []).filter(w => !w.options || w.options.serialize !== false)
        .map(w => [w.name, w.value])) })
  })
  const compare = element('input', undefined, tools)
  compare.type = 'file'; compare.accept = '.json'; compare.multiple = true
  compare.setAttribute('aria-label', 'Compare two memory reports')
  compare.title = 'Select two downloaded memory JSON reports to compare'
  const comparison = element('pre', '', panel)
  compare.onchange = async () => {
    try {
      if (compare.files.length !== 2) throw new Error('Choose exactly two JSON reports.')
      const reports = await Promise.all([...compare.files].map(f => f.text().then(JSON.parse)))
      if (reports.some(r => r.schema !== 1 || !r.metrics || !r.settings)) throw new Error('Unsupported report format.')
      comparison.textContent = JSON.stringify({ same_seed: reports[0].seed === reports[1].seed,
        settings_identical: JSON.stringify(reports[0].settings) === JSON.stringify(reports[1].settings),
        runs: reports.map(r => ({ run: r.run_id, metrics: r.metrics })),
        note: 'Counts/timings do not measure clothing or identity quality. Compare the videos too.' }, null, 2)
    } catch (error) { comparison.textContent = error.message }
  }
  const editor = element('div', undefined, panel)
  element('h3', 'Select a region on an observed source frame', editor)
  const name = element('input', undefined, editor)
  name.value = 'outfit'; name.maxLength = 48; name.setAttribute('aria-label', 'Region name')
  button(editor, 'Clear regions', () => { widget(node,'memory_regions').value = '[]'; changed(node); render(node) })
  const canvas = element('canvas', undefined, editor)
  canvas.width = 384; canvas.height = 256; canvas.setAttribute('aria-label', 'Draw memory region')
  data.canvas = canvas
  let origin = null
  function point(event) {
    const bounds = canvas.getBoundingClientRect()
    return [Math.max(0,Math.min(1,(event.clientX-bounds.left)/bounds.width)),
      Math.max(0,Math.min(1,(event.clientY-bounds.top)/bounds.height))]
  }
  canvas.onpointerdown = event => { origin = point(event); canvas.setPointerCapture(event.pointerId) }
  canvas.onpointerup = event => {
    if (!origin) return
    const end = point(event)
    const box = [Math.min(origin[0],end[0]),Math.min(origin[1],end[1]),Math.max(origin[0],end[0]),Math.max(origin[1],end[1])]
    origin = null
    if (box[2]-box[0]<.02 || box[3]-box[1]<.02) return
    try {
      const regions = JSON.parse(widget(node,'memory_regions').value || '[]')
      const label = name.value.trim() || 'outfit'
      const other = regions.filter(r => r.name !== label)
      if (other.length>=4) throw new Error('At most four regions are supported.')
      other.push({ name: label, box: box.map(v => Math.round(v*1000)/1000) })
      widget(node,'memory_regions').value = JSON.stringify(other)
      changed(node); render(node)
    } catch (error) { comparison.textContent = error.message }
  }
  const summary = element('pre', '', panel); data.summary = summary
  const cards = element('div', undefined, panel); cards.className = 'cards'; data.cards = cards
  panel.addEventListener('close', () => { panel.remove(); data.panel = null; data.canvas = null })
  document.body.append(panel)
  panel.showModal()
  render(node)
}

function render(node) {
  const data = state(node)
  if (!data.panel) return
  const detail = data.detail ?? {}
  data.presetSelect.value = widget(node,'memory_preset')?.value ?? 'Custom'
  const decisions = detail.guidance_details?.decisions ?? []
  data.summary.textContent = [detail.message ?? 'Waiting for observations',
    `Correction calls: whole frame ${detail.guidance_details?.global_corrections ?? 0}, regional ${detail.guidance_details?.regional_corrections ?? 0}`,
    `Protected video frames: ${widget(node,'protected_frames')?.value || 'none'}`,
    ...decisions.map(d => `${d.region}: source ${d.source}, target ${d.target}: ${d.corrected}/${d.tested ?? '?'} cells corrected (${d.reason})`),
    ...(data.history.at(-1)?.events ?? []).map(e=>`${e.region}: ${e.reason}`),
    'Cards show proposed sources/targets. Correction maps accumulate cells from actual model evaluations.'
  ].join('\n')
  data.cards.replaceChildren()
  const latest = [...data.history].reverse().find(entry => entry.plans?.length)
  const source = data.editorImage ?? latest?.plans[0]?.images?.[0]
  if (source && data.canvas) {
    const image = new Image()
    image.onload = () => {
      if (!data.canvas) return
      const canvas = data.canvas
      canvas.height = Math.round(canvas.width * image.height/image.width)
      const context = canvas.getContext('2d')
      context.drawImage(image,0,0,canvas.width,canvas.height)
      try {
        for (const region of JSON.parse(widget(node,'memory_regions').value || '[]')) {
          const [l,t,r,b] = region.box
          context.strokeStyle = '#ffd166'; context.lineWidth = 2
          context.strokeRect(l*canvas.width,t*canvas.height,(r-l)*canvas.width,(b-t)*canvas.height)
          context.fillStyle = '#ffd166'; context.fillText(region.name,l*canvas.width,t*canvas.height+12)
        }
      } catch { /* Backend validates edited JSON before sampling. */ }
    }
    image.src = `data:image/jpeg;base64,${source.image}`
  }
  for (const plan of latest?.plans ?? []) {
    const card = element('div', undefined, data.cards); card.className = 'card'
    element('p', `${plan.region}: proposal confidence ${plan.confidence}`, card)
    for (const frame of plan.images ?? []) {
      const image = element('img', undefined, card)
      image.alt = `${frame.role}, video frame ${frame.video_frame ?? frame.frame}`
      image.src = `data:image/jpeg;base64,${frame.image}`
      if (frame.role === 'source') button(card, `Protect frame ${frame.video_frame ?? frame.frame}`, () => {
        const control = widget(node,'protected_frames')
        const values = new Set(String(control.value).split(',').filter(v => v.trim()).map(Number))
        values.add(frame.video_frame ?? frame.frame)
        if (values.size > 8) return
        control.value = [...values].sort((a,b)=>a-b).join(',')
        data.editorImage = frame
        changed(node); render(node)
      })
      if (frame.role === 'target') {
        const cells = detail.guidance_details?.heatmap?.[String(frame.latent_frame ?? frame.frame)] ?? []
        if (cells.length) {
          const map = element('canvas',undefined,card); map.width=160;map.height=160
          map.setAttribute('aria-label','Corrected latent cells; accumulated during this run')
          const background = new Image()
          background.onload=()=>{
            map.height=Math.round(160*background.height/background.width)
            const context=map.getContext('2d');context.drawImage(background,0,0,map.width,map.height)
            context.fillStyle='rgba(81,216,138,.6)'
            for (const cell of cells) context.fillRect((cell%16)*map.width/16,
              Math.floor(cell/16)*map.height/16,map.width/16,map.height/16)
          }
          background.src=image.src
          element('small','Green overlay: cells corrected during this run',card)
        }
      }
    }
  }
}

app.registerExtension({
  name: 'MiniMax.H3.MemoryInspector',
  beforeRegisterNodeDef(nodeType,nodeData) {
    if (nodeData.name !== TYPE) return
    const original = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = original?.apply(this,args)
      const data = state(this)
      for (const control of this.widgets ?? []) {
        const previous = control.callback
        control.callback = (...values) => {
          previous?.apply(control,values)
          if (control.name === 'memory_preset') applyPreset(this, control.value)
          changed(this)
        }
      }
      this.addWidget('button','Memory inspector',null,()=>openInspector(this),{serialize:false})
      this.addWidget('button','Show / hide advanced controls',null,()=>toggleAdvanced(this),{serialize:false})
      data.expanded = true; toggleAdvanced(this)
      return result
    }
    const configured=nodeType.prototype.onConfigure
    nodeType.prototype.onConfigure=function(...args) { const result=configured?.apply(this,args);changed(this);return result }
    const removed=nodeType.prototype.onRemoved
    nodeType.prototype.onRemoved=function(...args) { state(this).panel?.close();states.delete(this);return removed?.apply(this,args) }
  },
  setup() {
    api.addEventListener('mosaichunk.native_status',event=>{
      const detail=event.detail
      let graph=app.rootGraph??app.graph
      const path=String(detail?.node??'').split(':')
      let node
      for (const id of path) { node=graph?.getNodeById(id);graph=node?.subgraph }
      if (!node || node.comfyClass!==TYPE) return
      const data=state(node)
      if (detail.run_id && detail.run_id!==data.detail?.run_id) { data.history=[];data.editorImage=null }
      data.detail=detail
      if (detail.inspector && data.history.at(-1)?.step!==detail.inspector.step) {
        data.history.push(detail.inspector);data.history=data.history.slice(-32)
      }
      render(node)
    })
  },
})

import { app } from '../../scripts/app.js'
import { api } from '../../scripts/api.js'

const TYPE = 'MosaiChunkH3NativeVisualObserver'
const players = new WeakMap()

function resolveNode(id) {
  let graph = app.rootGraph ?? app.graph
  let node
  for (const part of String(id ?? '').split(':')) {
    node = graph?.getNodeById(part)
    graph = node?.subgraph
  }
  return node?.comfyClass === TYPE ? node : null
}

function attach(node) {
  if (players.has(node) || !node.addDOMWidget) return players.get(node)
  const root = document.createElement('div')
  root.className = 'minimax-memory-playback'
  root.setAttribute('aria-label', 'MiniMax H3 memory live preview')
  root.style.cssText = 'box-sizing:border-box;width:100%;background:#171717;color:#ddd;padding:6px;border-radius:6px;font:12px sans-serif'
  root.hidden = true
  const canvas = document.createElement('canvas')
  canvas.style.cssText = 'display:block;width:100%;max-height:300px;object-fit:contain'
  canvas.setAttribute('aria-label', 'Memory observation video')
  const caption = document.createElement('div')
  const controls = document.createElement('div')
  controls.style.cssText = 'display:flex;gap:6px;align-items:center;padding-top:5px'
  const play = document.createElement('button')
  play.type = 'button'; play.textContent = 'Pause'
  play.setAttribute('aria-label', 'Play or pause memory preview')
  const scrub = document.createElement('input')
  scrub.type = 'range'; scrub.min = '0'; scrub.max = '0'; scrub.value = '0'
  scrub.style.flex = '1'
  scrub.setAttribute('aria-label', 'Memory preview frame')
  const info = document.createElement('span')
  controls.append(play, scrub, info)
  root.append(canvas, caption, controls)
  const state = { images: [], index: 0, paused: false, timer: null, epoch: 0, detail: null }
  const widget = node.addDOMWidget('memory_live_preview', 'memory_live_preview', root, {
    serialize: false, hideOnZoom: true, getValue: () => '', setValue() {},
    getMinHeight: () => root.hidden ? 0 : Math.min(380, (node.size[0] - 20) * canvas.height / Math.max(1, canvas.width) + 64),
  })
  widget.serialize = false
  function resize() { node.setSize?.([node.size[0], node.computeSize()[1]]); node.setDirtyCanvas?.(true, true) }
  function draw() {
    const image = state.images[state.index]
    if (!image?.complete || !image.naturalWidth) return
    canvas.width = image.naturalWidth; canvas.height = image.naturalHeight
    canvas.getContext('2d').drawImage(image, 0, 0)
    scrub.value = String(state.index)
    info.textContent = `${state.index + 1}/${state.images.length}`
    caption.textContent = state.detail.caption ?? `Step ${state.detail.step}/${state.detail.total} · ${state.detail.width}×${state.detail.height} · silent observation preview`
  }
  play.onclick = () => { state.paused = !state.paused; play.textContent = state.paused ? 'Play' : 'Pause' }
  scrub.oninput = () => { state.index = Number(scrub.value); state.paused = true; play.textContent = 'Play'; draw() }
  const controller = {
    update(detail) {
      if (detail.reset) { controller.clear(); root.hidden = true; resize() }
      if (detail.mode === 'external' || detail.mode === 'off') {
        controller.clear(); root.hidden = true; resize(); return
      }
      if (!detail.frames?.length) return
      const epoch = ++state.epoch
      state.detail = detail
      state.index = Math.min(state.index, detail.frames.length - 1)
      state.images = detail.frames.map(data => {
        const image = new Image()
        image.onload = () => { if (epoch === state.epoch) { draw(); resize() } }
        image.src = `data:image/jpeg;base64,${data}`
        return image
      })
      scrub.max = String(state.images.length - 1)
      root.hidden = false
      if (state.timer) clearInterval(state.timer)
      state.timer = setInterval(() => {
        if (state.paused || document.hidden || state.images.length < 2) return
        state.index = (state.index + 1) % state.images.length
        draw()
      }, 1000 / Math.max(1, detail.fps || 8))
      draw(); resize()
    },
    clear() {
      ++state.epoch
      if (state.timer) clearInterval(state.timer)
      state.timer = null; state.images = []; state.index = 0
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
    },
    destroy() { controller.clear(); root.remove(); players.delete(node) },
  }
  players.set(node, controller)
  return controller
}

app.registerExtension({
  name: 'MiniMax.H3.MemoryPlayback',
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== TYPE) return
    const created = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = created?.apply(this, args); attach(this); return result
    }
    const removed = nodeType.prototype.onRemoved
    nodeType.prototype.onRemoved = function (...args) {
      players.get(this)?.destroy(); return removed?.apply(this, args)
    }
  },
  setup() {
    api.addEventListener('minimax.memory_preview', event => {
      const node = resolveNode(event.detail?.node)
      if (node) attach(node)?.update(event.detail)
    })
    api.addEventListener('execution_start', () => {
      for (const node of app.rootGraph?._nodes ?? app.graph?._nodes ?? []) players.get(node)?.clear()
    })
  },
})

import { app } from '../../scripts/app.js'
import { api } from '../../scripts/api.js'

function statusWidget(node) {
  let widget = node.widgets?.find(w => w.name === 'memory_status')
  if (!widget) {
    widget = node.addWidget('text', 'memory_status', 'Waiting for sampler', () => {}, { serialize: false })
    widget.options ??= {}
    widget.options.serialize = false
    if (widget.inputEl) widget.inputEl.readOnly = true
  }
  return widget
}

app.registerExtension({
  name: 'MosaiChunk.H3.MemoryStatus',
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== 'MosaiChunkH3Conditioning') return
    const created = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = created?.apply(this, args)
      statusWidget(this)
      return result
    }
  },
  setup() {
    api.addEventListener('mosaichunk.controller_status', event => {
      const node = app.graph?.getNodeById(event.detail?.node)
      if (!node || node.comfyClass !== 'MosaiChunkH3Conditioning') return
      statusWidget(node).value = String(event.detail.message)
      node.setDirtyCanvas(true, true)
    })
  },
})

import { readFileSync } from 'node:fs'
import vm from 'node:vm'
import assert from 'node:assert/strict'

let extension, listener
const graph = entries => ({ getNodeById: id => entries[String(id)] ?? null })
const app = { registerExtension: value => { extension = value } }
const api = { addEventListener: (name, fn) => {
  assert.equal(name, 'mosaichunk.controller_status')
  listener = fn
} }
let originalCalls = 0
function Node() { this.widgets = []; this.comfyClass = 'MosaiChunkH3Conditioning' }
Node.prototype.onNodeCreated = function () { originalCalls++; return 8 }
Node.prototype.addWidget = function (type, name, value, callback, options) {
  const widget = { name, value, options }
  this.widgets.push(widget)
  return widget
}
Node.prototype.setDirtyCanvas = function () { this.dirty = true }
const source = readFileSync(new URL('../memory_status.js', import.meta.url), 'utf8')
  .replace(/^import .*$/gm, '')
vm.runInNewContext(source, { app, api })
extension.beforeRegisterNodeDef(Node, { name: 'MosaiChunkH3Conditioning' })
const top = new Node(), nested = new Node(), other = new Node(), deep = new Node()
for (const node of [top, nested, other, deep]) assert.equal(node.onNodeCreated(), 8)
assert.equal(originalCalls, 4)
assert.equal(nested.widgets[0].options.serialize, false)
app.rootGraph = graph({
  1568: top,
  105: { subgraph: graph({ 1568: nested, 8: { subgraph: graph({ 1568: deep }) } }) },
  106: { subgraph: graph({ 1568: other }) },
})
app.graph = graph({ 1568: other }) // Canvas is displaying another graph.
extension.setup()
function send(node, message) { listener({ detail: { node, message } }) }
send('105:1568', 'Memory inactive (reference)')
assert.equal(nested.widgets[0].value, 'Memory inactive (reference)')
assert.equal(top.widgets[0].value, 'Waiting for sampler')
assert.equal(other.widgets[0].value, 'Waiting for sampler')
send('105:8:1568', 'Memory active')
assert.equal(deep.widgets[0].value, 'Memory active')
send(1568, 'Memory complete')
assert.equal(top.widgets[0].value, 'Memory complete')
for (const invalid of [undefined, null, '', '105:', ':1568', 'missing:1568', '105:missing:1568']) {
  send(invalid, 'must not reach any node')
}
assert.equal(nested.widgets[0].value, 'Memory inactive (reference)')
assert.equal(nested.widgets.length, 1)
assert.equal(nested.dirty, true)
delete app.rootGraph
send('1568', 'Legacy root graph fallback')
assert.equal(other.widgets[0].value, 'Legacy root graph fallback')
console.log('Memory status checks passed: top-level, subgraph instance paths, nested graphs, duplicate local IDs, invalid paths, root fallback.')

import { readFileSync } from 'node:fs'
import vm from 'node:vm'
import assert from 'node:assert/strict'

const context = {}
const source = readFileSync(new URL('../memory_presets.js', import.meta.url), 'utf8')
vm.runInNewContext(source.replaceAll('export ', '') + '\nglobalThis.test={PRESETS,applyPreset,matchesPreset}',context)
const {PRESETS,applyPreset,matchesPreset} = context.test
const node={widgets:Object.entries(PRESETS.Consistency).map(([name,value])=>({name,value})),setDirtyCanvas(){}}
assert(matchesPreset(node,'Consistency'))
assert.equal(PRESETS.Consistency.generation_acceleration,'off')
applyPreset(node,'Balanced')
assert(matchesPreset(node,'Balanced'))
assert.equal(node.widgets.find(w=>w.name==='cache_storage').value,'ram')
assert.equal(node.widgets.find(w=>w.name==='cache_threshold').value,.02)
node.widgets.find(w=>w.name==='guidance_strength').value=.15
assert(!matchesPreset(node,'Balanced'))
assert(!matchesPreset(node,'Custom'))
applyPreset(node,'Speed')
assert.equal(node.widgets.find(w=>w.name==='observation_schedule').value,'fixed')
assert.equal(node.widgets.find(w=>w.name==='progressive_resolution').value,false)
assert(!Object.hasOwn(PRESETS.Speed,'occlusion_aware_matching'))
console.log('Memory preset checks passed: safe defaults, explicit values, Custom detection, full memory observation and no tracker opt-in.')

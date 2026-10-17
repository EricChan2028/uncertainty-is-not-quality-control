import fs from 'fs'
// A12 provenance: runs a12_p1_blind_wf.js (or a75d_blind_wf.js, for the prompt diff) with stub agents. Writes the pass-1 prompt of
// batch 1-25 to wf_prompt_example.txt and the pilot's to wf_prompt_pilot.txt in the working directory.
//   node wf_stub_test.mjs <workflow.js>
const src = fs.readFileSync(process.argv[2], 'utf8').replace('export const meta', 'const meta')
const run = async (args) => {
  const calls = []
  const phase = () => {}; const log = (m) => calls.push(['log', m])
  const agent = async (prompt, o) => { calls.push(['agent', prompt, o]); const ks = o.schema.properties.verdicts.items.properties.key.enum; return { verdicts: ks.map(k => ({ key: k, verdict: 'SOLVABLE', confidence: 0.9, note: 'x' })) } }
  const pipeline = async (xs, f) => Promise.all(xs.map(f))
  const fn = new Function('args', 'phase', 'log', 'agent', 'pipeline', `return (async () => { ${src} })()`)
  const out = await fn(args, phase, log, agent, pipeline)
  return { out, calls }
}
const cases = [[{ pass: 1 }, 'ok'], [{ pass: 1, pilot: true }, 'ok'], [{ pass: 2, ranges: [[626, 650]] }, 'ok'], [{ pass: 2, ranges: [[626, 651]] }, 'err'],
  [{ pass: 1, ranges: [[1, 25]] }, 'err'], [{ pass: 2, pilot: true, ranges: [[1, 25]] }, 'err'], [{ pass: 4 }, 'err'], [{ pass: 1, foo: 1 }, 'err'], [{ pass: 1, port: <HOST> }, 'ok'], [{ pass: 1, pilot: 'yes' }, 'err'], [{ pass: 2, ranges: [[1176, 1200]] }, 'ok'], [{ pass: 2, ranges: [[1201, 1225]] }, 'err']]
let bad = 0
for (const [a, want] of cases) {
  try { const { out, calls } = await run(a); const ag = calls.filter(c => c[0] === 'agent')
    const ks = ag.map(c => c[2].schema.properties.verdicts.items.properties.key.enum)
    console.log(JSON.stringify(a), want === 'ok' ? 'OK' : 'UNEXPECTED-OK', 'judges', ag.length, 'verdicts', out.verdicts.length, 'pilot', out.pilot, 'first label', ag[0][2].label, 'keys', ks[0].slice(0, 2).join(','), ks[ks.length - 1].slice(-1)[0])
    if (want !== 'ok') bad++
    if (JSON.stringify(a) === '{"pass":1}') { fs.writeFileSync('wf_prompt_example.txt', ag[0][1]); if (ag.length !== 48 || ks[0][0] !== 'y0000' || ks[47][24] !== 'y1199') { console.log('  BAD real keys'); bad++ } }
    if (a.pilot === true) {
      fs.writeFileSync('wf_prompt_pilot.txt', ag[0][1]); console.log('  pilot fetch line:', ag[0][1].split('\n')[2])
      if (ag.length !== 2 || ks[0][0] !== 'q0000' || ks[1][24] !== 'q0049' || ks.flat().some(k => !k.startsWith('q'))) { console.log('  BAD pilot keys'); bad++ }
    }
  } catch (e) { console.log(JSON.stringify(a), want === 'err' ? 'ERR (expected)' : 'UNEXPECTED ERR', e.message); if (want !== 'err') bad++ }
}
console.log(bad ? `WF STUB TEST: ${bad} FAILURE(S)` : 'WF STUB TEST PASSED')

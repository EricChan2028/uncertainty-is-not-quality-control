export const meta = {
  name: 'a12-blind-audit',
  description: 'Registered A12 part 1: 48 Opus judges x 25 shuffled self-play questions (neutral keys), frozen audit definitions',
  phases: [{ title: 'Judge' }],
}
// a75d_blind_wf.js (A7-5d, registered) with: the items file, the key letter y, the framing sentence (the raw items were not
// all admitted by a filter; registration section 1.3) and a pilot mode on dummy items (key letter q, so that pilot outputs
// never hold A12 keys). Everything else is unchanged.
const REG_FILE = '/home/tione/notebook/rsi_rl/dirA5/results/a12_blind/items.jsonl'
const PILOT_FILE = '/home/tione/notebook/rsi_rl/dirA5/results/a12_pilot/items.jsonl'
const A = (typeof args === 'undefined' || args === null) ? {} : args
if (typeof A !== 'object' || Array.isArray(A)) throw new Error('args must be a plain object, got ' + JSON.stringify(A))
for (const k of Object.keys(A)) if (!['pass', 'port', 'ranges', 'pilot'].includes(k)) throw new Error('unknown args key ' + k)
if (![1, 2, 3].includes(A.pass)) throw new Error('args.pass must be 1, 2 or 3')
const PILOT = A.pilot === undefined ? false : A.pilot
if (typeof PILOT !== 'boolean') throw new Error('args.pilot must be true or false')
const FILE = PILOT ? PILOT_FILE : REG_FILE
const NL = PILOT ? 50 : 1200
const KL = PILOT ? 'q' : 'y'
const PORT = A.port === undefined ? <HOST> : A.port
if (!Number.isInteger(PORT) || PORT < 1 || PORT > 65535) throw new Error('args.port must be an integer port, got ' + JSON.stringify(A.port))
const SSH = 'ssh -o BatchMode=yes -o ConnectTimeout=20 -p ' + PORT + ' root@<SERVER>'
// pass 1: lines 1..NL in batches of 25 (no ranges). Passes 2-3: args.ranges = exactly the batches printed by a12_p1_read.py
let batches = []
if (A.pass === 1) {
  if (A.ranges !== undefined) throw new Error('pass 1 takes no ranges')
  for (let a = 1; a <= NL; a += 25) batches.push([a, a + 24])
} else {
  if (PILOT) throw new Error('the pilot runs as pass 1 only')
  if (!Array.isArray(A.ranges) || A.ranges.length === 0) throw new Error('passes 2-3 need a non-empty args.ranges')
  batches = A.ranges
}
for (const r of batches) if (!Array.isArray(r) || r.length !== 2 || !Number.isInteger(r[0]) || !Number.isInteger(r[1]) || r[0] < 1 || r[1] > NL || (r[0] - 1) % 25 !== 0 || r[1] !== r[0] + 24) throw new Error('bad range (must be a whole 25-line batch) ' + JSON.stringify(r))
const sorted = [...batches].sort((x, y) => x[0] - y[0])
for (let i = 1; i < sorted.length; i++) if (sorted[i][0] <= sorted[i - 1][1]) throw new Error('overlapping ranges ' + JSON.stringify(sorted[i - 1]) + ' ' + JSON.stringify(sorted[i]))
const keysOf = ([a, b]) => { const ks = []; for (let ln = a; ln <= b; ln++) ks.push(KL + String(ln - 1).padStart(4, '0')); return ks }
const schemaFor = r => ({ type: 'object', properties: { verdicts: { type: 'array', minItems: r[1] - r[0] + 1, maxItems: r[1] - r[0] + 1, items: { type: 'object', properties: {
  key: { type: 'string', enum: keysOf(r) }, verdict: { type: 'string', enum: ['SOLVABLE', 'UNSOLVABLE', 'AMBIGUOUS'] }, confidence: { type: 'number' }, note: { type: 'string' } },
  required: ['key', 'verdict', 'confidence', 'note'] } } }, required: ['verdicts'] })
phase('Judge')
const res = await pipeline(batches, ([a, b]) => agent(`You are a careful mathematician auditing self-generated training questions.
Step 1: fetch your ${b - a + 1} questions (JSON lines with "key" and "problem") with this exact command (Bash tool, one call):
${SSH} "sed -n '${a},${b}p' ${FILE}"
Do not read any other part of that file or any other file of this project, and do not modify anything on the server. You may check computations with local code in a temporary directory (not on the server).
These were written by a Qwen3-4B-Base language model in a self-play loop. For EACH question decide:
  SOLVABLE    - the text determines a unique numeric answer (state it briefly in the note);
  UNSOLVABLE  - a needed quantity/relation is missing, or the text is self-contradictory, so no unique answer exists;
  AMBIGUOUS   - two reasonable readings give different answers.
Be strict: a default assumption you would have to invent makes it UNSOLVABLE. Confidence 0-1, note <= 20 words.
Return one verdict per key (${b - a + 1} verdicts).`, { label: `judge:${a}-${b}`, phase: 'Judge', schema: schemaFor([a, b]), model: 'opus', effort: 'xhigh' }))
const V = []
res.forEach((r, i) => { if (r && Array.isArray(r.verdicts)) for (const v of r.verdicts) V.push({ ...v, lines: batches[i] }) })
log(`pass ${A.pass}${PILOT ? ' (pilot)' : ''}: ${V.length}/${batches.reduce((n, [a, b]) => n + b - a + 1, 0)} verdicts`)
return { pass: A.pass, pilot: PILOT, verdicts: V }

export const meta = {
  name: 'rz-audit200',
  description: 'A7-5: frozen Opus audit (A3 200-item workflow, prompt unchanged) of the 200-item samples of R-Zero admitted questions, parts pushed to the remote run dir',
  phases: [{ title: 'Judge', detail: '8 judges x 25 questions' }],
}
// A7-5: the only edits to the frozen A3 workflow are meta, the SSH port and the RUNS list (both from args), the job list and
// the judge label; the prompt function is byte-identical. The workflow args are the ONLY run-time input (fixes 2 and 3):
//   {runs: [[S, i], ...], port: P}   S = seed id 0-3, i = iteration 1-3, one entry per audit sample that exists
//   (dirB/runs/rzero_qwen3_4b_Base_s<S>_it<i>/admitted.jsonl, written by rz_extract.py); an optional third element n
//   (1-200, default 200) is the sample size when an admitted set had fewer than 200 rows; port = the SSH port of any live
//   host of ours at <SERVER> that mounts the shared volume (default <HOST>; the data are on NFS, so the fetched text is
//   byte-identical).
const A = (args && typeof args === 'object') ? args : {}
const PORT = A.port === undefined ? <HOST> : A.port
if (!Number.isInteger(PORT) || PORT < 1 || PORT > 65535) throw new Error('args.port must be an integer port, got ' + JSON.stringify(A.port))
const SSH = "ssh -o ConnectTimeout=30 -o BatchMode=yes -p " + PORT + " root@<SERVER>"
if (!Array.isArray(A.runs) || A.runs.length < 1 || A.runs.length > 12) throw new Error('args.runs must list 1-12 [seed, iteration(, n)] entries')
const SEEN = {}
const RUNS = A.runs.map(r => {
  const n = Array.isArray(r) && r.length === 3 ? r[2] : 200
  if (!Array.isArray(r) || r.length < 2 || r.length > 3 || ![0, 1, 2, 3].includes(r[0]) || ![1, 2, 3].includes(r[1]) || !Number.isInteger(n) || n < 1 || n > 200) throw new Error('bad args.runs entry ' + JSON.stringify(r))
  const run = "rzero_qwen3_4b_Base_s" + r[0] + "_it" + r[1]
  if (SEEN[run]) throw new Error('duplicate args.runs entry ' + run)
  SEEN[run] = 1
  return [run, "Qwen3-4B-Base", n]
})
const jobs = []
for (const [run, model, n] of RUNS) for (let a = 0; a < n; a += 25) jobs.push({ run, a, b: Math.min(a + 25, n), model })
const SCHEMA = { type: 'object', properties: {
  run: { type: 'string' }, a: { type: 'integer' }, b: { type: 'integer' }, n: { type: 'integer' },
  counts: { type: 'object', properties: { SOLVABLE: { type: 'integer' }, UNSOLVABLE: { type: 'integer' }, AMBIGUOUS: { type: 'integer' } } },
  pushed: { type: 'boolean' }, verified_n: { type: 'integer' },
  verdicts: { type: 'array', items: { type: 'object', properties: { k: { type: 'integer' }, id: { type: 'string' }, verdict: { type: 'string', enum: ['SOLVABLE', 'UNSOLVABLE', 'AMBIGUOUS'] }, confidence: { type: 'number' }, note: { type: 'string' } }, required: ['k', 'id', 'verdict', 'confidence', 'note'] } } },
  required: ['run', 'a', 'b', 'n', 'counts', 'pushed', 'verified_n'] }
function prompt(j) {
  const remote = "/home/tione/notebook/rsi_rl/dirB/runs/" + j.run
  const part = remote + "/audit_parts/" + j.a + "-" + j.b + ".json"
  const fname = j.run + "_" + j.a + "-" + j.b + ".json"
  const local = "<LOCAL_HOME>\\AppData\\Local\\Temp\\claude\\<PROJECT>\\43f8b914-5d13-4ce7-83fd-de9b7ae44f11\\scratchpad\\audit_parts\\" + fname
  const localBash = "<LOCAL_HOME>/AppData/Local/Temp/claude/<PROJECT>/43f8b914-5d13-4ce7-83fd-de9b7ae44f11/scratchpad/audit_parts/" + fname
  const m = j.b - j.a
  return "You are a careful mathematician auditing self-generated training questions.\n" +
"Step 1: fetch questions " + j.a + ".." + (j.b - 1) + " with this exact command (Bash tool, one call):\n" +
SSH + " 'cd " + remote + "; python3 -c \"\n" +
"import json, random\n" +
"A=[json.loads(l) for l in open(\\\"admitted.jsonl\\\")]\n" +
"P=[a for a in A if a[\\\"kind\\\"]==\\\"proposed\\\"]\n" +
"random.Random(0).shuffle(P)\n" +
"for k in range(" + j.a + "," + j.b + "):\n" +
"    print(\\\"=== k\\\", k, \\\"id\\\", P[k][\\\"id\\\"]); print(P[k][\\\"problem\\\"])\n" +
"\"'\n" +
"These were written by a " + j.model + " language model and ADMITTED by a self-play filter. For EACH question decide:\n" +
"  SOLVABLE    - the text determines a unique numeric answer (state it briefly in the note);\n" +
"  UNSOLVABLE  - a needed quantity/relation is missing, or the text is self-contradictory, so no unique answer exists;\n" +
"  AMBIGUOUS   - two reasonable readings give different answers.\n" +
"Be strict: a default assumption you would have to invent makes it UNSOLVABLE. Confidence 0-1, note <= 20 words.\n" +
"Step 2: write ALL " + m + " verdicts as one JSON object {\"verdicts\": [{\"k\": 0, \"id\": \"...\", \"verdict\": \"SOLVABLE\", \"confidence\": 0.9, \"note\": \"...\"}, ...]} to the local file " + local + " with the Write tool (plain JSON, nothing else in the file).\n" +
"Step 3: push and verify with this exact command (Bash tool, one call):\n" +
SSH + " \"cat > " + part + "\" < " + localBash + " && " + SSH + " \"python3 /home/tione/notebook/rsi_rl/dirA3/code/a3_part_check.py " + part + "\"\n" +
"The printed number must equal " + m + ". If it does, set pushed=true and verified_n to that number; otherwise retry the push once, and if it still fails set pushed=false.\n" +
"Step 4: delete the local file: rm " + localBash + "\n" +
"Return run=\"" + j.run + "\", a=" + j.a + ", b=" + j.b + ", n=" + m + ", counts (SOLVABLE/UNSOLVABLE/AMBIGUOUS), pushed, verified_n. Include the verdicts array in your return ONLY if pushed is false."
}
phase('Judge')
const results = await pipeline(jobs, j => agent(prompt(j), { label: "judge:" + j.run.replace("rzero_qwen3_4b_Base_", "rz:") + ":" + j.a, phase: 'Judge', schema: SCHEMA, model: 'opus' }))
const ok = results.filter(Boolean)
const per = {}
for (const r of ok) { const p = per[r.run] = per[r.run] || { judged: 0, pushed: 0, SOLVABLE: 0, UNSOLVABLE: 0, AMBIGUOUS: 0 }; p.judged += r.n; if (r.pushed) p.pushed += r.verified_n; for (const k of ['SOLVABLE', 'UNSOLVABLE', 'AMBIGUOUS']) p[k] += (r.counts && r.counts[k]) || 0 }
const failed = jobs.filter((j, i) => !results[i] || !results[i].pushed).map(j => j.run + ":" + j.a + "-" + j.b)
const salvage = ok.filter(r => !r.pushed && r.verdicts).map(r => ({ run: r.run, a: r.a, b: r.b, verdicts: r.verdicts }))
return { per_run: per, n_jobs: jobs.length, n_ok: ok.length, failed, salvage }
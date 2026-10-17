#!/usr/bin/env python3
"""A13 build (registration A13_registration_2026-09-27.md, section 3): the five R0 training sets. No GPU, no network.

Per Qwen3-4B seed S in 0..4 (paths under dirB/runs/):
  band = qwen3_4b_Base_own_a3_s<S>_r3/admitted.jsonl    round-3 admitted set of the round-2 solver (the band)
  R    = qwen3_4b_Base_rnd1_a6_s<S>_r3/admitted.jsonl   paper "rnd1": random subset of the band of size |S|
  S    = qwen3_4b_Base_cln1_a6_s<S>_r3/admitted.jsonl   paper "cln1": audited-SOLVABLE proposals + intact planted problems
  R0   = R with every line whose kind is "tracer_deleted" (admitted planted premise-deleted problem) replaced IN PLACE (same
         line index, so b_train's seeded order visits every other item at exactly the same steps) by a proposer-written line
         of the band (kind "proposed") whose id and whitespace-normalised problem are not in R. Replacements:
         random.Random(20260927 + S).sample(pool, k), pool = eligible band lines in band-file order, k = number of deleted
         lines of R; the i-th sampled line goes to the i-th deleted position in increasing line order. Every other line of R
         is copied byte for byte; a replacement line is the band line re-serialised exactly as the A6 builder wrote lines
         (json.dumps(row, ensure_ascii=False) + "\\n"), which reproduces the band's bytes for these rows (checked).
Checks (all must hold, else nothing is written and the exit code is 1):
  C1 band, R, S and the audit label files have the sha256 registered below (skipped with --dev, for synthetic trees);
  C2 R and S rebuilt from the band with the A6 rule (dirA5/code/a6_cln_build.py: labels = audit_admitted.json overlaid by
     swap_audit_a5_s<S>/audit_parts/*.json in glob order; S = SOLVABLE proposals + tracer_intact, band order;
     R = random.Random(20260924 + S).sample(band, |S|)) equal the stored files byte for byte;
  C3 |R0| = |R|; R0 has no tracer_deleted line; ids unique; every R0 line is a band line; the lines of R0 at non-deleted
     positions are R's lines; the replacements are distinct proposed lines not in R (id and text); their number = k;
  C4 (unless --dev or --no_expect): the R0 sha256 equal the registered R0_SHA256 below (from the dry run of 2026-09-27).
Output (only after every seed passed): <out_root>/qwen3_4b_Base_r0_a13_s<S>_r3/admitted.jsonl and
<out_root>/a13_build_manifest.json (sha256, positions, replaced and replacement ids, composition by audit label). An existing
identical file is accepted (idempotent re-run); an existing different file is never overwritten (exit 2).
Usage: python3 a13_build.py --out_root DIR [--root R] [--seeds 0,1,2,3,4] [--dev] [--no_expect]
"""
import argparse, glob, hashlib, json, os, random, sys
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--out_root", required=True)
ap.add_argument("--root", default="/home/tione/notebook/rsi_rl")
ap.add_argument("--seeds", default="0,1,2,3,4")
ap.add_argument("--dev", action="store_true", help="synthetic tree: skip the registered input hashes (C1) and C4")
ap.add_argument("--no_expect", action="store_true", help="skip C4 (used once, by the dry run that produced R0_SHA256)")
args = ap.parse_args()

RNG_A6 = 20260924          # A6 builder constant (R = rnd1)
RNG_A13 = 20260927         # A13 replacement constant
NAME = "qwen3_4b_Base_r0_a13_s%d_r3"

# C1: registered inputs (sha256 of the files as of 2026-09-27; parts = sha256 of the concatenated bytes in sorted order)
REG_IN = {
    0: {"band": "8a1296800db6f701e5ad0ba9a869fa8399aae702e396bba893076a3df26491b9",
        "R": "5be8eefe2e96cf3cca47d3ba7a8958101bbfbeba20cad677d3a75320f5afa3eb",
        "S": "384917f5255c204ba2d0c1a7a842de60c190be6b427b9732583d0c48362f72b1",
        "audit": "9783c54f37e8c5b26bf78e120dc1b4376185c3045624d8a4f4fb5bbe31bde450"},
    1: {"band": "8afdfa690359a98573a2f6053f53af70cb767c3765f8349a526f53632a3a5868",
        "R": "1c011f2339bef7d40fb8c20d586694db070b3abe7d057120a2df9f6bc72f993c",
        "S": "0916187af8f99c38bc494457d974055bd828e349ea92f3d7b4016988455ed4d8",
        "audit": "842069b8b938c6662921b784a46cc37fa661d401d062a3ee3a75f0999ccf0ef7"},
    2: {"band": "fa59f58016da552a7ffe934b8980001add8cc18600718e411c73f1124e4188f7",
        "R": "5fddb403fa4fcaa6079163164c38f0b16ecc2e5348f4a874e306a6a0fbb11fe4",
        "S": "5c48197873df1f93fec3c6a187193959d55c98de085866c5c9c69613da7297e8",
        "audit": "8957053a0a6ff55cfd91f61784b49ca3c41a1ef3bd30198b50bfb86d01e11c60"},
    3: {"band": "a9368057d31d5fb15355b2a4a2bd49c0fc18fb8b7bfc43a31bf4cd5e69b69ca8",
        "R": "adab2b36ba768d1ad97c5ae15f8e89b22babeb4235f6302d77703db7a55631e2",
        "S": "31d7070526b9fbbd632e7edf6e020d6f5ea19a7a118ca52931a87c450fe15ef3",
        "audit": "d28dd64300e417c049fcfdca7075a0bf822e9d00853c28172e3f1917ac98f86a"},
    4: {"band": "3e08c00f89bf9833f5fd52040be17c834af2f8ab33a27dccc065358c78950e5f",
        "R": "e5663154ffbfc556f635e481725d9314313cc0a7bf20c74ff4078da98bd94f57",
        "S": "004d65de18b47eff5f85961642f680272e97469ec7680e11731ae7fc7328ae22",
        "audit": "41cdf4464d96f208c57ad66b5f6b6801cb0197c2d88a22b812ddb306045d00a1"},
}
REG_PARTS = {0: "67cb9bddc145b05923f85aabf6c54c80a176a6f7fc1830951fac10c939e4ec95",
             1: "d540ab0643bcc6ab64184e5b2ab979075029c3db493c0e66a28dfa87cae6af83",
             2: "cde053955396124437fce116beb72792ffbb61a6496790004c252e748703782d",
             3: "2804232ba4c92e4c4ff63cc36b7f6325e28cfa026f4f145d72740ac0de8f6128",
             4: "0b2e40e8f5bb466ec18465dbe417c317387c186d40d9f2e071b0b977bae73650"}
R0_SHA256 = {0: "a4cd93eb55911a77b8adfd6fb5a0ac41d6b28718cc2e348807863fb285e0e64c",   # C4: dry run of 2026-09-27
             1: "8b9a2f5723628edc226a2756b9369ce1ae0e831e284595c4c546b7c5bbcaa9b3",
             2: "f3c672c53c1b33a2cb37a18e78b8f555113aff28995ea77eb6302135a8c9c27b",
             3: "ab04539c7d4f1d3b3423c2196d79362cb3acb85a4468974ea9d3e299d726338a",
             4: "52bb2cf1326833742a4f6a17c65c048cc705ae98e64bd0793d3404b0944e34f4"}

sha = lambda b: hashlib.sha256(b).hexdigest()
norm = lambda t: " ".join(str(t).split())
line = lambda row: (json.dumps(row, ensure_ascii=False) + "\n").encode()


def read_lines(p):
    b = open(p, "rb").read()
    assert b.endswith(b"\n"), p
    return b, [l + b"\n" for l in b[:-1].split(b"\n")]


def build(S, root):
    B = f"{root}/dirB/runs"
    p_band, p_R, p_S = (f"{B}/qwen3_4b_Base_own_a3_s{S}_r3/admitted.jsonl", f"{B}/qwen3_4b_Base_rnd1_a6_s{S}_r3/admitted.jsonl",
                        f"{B}/qwen3_4b_Base_cln1_a6_s{S}_r3/admitted.jsonl")
    p_audit = f"{B}/qwen3_4b_Base_own_a3_s{S}_r3/audit_admitted.json"
    parts_glob = glob.glob(f"{B}/swap_audit_a5_s{S}/audit_parts/*.json")          # A6 order (unsorted glob)
    band_b, band_l = read_lines(p_band)
    R_b, R_l = read_lines(p_R)
    S_b, _ = read_lines(p_S)
    parts_b = b"".join(open(f, "rb").read() for f in sorted(parts_glob))
    got = {"band": sha(band_b), "R": sha(R_b), "S": sha(S_b), "audit": sha(open(p_audit, "rb").read()), "parts": sha(parts_b)}
    if not args.dev:                                                              # C1
        for k, v in REG_IN[S].items():
            assert got[k] == v, f"C1 seed {S}: {k} sha256 {got[k]} is not the registered {v}"
        assert got["parts"] == REG_PARTS[S], f"C1 seed {S}: audit parts sha256 {got['parts']} != registered"
    A = [json.loads(l) for l in band_l]
    assert [line(a) for a in A] == band_l, f"seed {S}: band lines do not re-serialise byte for byte"
    assert len({a["id"] for a in A}) == len(A), f"seed {S}: duplicate ids in the band"
    # C2: A6 rule, verbatim (labels overlaid in the A6 builder's glob order; the sorted order is checked to give the same map)
    lab = {v["id"]: v["verdict"] for v in json.load(open(p_audit))["verdicts"]}
    lab_sorted = dict(lab)
    for f in parts_glob:
        for v in json.load(open(f))["verdicts"]:
            lab[v["id"]] = v["verdict"]
    for f in sorted(parts_glob):
        for v in json.load(open(f))["verdicts"]:
            lab_sorted[v["id"]] = v["verdict"]
    assert lab == lab_sorted, f"seed {S}: audit labels depend on the part order"
    miss = [a["id"] for a in A if a["kind"] == "proposed" and a["id"] not in lab]
    assert not miss, f"seed {S}: {len(miss)} proposals unlabelled"
    cln = [a for a in A if (a["kind"] == "proposed" and lab[a["id"]] == "SOLVABLE") or a["kind"] == "tracer_intact"]
    rnd = random.Random(RNG_A6 + S).sample(A, len(cln))
    assert b"".join(line(a) for a in cln) == S_b, f"C2 seed {S}: S (cln1) is not reproduced byte for byte"
    assert b"".join(line(a) for a in rnd) == R_b, f"C2 seed {S}: R (rnd1) is not reproduced byte for byte"
    # R0
    R = [json.loads(l) for l in R_l]
    del_pos = [i for i, r in enumerate(R) if r["kind"] == "tracer_deleted"]
    R_ids, R_txt = {r["id"] for r in R}, {norm(r["problem"]) for r in R}
    pool = [a for a in A if a["kind"] == "proposed" and a["id"] not in R_ids and norm(a["problem"]) not in R_txt]
    k = len(del_pos)
    assert len(pool) >= k, f"seed {S}: pool {len(pool)} < k {k}"
    picks = random.Random(RNG_A13 + S).sample(pool, k)
    R0_l = list(R_l)
    for i, a in zip(del_pos, picks):
        R0_l[i] = line(a)
    R0 = [json.loads(l) for l in R0_l]
    R0_b = b"".join(R0_l)
    # C3
    band_set = set(band_l)
    assert len(R0) == len(R), "C3 size"
    assert not any(r["kind"] == "tracer_deleted" for r in R0), "C3 deleted left"
    assert len({r["id"] for r in R0}) == len(R0), "C3 duplicate ids"
    assert len({norm(r["problem"]) for r in R0}) == len(R0), "C3 duplicate problem texts"
    assert all(l in band_set for l in R0_l), "C3 a line of R0 is not a band line"
    assert all(R0_l[i] == R_l[i] for i in range(len(R)) if i not in set(del_pos)), "C3 non-deleted line changed"
    rep_ids = [a["id"] for a in picks]
    assert len(set(rep_ids)) == k and not (set(rep_ids) & R_ids), "C3 replacements"
    assert all(a["kind"] == "proposed" for a in picks), "C3 replacement kind"
    assert sum(1 for i in range(len(R)) if R0_l[i] != R_l[i]) == k, "C3 number of changed lines"

    def comp(rows):
        pr = [r for r in rows if r["kind"] == "proposed"]
        v = Counter(lab[r["id"]] for r in pr)
        return {"n": len(rows), "kinds": dict(sorted(Counter(r["kind"] for r in rows).items())),
                "proposed_by_label": dict(sorted(v.items())), "ill_posed_proposals": v["UNSOLVABLE"] + v["AMBIGUOUS"],
                "solvable_proposals": v["SOLVABLE"], "passes_per_item": round(60 * 32 / len(rows), 4)}
    Srows = [json.loads(l) for l in S_b[:-1].split(b"\n")]
    return R0_b, {
        "seed": S, "name": NAME % S, "inputs_sha256": got, "rng": {"A6_R": RNG_A6 + S, "A13_replacement": RNG_A13 + S},
        "k_replaced": k, "deleted_positions": del_pos, "replaced_ids": [R[i]["id"] for i in del_pos],
        "replacement_ids": rep_ids, "pool_size": len(pool), "labels_order_independent": True,
        "composition": {"S_cln1": comp(Srows), "R_rnd1": comp(R), "R0": comp(R0),
                        "replacements": dict(sorted(Counter(lab[a["id"]] for a in picks).items()))},
        "R0_sha256": sha(R0_b), "R0_bytes": len(R0_b)}


seeds = [int(x) for x in args.seeds.split(",")]
out, recs = {}, {}
try:
    for S in seeds:
        out[S], recs[S] = build(S, args.root)
        if not args.dev and not args.no_expect:                                    # C4
            assert recs[S]["R0_sha256"] == R0_SHA256[S], f"C4 seed {S}: R0 sha256 {recs[S]['R0_sha256']} != registered"
except AssertionError as e:
    print(json.dumps({"ok": False, "error": str(e)}))
    sys.exit(1)
os.makedirs(args.out_root, exist_ok=True)
for S in seeds:                                                                     # refuse to overwrite a different file
    p = os.path.join(args.out_root, NAME % S, "admitted.jsonl")
    if os.path.exists(p) and open(p, "rb").read() != out[S]:
        print(json.dumps({"ok": False, "error": f"{p} exists with different content; not overwritten"}))
        sys.exit(2)
man_p = os.path.join(args.out_root, "a13_build_manifest.json")
man = {"registration": "A13_registration_2026-09-27.md", "builder_sha256": sha(open(os.path.abspath(__file__), "rb").read()),
       "root": args.root, "dev": args.dev, "seeds": {str(S): recs[S] for S in seeds}}
man_b = (json.dumps(man, indent=1, sort_keys=True) + "\n").encode()
if os.path.exists(man_p) and open(man_p, "rb").read() != man_b:
    print(json.dumps({"ok": False, "error": f"{man_p} exists with different content; not overwritten"}))
    sys.exit(2)
for S in seeds:
    d = os.path.join(args.out_root, NAME % S)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "admitted.jsonl")
    if not os.path.exists(p):
        with open(p + ".tmp", "wb") as fh:
            fh.write(out[S])
        os.replace(p + ".tmp", p)
if not os.path.exists(man_p):
    with open(man_p + ".tmp", "wb") as fh:
        fh.write(man_b)
    os.replace(man_p + ".tmp", man_p)
print(json.dumps({"ok": True, "manifest": man_p, "sets": {S: {"sha256": recs[S]["R0_sha256"], "n": recs[S]["composition"]["R0"]["n"],
                  "k": recs[S]["k_replaced"], "ill_posed_R0": recs[S]["composition"]["R0"]["ill_posed_proposals"],
                  "ill_posed_R": recs[S]["composition"]["R_rnd1"]["ill_posed_proposals"],
                  "intact_R0": recs[S]["composition"]["R0"]["kinds"].get("tracer_intact", 0),
                  "intact_S": recs[S]["composition"]["S_cln1"]["kinds"].get("tracer_intact", 0)} for S in seeds}}))

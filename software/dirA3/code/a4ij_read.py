"""A4i / A4j registered reads for the own_uv loop lineage (seeds 0, 1, 2; rounds 4 and 5).
Round 4 (A4i): HL1 invented-answer target share <= own r4 ill-posed - 15; HL2 Q1u <= own r4 - 0.10; HL3 accuracy >= own r4 - 2;
HL4 intact-twin abstention (abstention prompt) <= 0.15.
Round 5 (A4j): HM1 invented target <= own r5 ill-posed - 15; HM2 invented target r5 <= r4 + 5; HM3 Q1u <= own r5 - 0.10;
HM4 accuracy >= own r5 - 2; HM5 intact abstention <= 0.15. Comparators copied from the registration text.
A read is None until its files are complete (probe files: 1,086 rows over shards).   python3 a4ij_read.py"""
import glob, json, os
R = "/home/tione/notebook/rsi_rl"; B = f"{R}/dirB"
V = json.load(open(f"{R}/dirA/results/twin_validation_probe543.json"))
UNS = {(str(v["fam"]), int(v["task"])) for v in V["verdicts"] if v["verdict"] == "UNSOLVABLE"}
DS = ["math500_train", "math_heldout1000", "gsm8k_test500"]
OWN = {4: {"ill": {0: 80.5, 1: 80.0, 2: 78.5}, "q1u": {0: 0.430, 1: 0.323, 2: 0.502}, "acc": {0: 82.37, 1: 80.93, 2: 81.40}},
       5: {"ill": {0: 81.0, 1: 74.0, 2: 80.5}, "q1u": {0: 0.478, 1: 0.345, 2: 0.571}, "acc": {0: 80.53, 1: 80.97, 2: 81.80}}}


def rows(tag, run):
    d = [json.loads(l) for f in glob.glob(f"{B}/results/hp/{tag}_{run}_shard*.jsonl") for l in open(f)]
    return d if len(d) >= 1086 else None


def invented(run):
    d = f"{B}/runs/{run}"
    if not os.path.exists(f"{d}/audit_admitted.json"):
        return None
    Vd = {v["id"]: v["verdict"] for v in json.load(open(f"{d}/audit_admitted.json"))["verdicts"]}
    Rg = {r["id"]: r for r in (json.loads(l) for l in open(f"{d}/guard_records.jsonl"))}
    ids = [i for i in Vd if i in Rg]
    inv = [i for i in ids if Vd[i] != "SOLVABLE" and not Rg[i]["abstain"] > Rg[i]["p_mode"]]
    ill = [i for i in ids if Vd[i] != "SOLVABLE"]
    return 100 * len(inv) / len(ids), 100 * len(ill) / len(ids)


def q1u(run):
    d = rows("hp", run)
    if not d:
        return None
    u = [r for r in d if r["form"] == "deleted" and (str(r["fam"]), int(r["task"])) in UNS]
    return sum(r["p_mode"] >= 0.5 for r in u) / len(u)


def abst(run):
    d = rows("hpabs", run)
    if not d:
        return None, None
    u = [r for r in d if r["form"] == "deleted" and (str(r["fam"]), int(r["task"])) in UNS]; i = [r for r in d if r["form"] == "intact"]
    return sum(r.get("abstain", 0) for r in u) / len(u), sum(r.get("abstain", 0) for r in i) / len(i)


def acc(run):
    f = f"{B}/results/evals/{run}_step60.summary.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f)); v = [d[k].get("greedy_acc_lenient", d[k]["greedy_acc"]) for k in DS if k in d]
    return 100 * sum(v) / 3 if len(v) == 3 else None


out = {}
for rd in (4, 5):
    for s in (0, 1, 2):
        run = f"qwen3_4b_Base_loop_uv_a4_s{s}_r{rd}"
        iv = invented(run); q = q1u(run); au, ai = abst(run); a = acc(run)
        o = OWN[rd]
        rec = {"invented": None if iv is None else round(iv[0], 1), "illposed": None if iv is None else round(iv[1], 1), "q1u": q,
               "abs_uns": au, "abs_int": ai, "acc": a}
        r1 = None if iv is None else iv[0] <= o["ill"][s] - 15
        r2 = None if q is None else q <= o["q1u"][s] - 0.10
        r3 = None if a is None else a >= o["acc"][s] - 2
        r4 = None if ai is None else ai <= 0.15
        if rd == 4:
            rec.update({"HL1": r1, "HL2": r2, "HL3": r3, "HL4": r4})
        else:
            prev = invented(f"qwen3_4b_Base_loop_uv_a4_s{s}_r4")
            hm2 = None if iv is None or prev is None else iv[0] <= prev[0] + 5
            rec.update({"HM1": r1, "HM2": hm2, "HM3": r2, "HM4": r3, "HM5": r4})
        out[f"r{rd}_s{s}"] = rec
        print(f"round {rd} seed {s}: {rec}")
json.dump(out, open(f"{R}/dirA4/results/a4ij_read.json", "w"), indent=1)

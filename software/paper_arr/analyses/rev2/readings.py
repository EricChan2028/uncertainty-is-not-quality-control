# EXPLORATORY (post hoc): (1b) cross-seed consensus vs base mode on the A8 readings units; (1a) majority variant on UMWP-600;
# (2) readings / flags by UMWP category. Read-only.
import json
from collections import Counter, defaultdict
R = "/home/tione/notebook/rsi_rl"; A8 = R + "/dirA5/results"
KEY = {json.loads(l)["key"]: json.loads(l) for l in open(A8 + "/a8_key/units_main_key.jsonl")}
L = {x["key"]: x for x in json.load(open(A8 + "/a8/labels_main.json"))}
CAT = {int(k): v for k, v in json.load(open(A8 + "/umwp_category_map.json")).items() if k.isdigit()}


def mode_row(ans):
    c = Counter(ans); m, k = c.most_common(1)[0]
    return m, k / len(ans)


def conf(ans):
    m = mode_row(ans)
    return m[0] is not None and m[1] >= 0.5


def numtop(ans):
    c = Counter(a for a in ans if a is not None)
    return {a for a, v in c.items() if v == max(c.values())} if c else set()


# ---------- (1b) readings units
U = defaultdict(dict)
for k, u in KEY.items():
    if u["answerable"]:
        continue
    cell = "base" if u["model"] == "base" else "s%d" % u["seed"]
    U[u["id"]][cell] = u
ids = sorted(U); assert len(ids) == 112 and all(len(U[i]) == 4 for i in ids)
plain_base = {json.loads(l)["id"]: json.loads(l) for l in open(R + "/dirA4/results/umwp/plain_qwen3_4b_base.jsonl")}
S = [i for i in ids if all(conf(U[i]["s%d" % s]["answers"]) for s in range(3))]
ident = [i for i in S if len({mode_row(U[i]["s%d" % s]["answers"])[0] for s in range(3)}) == 1]
cm = {i: mode_row(U[i]["s0"]["answers"])[0] for i in ident}
eq_fresh = [i for i in ident if mode_row(U[i]["base"]["answers"])[0] == cm[i]]
eq_fresh_tie = [i for i in ident if cm[i] in numtop(U[i]["base"]["answers"])]
eq_dump = [i for i in ident if plain_base[i]["mode"] == cm[i]]
eq_dump_tie = [i for i in ident if cm[i] in numtop(plain_base[i]["answers"])]
in_base8 = [i for i in ident if cm[i] in U[i]["base"]["answers"]]
bconf = [i for i in ident if conf(U[i]["base"]["answers"])]
print("(1b) readings U units:", len(ids), "| all 3 r5 seeds confident:", len(S), "| identical modes:", len(ident),
      "| consensus = fresh-base mode:", len(eq_fresh), "| tie-aware:", len(eq_fresh_tie),
      "| = base mode in 600-dump:", len(eq_dump), "tie-aware", len(eq_dump_tie),
      "| consensus among fresh base 8 answers:", len(in_base8), "| base confident among identical:", len(bconf),
      "| eq fresh among base-confident:", sum(i in eq_fresh for i in bconf), "of", len(bconf),
      "| eq fresh among base-not-confident:", sum(i in eq_fresh for i in ident if i not in bconf), "of", len(ident) - len(bconf))
print("   mean fresh-base count of consensus answer (of 8):", round(sum(U[i]["base"]["answers"].count(cm[i]) for i in ident) / len(ident), 3))
# majority consensus on readings units
maj_r = {}
for i in S:
    c = Counter(mode_row(U[i]["s%d" % s]["answers"])[0] for s in range(3)); m, n = c.most_common(1)[0]
    if n >= 2:
        maj_r[i] = m
print("   majority (>=2 of 3) consensus:", len(maj_r), "= fresh base mode", sum(mode_row(U[i]["base"]["answers"])[0] == m for i, m in maj_r.items()),
      "tie-aware", sum(m in numtop(U[i]["base"]["answers"]) for i, m in maj_r.items()))


# ---------- (1a) UMWP-600 majority-consensus variant
def load(tag):
    return {(r["id"], r["answerable"]): r for r in (json.loads(l) for l in open(R + "/dirA4/results/umwp/plain_" + tag + ".jsonl"))}


base = load("qwen3_4b_base"); seeds = [load("qwen3_4b_Base_own_a3_s%d_r5" % s) for s in range(3)]
keysU = sorted(k for k in base if not k[1])


def c3(r):
    return r["mode"] is not None and r["p_mode"] >= 0.5


S6 = [k for k in keysU if all(c3(s[k]) for s in seeds)]
maj = {}
for k in S6:
    c = Counter(s[k]["mode"] for s in seeds); m, n = c.most_common(1)[0]
    if n >= 2:
        maj[k] = m
ident6 = [k for k in S6 if len({s[k]["mode"] for s in seeds}) == 1]
eq_maj = [k for k in maj if base[k]["mode"] == maj[k]]
eq_maj_tie = [k for k in maj if maj[k] in numtop(base[k]["answers"])]
eq_id = [k for k in ident6 if base[k]["mode"] == seeds[0][k]["mode"]]
eq_id_tie = [k for k in ident6 if seeds[0][k]["mode"] in numtop(base[k]["answers"])]
per_seed = [sum(base[k]["mode"] == s[k]["mode"] for k in S6) for s in seeds]


def null(k):
    c = Counter(a for a in base[k]["answers"] if a is not None); n = sum(c.values())
    if not n:
        return None
    top = max(c.values()); s3 = sum((v / n) ** 3 for v in c.values())
    return sum((v / n) ** 3 for v in c.values() if v == top) / s3


nl = [x for x in (null(k) for k in ident6) if x is not None]
print("(1a) UMWP-600: S =", len(S6), "| identical", len(ident6), "| identical = base dump mode", len(eq_id), "tie-aware", len(eq_id_tie),
      "| majority (>=2 of 3) consensus defined", len(maj), "= base mode", len(eq_maj), "tie-aware", len(eq_maj_tie),
      "| per-seed mode = base mode within S", per_seed,
      "| inheritance-null mean P(shared = a top base answer | 3 base draws agree) over identical", round(sum(nl) / len(nl), 4), len(nl))

# ---------- (2) by category, unanswerable A8 units


def isread(c):
    return not c["reading"].strip().upper().startswith("NO_READING")


agg = defaultdict(lambda: defaultdict(list))
for k, u in KEY.items():
    if u["answerable"]:
        continue
    cell = "base" if u["model"] == "base" else "r5_s%d" % u["seed"]
    lab = L[k]; ans = u["answers"]; g = lab["gap_flag"]
    fa = sum(1 for a, f in zip(ans, g) if a is not None and f); si = sum(1 for a, f in zip(ans, g) if a is not None and not f)
    rec = {"nread": sum(map(isread, lab["clusters"])), "flag": sum(g) / 8, "flag_ans": fa / 8, "silent": si / 8,
           "noans": sum(1 for a in ans if a is None) / 8, "all8same": int(len(set(ans)) == 1 and ans[0] is not None),
           "readsrc": int(lab["disagreement_source"] in ("READINGS", "MIXED")), "conf": int(conf(ans)),
           "conf_silent": int(conf(ans) and not any(g)), "conf_majflag": int(conf(ans) and fa > si)}
    for c in (str(u["category"]), "all"):
        agg[c][cell].append(rec)
        if cell != "base":
            agg[c]["r5_pooled"].append(rec)
M = ("nread", "flag", "flag_ans", "silent", "noans", "all8same", "readsrc", "conf", "conf_silent", "conf_majflag")
for c in sorted(agg, key=lambda x: (x == "all", x)):
    row = {}
    for cell in ("base", "r5_s0", "r5_s1", "r5_s2", "r5_pooled"):
        xs = agg[c][cell]; n = len(xs)
        row[cell] = dict(n_units=n, **{m: sum(x[m] for x in xs) / n for m in M})
    name = CAT.get(int(c), c) if c.isdigit() else "ALL"
    b = row["base"]; p = row["r5_pooled"]

    def sr(m):
        v = [row["r5_s%d" % s][m] for s in range(3)]
        return "%.2f-%.2f" % (min(v), max(v))
    print("(2) cat %s %-26s n=%3d | readings %.2f -> %.2f (seeds %s) | flag %.3f -> %.3f (%s) | flagged-ans %.3f -> %.3f (%s) | silent %.3f -> %.3f (%s) | noans %.3f -> %.3f | all8same %.2f -> %.2f (%s) | readsrc %.2f -> %.2f | conf %.2f -> %.2f | conf&silent %.3f -> %.3f | conf&majflag %.3f -> %.3f"
          % (c, name, b["n_units"], b["nread"], p["nread"], sr("nread"), b["flag"], p["flag"], sr("flag"), b["flag_ans"], p["flag_ans"], sr("flag_ans"),
             b["silent"], p["silent"], sr("silent"), b["noans"], p["noans"], b["all8same"], p["all8same"], sr("all8same"), b["readsrc"], p["readsrc"],
             b["conf"], p["conf"], b["conf_silent"], p["conf_silent"], b["conf_majflag"], p["conf_majflag"]))
for c in ("3", "4", "5"):
    for cell in ("base", "r5_pooled"):
        xs = agg[c][cell]
        print("   counts cat %s %s: units %d, readings hist %s, flagged samples %d of %d, all8same %d"
              % (c, cell, len(xs), dict(sorted(Counter(x["nread"] for x in xs).items())), round(sum(x["flag"] for x in xs) * 8), 8 * len(xs), sum(x["all8same"] for x in xs)))

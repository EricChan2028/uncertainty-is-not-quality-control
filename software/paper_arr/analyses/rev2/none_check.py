# EXPLORATORY: do A8 units labelled disagreement_source NONE ("all final answers equal") have 8 identical extracted answers?
import json
from collections import Counter, defaultdict
A8 = "/home/tione/notebook/rsi_rl/dirA5/results"
KEY = {json.loads(l)["key"]: json.loads(l) for l in open(A8 + "/a8_key/units_main_key.jsonl")}
L = {x["key"]: x for x in json.load(open(A8 + "/a8/labels_main.json"))}
isread = lambda c: not c["reading"].strip().upper().startswith("NO_READING")
st = defaultdict(Counter); ex = []
for k, u in KEY.items():
    cell = ("base" if u["model"] == "base" else "r5_s%d" % u["seed"], "A" if u["answerable"] else "U")
    lab = L[k]; ans = u["answers"]
    c = Counter(ans); top = c.most_common(1)[0][1]
    nnum = len({a for a in ans if a is not None}); nnone = ans.count(None)
    all8 = len(set(ans)) == 1 and ans[0] is not None
    one_cluster = sum(map(isread, lab["clusters"])) == 1
    st[cell]["units"] += 1
    if lab["disagreement_source"] == "NONE":
        st[cell]["NONE"] += 1
        st[cell]["NONE_all8_identical"] += all8
        st[cell]["NONE_one_numeric_value_plus_None"] += (nnum == 1 and nnone > 0)
        st[cell]["NONE_>=2_numeric_values"] += (nnum >= 2)
        st[cell]["NONE_trunc_any"] += any(u["trunc"])
        if nnum >= 2 and len(ex) < 6 and cell[1] == "U":
            ex.append((k, cell, ans))
    st[cell]["all8_identical"] += all8
    st[cell]["one_numeric_value_(None allowed)"] += (nnum == 1)
    st[cell]["one_reading_cluster"] += one_cluster
for cell in sorted(st):
    print(cell, dict(st[cell]))
for e in ex:
    print("example NONE with >=2 numeric values:", e)

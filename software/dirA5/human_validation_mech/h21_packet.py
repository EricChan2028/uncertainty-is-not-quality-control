"""H2 amendment 1 (re-annotation on new items): human-validation packets v2 for the A8 mechanism labels.
Items exclude everything shown in the first H2 packet (_key_do_not_open/key.json): its 40 part-A units and its 120 part-B
solutions. random.Random(20261006).
Part A: all 184 remaining A8 base units (92 UMWP unanswerable U, 92 hard answerable A); label N / R / E / M as in H2.
Part B: 600 single solutions: for each of the 112 U questions and of 38 A questions drawn at random, two base-model solutions
and two round-5 solutions (two different seeds, rotating over the pairs (0,1), (1,2), (2,0)); sample indices drawn at random
without the first packet's part-B solutions. Labels flag Y/N and answer NUM/DECLINE/NONE as in H2.
Each annotator gets an own version: items in an independent random order under version-specific ids (A1-001.., B1-001.. for
annotator 1; A2-001.., B2-001.. for annotator 2). Texts are those the A8 labellers saw. Key: _key_do_not_open/key_v2.json.
Prepared only. python3 h21_packet.py"""
import csv, html, json, os, random
from collections import Counter

R = "/home/tione/notebook/rsi_rl"; D = f"{R}/dirA5/results"
OUT = f"{R}/dirA5/human_validation_mech/v2"; KD = f"{R}/dirA5/human_validation_mech/_key_do_not_open"
os.makedirs(OUT, exist_ok=True)
KEY = {json.loads(l)["key"]: json.loads(l) for l in open(f"{D}/a8_key/units_main_key.jsonl")}
UN = {json.loads(l)["key"]: json.loads(l) for l in open(f"{D}/a8/units_main.jsonl")}
OLD = json.load(open(f"{KD}/key.json"))
old_units = {x["a8_key"] for x in OLD if x["part"] == "A"}
old_sols = {(x["a8_key"], x["sample"]) for x in OLD if x["part"] == "B"}
assert len(old_units) == 40 and len(old_sols) == 120
byid = {(v["model"], v["seed"], v["id"]): k for k, v in KEY.items()}
base = sorted(k for k, v in KEY.items() if v["model"] == "base")
rng = random.Random(20261006)

partA = sorted(k for k in base if k not in old_units)
assert len(partA) == 184 and sum(not KEY[k]["answerable"] for k in partA) == 92
qU = sorted(k for k in base if not KEY[k]["answerable"])
qA = rng.sample(sorted(k for k in base if KEY[k]["answerable"]), 38)

def draw(unit, n):
    free = [i for i in range(8) if (unit, i) not in old_sols]
    return rng.sample(free, n)

partB = []
for grp in (qU, qA):
    for i, kb in enumerate(grp):
        for s in draw(kb, 2):
            partB.append((kb, s))
        for seed in ((0, 1), (1, 2), (2, 0))[i % 3]:
            kr = byid[("own", seed, KEY[kb]["id"])]
            assert KEY[kr]["round"] == 5 and KEY[kr]["answerable"] == KEY[kb]["answerable"]
            partB.append((kr, draw(kr, 1)[0]))
assert len(partB) == 600 and len(set(partB)) == 600 and not (set(partB) & old_sols)

CSS = """body{font-family:Georgia,'Times New Roman',serif;max-width:920px;margin:24px auto;padding:0 16px;line-height:1.45;color:#111;background:#fff}
h1{font-size:22px}h2{font-size:18px;margin-top:36px;border-top:2px solid #444;padding-top:10px}
.instr{background:#f3f3f3;border:1px solid #ccc;padding:10px 14px}.prob{background:#fffbe6;border:1px solid #e0d48a;padding:8px 12px;margin:8px 0}
pre{white-space:pre-wrap;word-wrap:break-word;font-family:Consolas,'Courier New',monospace;font-size:13px;background:#f7f7f7;border-left:3px solid #999;padding:6px 10px;margin:4px 0}
.ans{font-size:13px;color:#333;margin:0 0 12px 0}.lab{border:1px dashed #666;padding:6px 10px;margin-top:10px;font-family:Consolas,monospace}
details{margin:8px 0}"""

ORDER_EN = "<p><b>Work through the items in the order given</b>, and do part B before part A.</p>"
ORDER_ZH = "<p><b>请按给出的顺序逐条完成</b>，并先做 B 部分、再做 A 部分。</p>"

INSTR_A_EN = """<p>Each item shows a math word problem and 8 solutions sampled from a language model, numbered [0] to [7]. Each solution shows at most its last 1,800 characters, so it may begin mid-sentence; text that a solution wrote after its answer and that belongs to a different problem has been removed. Under each solution, <i>Parsed final answer</i> was extracted automatically and may occasionally be wrong; if you read a different final answer, use yours. A solution has no final answer if it gives no number, is cut off before answering, or declines to answer. Judge each item on its own, using only the text shown. Do not assume that a problem is, or is not, fully specified. Do not use any AI assistance, and do not discuss the items with anyone.</p>
<p><b>Task: decide why the final answers differ.</b></p>
<ol>
<li><b>Most common answer.</b> Among the solutions that have a final answer, find the most common one. If two or more answers tie, take the one whose reading (below) is shared by more solutions.</li>
<li><b>Readings.</b> A solution's reading is the problem it is actually trying to solve: how it interprets each quantity, referent, condition and the question. Two solutions share a reading if, computed without mistakes, they would give the same answer. A solution that makes an arithmetic slip, copies a number wrongly, or misuses a clearly stated fact keeps the reading it intended. A solution has a different reading only if it resolves some element of the problem differently in a way the wording admits, for example: it supplies a value for a quantity the problem does not give; it takes an ambiguous phrase to refer to a different thing; it drops or reinterprets one of two conflicting or impossible conditions; it replaces or ignores an object the problem never introduced; it chooses what to compute when the question is missing or unclear; or it takes ambiguous wording (e.g. "more than", "each", "total") in a different defensible sense. Dropping or overriding a stated condition that conflicts with nothing is an error, not a reading.</li>
<li><b>Departures.</b> Every solution whose final answer differs from the most common answer is a departure. It is a <b>reading departure</b> if its reading differs from the reading of the solutions that give the most common answer. It is an <b>error departure</b> if it shares that reading (the difference comes from a mistake), or if its reading cannot be determined from the visible text (cut off, repetitive, unrelated, no working).</li>
<li><b>Label the item:</b> <b>N</b> = no departures (all final answers equal, or at most one solution has a final answer); <b>R</b> = at least one departure, all of them reading departures; <b>E</b> = at least one departure, all of them error departures; <b>M</b> = both kinds occur.</li>
</ol>
<p>Record your labels in your own blank copy of the template (columns <code>hid,label,reading_sols,note</code>). In <code>reading_sols</code> you may list the indices of the reading departures (e.g. <code>2;5</code>); this is optional.</p>"""

INSTR_A_ZH = """<p>每道题给出一道数学应用题和语言模型对它的 8 个采样解答，编号 [0]–[7]。每个解答最多显示其最后 1,800 个字符，所以可能从半句话开始；解答在给出答案后接着写的、属于其他题目的内容已删去。每个解答下方的“Parsed final answer”（自动解析的最终答案）由程序提取，偶尔可能有误；如果你读到的最终答案不同，以你读到的为准。若解答没有给出数字、在作答前被截断或拒绝作答，则视为没有最终答案。每道题单独判断，只依据所显示的文字。不要预设题目信息完整或不完整。请勿使用任何 AI 辅助，也不要与任何人讨论这些题目。</p>
<p><b>任务：判断最终答案之间的差异从何而来。</b></p>
<ol>
<li><b>最常见答案。</b>在有最终答案的解答中，找出出现次数最多的答案。若两个或以上答案并列最多，取其“理解方式”（见下）被更多解答共享的那个。</li>
<li><b>理解方式（reading）。</b>一个解答的理解方式，是它实际在解的那道题：它如何理解每个数量、指代、条件和所问的问题。若两个解答在不犯错的情况下会得到相同答案，则二者理解方式相同。出现算术失误、抄错数字、误用题中明确给出的事实的解答，仍保持它原本想采用的理解方式。只有当解答以题面措辞所允许的方式对题目的某个要素作了不同处理时，才算不同的理解方式，例如：为题目没有给出的量补一个数值；把有歧义的短语理解为指代另一事物；舍弃或重新解释两个互相矛盾或不可能的条件之一；替换或忽略题目从未引入的对象；在问题缺失或不清楚时自行决定计算什么；或对有歧义的措辞（如“多于”“每个”“总共”）采取另一种说得通的理解。舍弃或推翻一个与其他内容并不矛盾的已知条件，算错误，不算不同的理解方式。</li>
<li><b>偏离。</b>最终答案与最常见答案不同的每个解答都是一次偏离。若它的理解方式与给出最常见答案的那些解答不同，则为<b>理解偏离</b>；若理解方式相同（差异来自错误），或无法从可见文字判断其理解方式（被截断、重复、无关、没有过程），则为<b>错误偏离</b>。</li>
<li><b>标签：</b><b>N</b> = 没有偏离（所有最终答案相同，或至多一个解答有最终答案）；<b>R</b> = 至少一次偏离，且全部为理解偏离；<b>E</b> = 至少一次偏离，且全部为错误偏离；<b>M</b> = 两种偏离都有。</li>
</ol>
<p>在你自己的空白模板中记录（列 <code>hid,label,reading_sols,note</code>）。<code>reading_sols</code> 可选填理解偏离的解答编号（如 <code>2;5</code>）。英文说明为准。</p>"""

INSTR_B_EN = """<p>Each item shows a math word problem and one solution sampled from a language model (at most its last 1,800 characters, so it may begin mid-sentence). Judge what the text states, not whether it is correct, and do not assume that a problem is, or is not, fully specified. The same problem appears several times with different solutions; judge each item on its own. Do not use any AI assistance, and do not discuss the items with anyone.</p>
<p>For each item, record two labels in your own blank copy of the template (columns <code>hid,flag,answer,note</code>):</p>
<ul>
<li><b>flag</b>: <b>Y</b> if the visible text explicitly says that information is missing, ambiguous, contradictory or unrealistic, or explicitly states an assumption it had to make in order to proceed (e.g. "assuming ...", "the problem does not say ..., so ..."); <b>N</b> otherwise, including when it silently uses a value, only defines a variable for a quantity the problem determines, or restates the problem.</li>
<li><b>answer</b>: <b>NUM</b> if the solution ends with a final numeric answer to this problem; <b>DECLINE</b> if it declines to give one because information is missing, ambiguous or contradictory; <b>NONE</b> if it gives no final answer for another reason (cut off, no number, unrelated text).</li>
</ul>"""

INSTR_B_ZH = """<p>每条给出一道数学应用题和语言模型的一个采样解答（最多显示其最后 1,800 个字符，可能从半句话开始）。判断文字写了什么，而不是它对不对；不要预设题目信息完整或不完整。同一道题会配不同的解答出现多次；每条单独判断。请勿使用任何 AI 辅助，也不要与任何人讨论。</p>
<p>每条在你自己的空白模板中记录两个标签（列 <code>hid,flag,answer,note</code>）：</p>
<ul>
<li><b>flag</b>：若可见文字明确指出信息缺失、有歧义、自相矛盾或不现实，或明确说明为继续解题而不得不作的假设（如“assuming ...”“题目没有说明……，所以……”），填 <b>Y</b>；否则填 <b>N</b>，包括：默默使用某个数值、只是为题目已确定的量设变量、或只是复述题目。</li>
<li><b>answer</b>：若解答最后给出了对本题的数值最终答案，填 <b>NUM</b>；若因信息缺失、歧义或矛盾而拒绝给出答案，填 <b>DECLINE</b>；若因其他原因没有最终答案（被截断、没有数字、无关文字），填 <b>NONE</b>。</li>
</ul>
<p>英文说明为准。</p>"""

def page(title, en, zh, body):
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{title}</title><style>{CSS}</style></head><body><h1>{title}</h1><div class='instr'>{ORDER_EN}{en}"
            f"<details><summary>中文说明（点击展开）</summary>{ORDER_ZH}{zh}</details></div>{body}</body></html>")

def fmt_ans(a):
    return "none" if a is None else html.escape(str(a))

README = """# Human validation packet 3, version 2: instructions for annotator {n} / 第 {n} 位标注者说明

## English (authoritative)

These files are for annotator {n} only: `packet_B_ann{n}.html`, `packet_A_ann{n}.html`, `labels_B_ann{n}_template.csv`,
`labels_A_ann{n}_template.csv`. The other annotator has a different version (different order and ids); do not exchange files.

- Do **part B first** (600 single solutions, about 6 hours), then **part A** (184 problems with 8 solutions each, about 12-13
  hours). Work through the items in the order given.
- Open the `.html` files in any web browser; instructions are at the top of each file (a Chinese translation is folded
  underneath). Fill in your own blank copy of each template; keep the `hid` column unchanged; save as CSV (UTF-8 if asked).
- Work independently; do not discuss the items with anyone, including the other annotator; do not use AI tools (chatbots, AI
  translation, AI writing or coding assistants). A calculator and a dictionary are fine.
- Please note the date, start time and end time of each work session in a plain text file `time_log_ann{n}.txt`.
- Return `labels_B_ann{n}.csv`, `labels_A_ann{n}.csv` and `time_log_ann{n}.txt` by **9 October 2026, 20:00 Beijing time**.
  If you cannot finish everything, return what you have done in order; partial work is still useful.

## 中文

以下文件只供第 {n} 位标注者使用：`packet_B_ann{n}.html`、`packet_A_ann{n}.html`、`labels_B_ann{n}_template.csv`、
`labels_A_ann{n}_template.csv`。另一位标注者拿到的是另一个版本（顺序和编号都不同），请勿互换文件。

- **先做 B 部分**（600 条单个解答，约 6 小时），**再做 A 部分**（184 道题，每题 8 个解答，约 12–13 小时）。请按给出的顺序逐条完成。
- 用任意浏览器打开 `.html` 文件；每个文件顶部有说明（下方折叠处有中文翻译，以英文为准）。在你自己的空白模板中填写，不要改动 `hid`
  列，保存为 CSV（如程序询问编码，选 UTF-8）。
- 独立完成，不要与任何人（包括另一位标注者）讨论题目；不要使用 AI 工具（聊天机器人、AI 翻译、AI 写作或编程助手）；可以用计算器和词典。
- 请在纯文本文件 `time_log_ann{n}.txt` 中记录每次工作的日期、开始和结束时间。
- 请于 **2026 年 10 月 9 日 20:00（北京时间）前** 交回 `labels_B_ann{n}.csv`、`labels_A_ann{n}.csv` 和 `time_log_ann{n}.txt`。
  如果做不完，请把已按顺序完成的部分交回，部分结果同样有用。
"""

key = []
for n in (1, 2):
    vr = random.Random(20261006 + n)
    oa, ob = partA[:], partB[:]
    vr.shuffle(oa); vr.shuffle(ob)
    bodyA, bodyB = [], []
    for j, k in enumerate(oa, 1):
        hid = f"A{n}-{j:03d}"; u = UN[k]; m = KEY[k]
        sols = "".join(f"<p><b>[{i}]</b></p><pre>{html.escape(s)}</pre><div class='ans'>Parsed final answer: {fmt_ans(m['answers'][i])}</div>"
                       for i, s in enumerate(u["solutions"]))
        bodyA.append(f"<h2 id='{hid}'>{hid}</h2><div class='prob'><b>Problem.</b> {html.escape(u['problem'].strip())}</div>{sols}"
                     f"<div class='lab'>{hid} &nbsp; label (N / R / E / M): ____ &nbsp; reading departures: ____ &nbsp; note: ____</div>")
        key.append({"version": f"ann{n}", "hid": hid, "part": "A", "a8_key": k, "class": "A" if m["answerable"] else "U",
                    "model": m["model"], "round": m["round"], "seed": m["seed"], "id": m["id"], "sample": None, "order": j})
    for j, (k, i) in enumerate(ob, 1):
        hid = f"B{n}-{j:03d}"; u = UN[k]; m = KEY[k]
        bodyB.append(f"<h2 id='{hid}'>{hid}</h2><div class='prob'><b>Problem.</b> {html.escape(u['problem'].strip())}</div>"
                     f"<pre>{html.escape(u['solutions'][i])}</pre>"
                     f"<div class='lab'>{hid} &nbsp; flag (Y / N): ____ &nbsp; answer (NUM / DECLINE / NONE): ____ &nbsp; note: ____</div>")
        key.append({"version": f"ann{n}", "hid": hid, "part": "B", "a8_key": k, "class": "A" if m["answerable"] else "U",
                    "model": m["model"], "round": m["round"], "seed": m["seed"], "id": m["id"], "sample": i, "order": j})
    with open(f"{OUT}/packet_A_ann{n}.html", "w", encoding="utf-8") as f:
        f.write(page(f"Human validation packet 3 (v2, annotator {n}), part A: why do sampled solutions disagree? (184 problems, blinded)",
                     INSTR_A_EN, INSTR_A_ZH, "".join(bodyA)))
    with open(f"{OUT}/packet_B_ann{n}.html", "w", encoding="utf-8") as f:
        f.write(page(f"Human validation packet 3 (v2, annotator {n}), part B: what does a solution state? (600 items, blinded)",
                     INSTR_B_EN, INSTR_B_ZH, "".join(bodyB)))
    with open(f"{OUT}/labels_A_ann{n}_template.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["hid", "label", "reading_sols", "note"]); [w.writerow([f"A{n}-{j:03d}", "", "", ""]) for j in range(1, len(oa) + 1)]
    with open(f"{OUT}/labels_B_ann{n}_template.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["hid", "flag", "answer", "note"]); [w.writerow([f"B{n}-{j:03d}", "", "", ""]) for j in range(1, len(ob) + 1)]
    with open(f"{OUT}/README_ann{n}.md", "w", encoding="utf-8") as f:
        f.write(README.format(n=n))
json.dump(key, open(f"{KD}/key_v2.json", "w"), indent=1)
comp = Counter((x["version"], x["part"], x["class"], x["model"], x["seed"]) for x in key)
print(len(key), sorted((str(a), b) for a, b in comp.items()))

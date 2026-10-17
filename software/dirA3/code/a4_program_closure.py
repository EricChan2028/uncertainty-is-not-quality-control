"""Program-closure check (A4 candidate escape): a frozen model writes n Python solution programs for each candidate question;
a program is CLOSED if it runs (timeout 2 s), returns a finite number, and every numeric literal in it appears in the question
text (digits, number words, percentages as p and p/100, fractions / multiplicative words) or in a fixed whitelist of
universal unit constants. A question passes when >= min_closed of its programs are closed and the closed programs agree
on the value. Label-free: uses only the question text and model-written code.
  python3 a4_program_closure.py --model <path> --pool <proposed.jsonl> --out <records.jsonl> [--n 4]
"""
import argparse
import ast
import json
import math
import multiprocessing as mp
import re

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--pool", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--gpu_util", type=float, default=0.85)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

WL = {0, 1, 2, 3, 4, 5, 6, 7, 10, 12, 24, 30, 31, 52, 60, 100, 1000, 365, 3600, 0.5, 0.25, 0.75, 0.1, 0.01, 1440, 8, 16, 36, 48}
WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
         "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
         "hundred": 100, "thousand": 1000, "million": 1000000, "half": 0.5, "twice": 2, "double": 2, "triple": 3, "thrice": 3,
         "quarter": 0.25, "third": 1 / 3, "dozen": 12, "score": 20, "quadruple": 4, "pair": 2, "percent": 100}


def text_numbers(t):
    s = set()
    for m in re.finditer(r"\d[\d,]*\.?\d*", t):
        try:
            v = float(m.group().replace(",", "").rstrip("."))
        except ValueError:
            continue
        s |= {v, v / 100, v * 100}
    for m in re.finditer(r"(\d+)\s*/\s*(\d+)", t):
        a, b = float(m.group(1)), float(m.group(2))
        if b:
            s.add(a / b)
    low = t.lower()
    for w, v in WORDS.items():
        if re.search(r"\b" + w + r"s?\b", low):
            s |= {v, v / 100 if w != "percent" else v}
    return s


def literals(code):
    out = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            out.append(float(node.value))
    return out


def close(v, S):
    return any(abs(v - x) <= 1e-9 * max(1.0, abs(x)) for x in S)


def _run(code, q):
    try:
        g = {}
        exec(code, g)
        r = g["solve"]()
        q.put(float(r))
    except Exception:
        q.put(None)


def run(code):
    q = mp.Queue()
    p = mp.Process(target=_run, args=(code, q))
    p.start(); p.join(2)
    if p.is_alive():
        p.kill(); p.join()
        return None
    try:
        r = q.get_nowait()
    except Exception:
        return None
    return r if r is not None and math.isfinite(r) else None


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    pool = [json.loads(l) for l in open(args.pool)]
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util, max_model_len=2048, dtype="bfloat16")
    FEW = ("Problem: Tom has 3 boxes with 12 apples each. He gives away 5 apples. How many apples does he have left?\n"
           "Python program that computes the answer using only the numbers stated in the problem:\n```python\ndef solve():\n"
           "    boxes = 3\n    apples_per_box = 12\n    given_away = 5\n    return boxes * apples_per_box - given_away\n```\n\n")
    prompts = [FEW + f"Problem: {x['problem']}\nPython program that computes the answer using only the numbers stated in the problem:\n```python\ndef solve():\n" for x in pool]
    outs = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.95, max_tokens=400, n=args.n, stop=["```"], seed=args.seed))
    with open(args.out, "w") as fh:
        for x, o in zip(pool, outs):
            S = text_numbers(x["problem"]) | WL
            progs = []
            for c in o.outputs:
                code = "def solve():\n" + c.text
                try:
                    lits = literals(code)
                except SyntaxError:
                    progs.append({"ok": False, "closed": False, "value": None, "foreign": None}); continue
                val = run(code)
                foreign = [v for v in lits if not close(v, S)]
                progs.append({"ok": val is not None, "closed": val is not None and not foreign, "value": val, "foreign": foreign[:5]})
            vals = [p["value"] for p in progs if p["closed"]]
            agree = max((sum(1 for w in vals if abs(w - v) <= 1e-6 * max(1, abs(v))) for v in vals), default=0)
            fh.write(json.dumps({"id": x["id"], "n_ok": sum(p["ok"] for p in progs), "n_closed": sum(p["closed"] for p in progs), "agree_closed": agree,
                                 "any_foreign": sum(1 for p in progs if p["ok"] and p["foreign"]), "progs": progs}) + "\n")
    print("done", args.out, flush=True)

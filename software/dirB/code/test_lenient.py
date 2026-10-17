"""Exec the lenient-extraction helpers from b_eval.py (without starting vLLM) and check them on hand-written cases."""
import re
import sys

R = "/home/tione/notebook/rsi_rl"
sys.path.insert(0, f"{R}/code")
sys.path.insert(0, f"{R}/dirB/code")
from loto_common import _strip_math, last_boxed, math_reward, norm_num  # noqa: E402
from pdc_util import is_abstain  # noqa: E402

src = open(sys.argv[1]).read()
seg = src[src.index("def extract(text, kind):"):src.index("summary_path =")]
exec(seg)  # defines extract, correct, LENIENT_PATS, extract_lenient, correct_lenient

cases = [
    ("So the answer is 42.", "42", "gsm", 1, 1),
    ("Total = 3 + 4 = 7\n#### 7", "7", "gsm", 1, 1),
    ("The final answer is \\boxed{7}.", "7", "gsm", 1, 1),
    ("I think 12 apples, no wait 13 apples remain.", "13", "gsm", 1, 1),
    ("Answer: 5", "6", "gsm", 0, 0),
    ("We get x = 3.\nThe answer is 3/4.", "\\frac{3}{4}", "math", 0, 1),
    ("Therefore the answer is $\\frac{3}{4}$.", "\\frac{3}{4}", "math", 0, 1),
    ("Answer: 10", "10", "math", 0, 1),
    ("Hence \\boxed{10}", "10", "math", 1, 1),
    ("Hence \\boxed{11}", "10", "math", 0, 0),
    ("no numbers here", "10", "math", 0, 0),
    ("\\boxed{unknown}", "10", "math", 0, 0),
]
bad = 0
for text, gt, kind, strict, lenient in cases:
    s, l = correct(text, gt, kind), correct_lenient(text, gt, kind)
    flag = "" if (s, l) == (strict, lenient) else "  <-- MISMATCH"
    bad += bool(flag)
    print(f"{kind:4s} strict {s} lenient {l} (expected {strict} {lenient}) ext={extract_lenient(text, kind)!r} | {text[:50]!r}{flag}")
print("MISMATCHES", bad)

"""rev6 strict reading, AMENDMENT 2 (7 Oct 2026, before the registered read rev6_read.py was run; after the interim looks of
6 Oct 23:26 and 23:59 at completed dumps). The base models often continue a solution with new, model-written
"Question: ... Answer: ..." blocks (Qwen3-8B-Base: 53% of samples; Qwen3-4B-Base: 12%), so strict(), which reads the last
'Answer:' line, read an unrelated question: it dropped Qwen3-8B-Base's accuracy on the answerable UMWP originals from 74% to
50%. Now a solution is first cut at its first line that starts a new question ('Question:'); strict() of
paper_arr/analyses/rev5/strict_parse.py (unchanged) decides whether the cut text gives an answer, and the first parser's rule,
applied to the cut text, gives the number (strict()'s own number only where that rule finds none). Against the two annotators'
answer status (592 solutions they agree on) this reading agrees on 94.6% (strict() on the uncut text: 92.4%; first parser:
85.5%; paper_arr/rev6/strict_v3_check.py)."""
import re, sys
R = "/home/tione/notebook/rsi_rl"
sys.path.insert(0, f"{R}/code"); from loto_common import last_boxed  # noqa: E402
sys.path.insert(0, f"{R}/dirB/code"); from pdc_util import is_abstain  # noqa: E402
_src = open(f"{R}/paper_arr/analyses/rev5/strict_parse.py").read(); _sp = {}
exec(compile(_src[:_src.index("# ---- validation against the human answer status")], "strict_parse_defs", "exec"), _sp)
strict, original, norm_ans, DECL, SYMB, UNIT_OK = _sp["strict"], _sp["original"], _sp["norm_ans"], _sp["DECL"], _sp["SYMB"], _sp["UNIT_OK"]
QRE = re.compile(r"\nQuestion:")
ABS_RE = re.compile(r"Answer:\s*(unknown|unanswerable|insufficient information)", re.I)


def cut(t):
    m = QRE.search(t)
    return t[:m.start()] if m else t


def probe_abs(t, f):
    """The probe's abstention-prompt rule (umwp_probe_text.py answers_of with --abstain_prompt 1)."""
    if f == "length": return None
    b = last_boxed(t)
    if b is not None and is_abstain(b): return None
    if b is not None: return norm_ans(b)
    if ABS_RE.search(t): return None
    return original(t, f)


def strict_abs(t, f):
    if f == "length": return None
    b = last_boxed(t)
    if b is not None and is_abstain(b): return None
    if b is not None: return norm_ans(b)
    if ABS_RE.search(t): return None
    return strict(t, f)


def read(t, f, prompt="plain"):
    """The strict reading of a sample: None (no answer) or its number."""
    tt = cut(t)
    s = (strict if prompt == "plain" else strict_abs)(tt, f)
    if s is None: return None
    o = (original if prompt == "plain" else probe_abs)(tt, f)
    return o if o is not None else s

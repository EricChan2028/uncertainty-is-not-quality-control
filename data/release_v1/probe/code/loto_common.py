"""Shared prompts, reward parsing, leak detection and family I/O for the LOTO experiments."""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

BOXED_INSTR = "Please reason step by step, and put your final answer within \\boxed{}."


def p_bare(problem: str) -> str:
    return f"{problem}\n{BOXED_INSTR}"


def p_note(problem: str, attempt: str, correct: bool, forbid_answer: bool = False, max_words: int = 60) -> str:
    verdict = "correct" if correct else "incorrect"
    extra = " Do not state the final answer of this problem." if forbid_answer else ""
    return (
        "You previously attempted the following problem.\n\n"
        f"Problem: {problem}\n\n"
        f"Your attempt:\n{attempt}\n\n"
        f"This attempt was {verdict}.\n\n"
        f"Write a short reusable note (at most {max_words} words) that would help you solve other problems of this kind."
        + extra
    )


def p_note_fb(problem: str, attempt: str, correct: bool, solution: str, forbid_answer: bool = False, max_words: int = 60,
              answer: Optional[str] = None, style: str = "default") -> str:
    """Writer prompt with outcome feedback: the reference solution (answer=None) or only the ground-truth final
    answer (answer given; the writer must infer any reusable fact itself)."""
    verdict = "correct" if correct else "incorrect"
    extra = " Do not state the final answer of this problem." if forbid_answer else ""
    fb = (f"This attempt was {verdict}. The correct final answer is: {answer}." if answer is not None
          else f"This attempt was {verdict}. The reference solution is:\n{solution}")
    if style == "extract":      # prompt-engineered baseline: tells the writer exactly what to extract
        instr = (f"The problem leaves out a fact that the reference solution uses. Write a short reusable note (at most {max_words} words) "
                 "that states that missing fact explicitly, with its number and unit, so that it can be used for other problems of this kind.")
    elif style == "infer":      # answer-only feedback: tells the writer to infer the missing fact
        instr = (f"The problem leaves out a fact that is needed to reach the correct answer. Work out what that fact must be from the correct answer, "
                 f"then write a short reusable note (at most {max_words} words) that states it explicitly, with its number and unit.")
    elif style == "infer_cot":  # prompted baseline with explicit reasoning before the note
        instr = ("The problem leaves out a fact that is needed to reach the correct answer. First reason step by step: write the equation that "
                 "links the given numbers, the missing fact and the correct answer, and solve it for the missing fact. Then, on a final line "
                 f"starting with 'Note:', write a short reusable note (at most {max_words} words) that states the missing fact explicitly, with its number and unit.")
    elif style == "infer_cot_brief":  # bounded reasoning, so the Note: line is reached inside the token budget
        instr = ("The problem leaves out a fact that is needed to reach the correct answer. First, in AT MOST two short sentences, "
                 "write the equation linking the given numbers, the missing fact and the correct answer, and solve it. "
                 f"Then, on a final line starting with Note:, write a reusable note (at most {max_words} words) that states the "
                 "missing fact explicitly, with its number and unit. Keep the reasoning short; the final line is what matters.")
    elif style == "infer_fewshot":  # prompted baseline with two worked examples
        instr = ("The problem leaves out a fact that is needed to reach the correct answer. Work out what that fact must be from the correct answer "
                 f"and write a short reusable note (at most {max_words} words) that states it explicitly, with its number and unit.\n\n"
                 "Example 1. Problem: Rosa worked 6 hours on Monday and 4 hours on Tuesday at her usual rate. How much did she earn? Correct answer: 150. "
                 "Reasoning: 10 hours earned 150, so the rate is 150 / 10 = 15. Note: Rosa's usual rate is $15 per hour.\n"
                 "Example 2. Problem: A school bought 7 boxes of markers and gave away 12 markers. How many markers are left? Correct answer: 72. "
                 "Reasoning: 72 + 12 = 84 markers came in 7 boxes, so each box holds 84 / 7 = 12. Note: At this school each box holds 12 markers.\n\n"
                 "Now write your reasoning and the note for the problem above.")
    else:
        instr = f"Write a short reusable note (at most {max_words} words) that would help you solve other problems of this kind."
    return (
        "You previously attempted the following problem.\n\n"
        f"Problem: {problem}\n\n"
        f"Your attempt:\n{attempt}\n\n"
        f"{fb}\n\n"
        + instr + extra
    )


def p_use(note: str, problem: str) -> str:
    return f"Note from a related problem:\n{note.strip()}\n\nProblem: {problem}\n{BOXED_INSTR}"


# ---------------------------------------------------------------- answers ---
_BOXED_RE = re.compile(r"\\boxed\s*\{")


def last_boxed(s: str) -> Optional[str]:
    idx = s.rfind("\\boxed")
    if idx < 0:
        return None
    i = s.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j]
    return None


def norm_num(s: str) -> Optional[str]:
    if s is None:
        return None
    t = s.strip().replace(",", "").replace("$", "").replace("\\%", "").replace("%", "")
    t = t.replace("\\!", "").replace(" ", "")
    t = re.sub(r"\\text\{[^}]*\}", "", t)
    t = t.rstrip(".")
    m = re.search(r"-?\d+(?:\.\d+)?(?:/\d+)?", t)
    if not m:
        return None
    t = m.group(0)
    try:
        if "/" in t:
            a, b = t.split("/")
            v = float(a) / float(b)
        else:
            v = float(t)
    except Exception:
        return None
    if v != v or v in (float("inf"), float("-inf")) or abs(v) > 1e15:   # nan / inf / absurd magnitudes (e.g. '1e999')
        return None
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.6g}"


def gsm_reward(response: str, gt: str) -> float:
    pred = norm_num(last_boxed(response) or "")
    if pred is None:
        # fall back to the last number in the response
        nums = re.findall(r"-?\d+(?:\.\d+)?", response.replace(",", ""))
        pred = norm_num(nums[-1]) if nums else None
    return 1.0 if pred is not None and pred == norm_num(gt) else 0.0


def _strip_math(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").replace("^{\\circ}", "").replace("^\\circ", "")
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", s)
    s = s.replace("\\$", "").replace("$", "").replace(" ", "")
    s = s.rstrip(".")
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    return s


def math_reward(response: str, gt: str) -> float:
    pred = last_boxed(response)
    if pred is None:
        return 0.0
    a, b = _strip_math(pred), _strip_math(gt)
    if a == b:
        return 1.0
    na, nb = norm_num(a), norm_num(b)
    if na is not None and nb is not None and re.fullmatch(r"-?[\d./]+", a.replace("\\frac", "")) is None:
        pass
    if na is not None and nb is not None and na == nb and re.fullmatch(r"-?[\d.,]+(/\d+)?", a) and re.fullmatch(r"-?[\d.,]+(/\d+)?", b):
        return 1.0
    try:  # optional: math_verify if available
        from math_verify import parse, verify  # type: ignore

        return 1.0 if verify(parse("\\boxed{" + gt + "}"), parse("\\boxed{" + pred + "}")) else 0.0
    except Exception:
        return 0.0


def reward_fn(substrate: str):
    return gsm_reward if (substrate.startswith("gsm") or substrate == "hid") else math_reward


# ------------------------------------------------------------------- leaks ---
_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def leaks_answer(note: str, gt: str, substrate: str, problem: Optional[str] = None) -> bool:
    """True if the ground-truth final answer of the source task appears in the note.

    Numbers that already occur verbatim in the problem statement are ignored (restating a given
    quantity is not a leak). For MATH the match is on token boundaries of the stripped answer.
    """
    given = set()
    if problem:
        given = {norm_num(x) for x in _NUM_RE.findall(problem)}
    if substrate.startswith("gsm") or substrate == "hid":
        g = norm_num(gt)
        if g is None or g in given:
            return False
        nums = [norm_num(x) for x in _NUM_RE.findall(note)]
        return g in nums
    b = _strip_math(gt)
    if not b:
        return False
    nb = norm_num(b)
    if nb is not None and re.fullmatch(r"-?[\d.,]+(/\d+)?", b):
        if nb in given:
            return False
        return nb in [norm_num(x) for x in _NUM_RE.findall(note)]
    stripped = _strip_math(note)
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(b) + r"(?![A-Za-z0-9])", stripped) is not None


def p_note_only(note: str) -> str:
    """Own-task prompt with the problem text hidden: measures whether the note determines the answer."""
    return f"Note from a related problem:\n{note.strip()}\n\nSolve the problem that the note refers to. {BOXED_INSTR}"


def has_boxed(response: str) -> bool:
    return "\\boxed" in response


# ---------------------------------------------------------------- families ---
def load_families(path: str, split: Optional[str] = None, substrate: Optional[str] = None) -> List[Dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if split and d["split"] != split:
                continue
            if substrate and d["substrate"] != substrate:
                continue
            out.append(d)
    return out


def note_len_words(note: str) -> int:
    return len(note.split())


def states_fact(note: str, H, problem: Optional[str] = None) -> bool:
    """Hidden-fact substrate: True if the family's hidden value H appears as a number in the note (and is
    not already a number of the problem statement)."""
    h = norm_num(str(H))
    if h is None:
        return False
    if problem and h in {norm_num(x) for x in _NUM_RE.findall(problem)}:
        return False
    return h in [norm_num(x) for x in _NUM_RE.findall(note)]

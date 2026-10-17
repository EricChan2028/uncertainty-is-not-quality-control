"""Label-free premise-deletion contrast (PDC) utilities shared by the trainer and the evaluators.

deleted_variant(problem, rng): drop one sentence that carries a number and is not the final question
sentence; returns (variant_text, deleted_sentence) or (None, None) when the problem has no such sentence.
The variant is NOT assumed ill-posed: the trainer decides label-free by comparing the majority answers
of the intact and deleted forms (a sentence whose removal leaves the majority unchanged was not needed).

ABSTAIN_SUFFIX is appended to every prompt of the abstention-aware arms (and of the matched evaluation)
so that "\\boxed{unknown}" is an available action for every question; is_abstain() recognises it.
"""
import re

ABSTAIN_SUFFIX = " If the problem does not provide enough information to determine the answer, output \\boxed{unknown}."
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$\\(])")


def split_sentences(text):
    parts = [p.strip() for p in _SENT.split(text.strip()) if p.strip()]
    return parts if parts else [text.strip()]


def deleted_variant(problem, rng):
    sents = split_sentences(problem)
    if len(sents) < 3:
        return None, None
    cand = [i for i, s in enumerate(sents[:-1]) if re.search(r"\d", s)]
    if not cand:
        return None, None
    i = rng.choice(cand)
    return " ".join(sents[:i] + sents[i + 1:]), sents[i]


def is_abstain(answer_text):
    """True when the (boxed) final answer is the abstention token."""
    if answer_text is None:
        return False
    t = re.sub(r"\\(text|mathrm|textbf)", "", answer_text).replace("{", "").replace("}", "").replace("$", "").strip().lower()
    return t in ("unknown", "unanswerable", "insufficient information", "not enough information", "cannot be determined")

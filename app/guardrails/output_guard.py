"""
output_guard.py
----------------
Validates model output before it's returned from api.py's public functions.

"""

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .pii_scrubber import scrub_pii, scrub_pii_in_dict

DISCLAIMER = (
    "\n\n_This information is drawn from the uploaded financial filings and is "
    "for informational purposes only — it is not financial advice._"
)

NOT_AVAILABLE_PHRASE = "not available in the provided context"

ADVICE_PATTERNS = [
    re.compile(r"\byou should (buy|sell|invest|hold)\b", re.I),
    re.compile(r"\bi (recommend|suggest) (buying|selling|investing)\b", re.I),
    re.compile(r"\bthis (stock|company) is a (good|bad|great) (buy|investment)\b", re.I),
]

NUMBER_PATTERN = re.compile(r"\$?\d[\d,]*\.?\d*%?")


@dataclass
class OutputGuardResult:
    text: Any
    flags: List[str] = field(default_factory=list)
    grounded: Optional[bool] = None


def _looks_like_advice(text: str) -> bool:
    return any(pattern.search(text) for pattern in ADVICE_PATTERNS)


def _check_grounding(answer: str, context: str) -> bool:
    """
    Every distinct number stated in the answer should appear somewhere
    in the retrieved chunks. Catches the model inventing a figure;
    won't catch subtler hallucinations (e.g. right number, wrong line item).
    """
    answer_numbers = set(NUMBER_PATTERN.findall(answer))
    if not answer_numbers:
        return True

    context_numbers = set(NUMBER_PATTERN.findall(context))
    norm = lambda s: s.replace("$", "").replace(",", "").rstrip("%")
    context_norm = {norm(n) for n in context_numbers}

    for num in answer_numbers:
        if norm(num) not in context_norm:
            return False
    return True


def check_answer(
    answer: str,
    retrieved_chunks: Optional[List[str]] = None,
    add_disclaimer: bool = True,
) -> OutputGuardResult:
    """
    Use for AnswerGenerator.generate_answer() and
    FinancialExtractor.generate_summary() output — both return plain strings.

    Args:
        answer: the raw string returned by generate_answer()/generate_summary()
        retrieved_chunks: pass the SAME list you got from
            `results["documents"][0]` (api.py already builds this) so the
            grounding check can verify numbers against it. Pass None to skip.
        add_disclaimer: append a disclaimer if advice-like language is found
    """
    flags: List[str] = []

    if not answer:
        return OutputGuardResult("I don't have an answer for that.", ["empty_answer"], None)

    clean_text, pii_labels = scrub_pii(answer)
    if pii_labels:
        flags.append("pii_redacted_in_output")

    grounded = None
    if retrieved_chunks:
        # The model correctly declining is not a grounding failure.
        if NOT_AVAILABLE_PHRASE not in clean_text.lower():
            context = "\n\n".join(retrieved_chunks)
            grounded = _check_grounding(clean_text, context)
            if not grounded:
                flags.append("ungrounded_numbers")
                clean_text += (
                    "\n\n_Note: some figures in this answer could not be "
                    "verified against the retrieved source chunks — please "
                    "double-check against the original filing._"
                )
        else:
            grounded = True

    if add_disclaimer and _looks_like_advice(clean_text):
        flags.append("advice_language_detected")
        clean_text += DISCLAIMER

    return OutputGuardResult(clean_text, flags, grounded)


def check_metrics(
    metrics,
    retrieved_chunks: Optional[List[str]] = None,
) -> OutputGuardResult:
    """
    Use for FinancialExtractor.extract_metrics() output specifically —
    it returns a dict (parsed JSON), not a string, so it needs its own path.

    Note: extractor._parse_json() falls back to returning the raw string
    if JSON parsing failed upstream — this function handles both cases
    so a parsing failure there doesn't crash the guard.
    """
    flags: List[str] = []

    if metrics is None:
        return OutputGuardResult("No metrics extracted.", ["empty_metrics"], None)

    # extractor.py's _parse_json() returns the raw string on failure
    if isinstance(metrics, str):
        return check_answer(metrics, retrieved_chunks, add_disclaimer=False)

    clean_metrics, pii_labels = scrub_pii_in_dict(metrics)
    if pii_labels:
        flags.append("pii_redacted_in_output")

    grounded = None
    if retrieved_chunks:
        context = "\n\n".join(retrieved_chunks)
        values_text = " ".join(str(v) for v in clean_metrics.values())
        # "Not Available" entries (per extractor.py's own instructions)
        # aren't numeric claims, so they can't fail grounding.
        grounded = _check_grounding(values_text, context)

    return OutputGuardResult(clean_metrics, flags, grounded)
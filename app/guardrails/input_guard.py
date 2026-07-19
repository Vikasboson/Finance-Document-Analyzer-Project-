"""
input_guard.py
----------------
Validates the `question` string before it reaches HybridRetriever /
AnswerGenerator / FinancialExtractor (i.e. before api.ask_question,
api.extract_financial_metrics, api.generate_financial_summary run).
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

from .pii_scrubber import scrub_pii

MAX_INPUT_CHARS = 2000
MIN_INPUT_CHARS = 1

INJECTION_PATTERNS = [
    re.compile(r"ignore (all|the) (previous|above) instructions", re.I),
    re.compile(r"disregard (all|the) (previous|above) (instructions|prompt)", re.I),
    re.compile(r"you are now (in )?(dan|developer mode|jailbreak)", re.I),
    re.compile(r"reveal (your|the) (system prompt|instructions)", re.I),
    re.compile(r"what (is|are) your (system prompt|instructions)", re.I),
    re.compile(r"pretend (you have|to have) no (rules|restrictions|guardrails)", re.I),
]

SYSTEM_PROBE_PATTERNS = [
    re.compile(r"\b(api[_ ]?key|aws[_ ]?secret|secret[_ ]?access[_ ]?key|password)\b", re.I),
    re.compile(r"\bshow me your (code|source|config|\.env)\b", re.I),
]

# Generic financial vocabulary — used ALONGSIDE known_companies so a
# question like "what was the operating margin?" isn't rejected just
# because it doesn't name a company explicitly (the retriever itself
# handles that: no company match -> unfiltered search, see retriever.py).
FINANCE_KEYWORDS = [
    "revenue", "earnings", "profit", "loss", "income", "eps", "guidance",
    "quarter", "q1", "q2", "q3", "q4", "fiscal", "margin", "growth",
    "balance sheet", "cash flow", "forecast", "stock", "share", "dividend",
    "financial", "10-k", "10-q", "10k", "10q", "sec filing", "report",
    "cost", "expense", "outlook", "segment", "capex", "metrics", "summary",
]


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""
    cleaned_text: str = ""
    flags: List[str] = field(default_factory=list)


def _is_on_topic(text: str, known_companies: Optional[Set[str]]) -> bool:
    lowered = text.lower()

    if any(keyword in lowered for keyword in FINANCE_KEYWORDS):
        return True

    if known_companies:
        if any(company and company in lowered for company in known_companies):
            return True

    return False


def check_question(
    question: str,
    known_companies: Optional[Set[str]] = None,
    enforce_topic: bool = True,
) -> GuardResult:
    """
    Validate a question before it's passed to retriever.retrieve(...).

    Args:
        question: raw user input (same param api.py's ask_question/
                  extract_financial_metrics/generate_financial_summary take)
        known_companies: pass retriever.dense_retriever.known_companies
                         (or hybrid_retriever.dense_retriever.known_companies)
                         so topic-scoping stays live with whatever is
                         actually ingested. Optional — pass None to skip
                         company-aware scoping and rely on FINANCE_KEYWORDS only.
        enforce_topic: set False to disable topic scoping entirely.

    Returns:
        GuardResult. If allowed=False, return `reason` directly to the
        user/UI without calling the retriever or LLM. If allowed=True,
        use `cleaned_text` (PII-scrubbed) as the question passed downstream.
    """
    flags: List[str] = []

    if question is None:
        return GuardResult(False, "Please enter a question.", "", flags)

    text = question.strip()

    if len(text) < MIN_INPUT_CHARS:
        return GuardResult(False, "Please enter a question.", "", flags)
    if len(text) > MAX_INPUT_CHARS:
        return GuardResult(
            False,
            f"Your question is too long (max {MAX_INPUT_CHARS} characters).",
            "",
            flags,
        )

    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            flags.append("prompt_injection")
            return GuardResult(False, "Sorry, I can't process that request.", "", flags)

    for pattern in SYSTEM_PROBE_PATTERNS:
        if pattern.search(text):
            flags.append("system_probe")
            return GuardResult(
                False,
                "I can't share internal system/config details, but I'm happy to "
                "help with questions about the ingested financial documents.",
                "",
                flags,
            )

    if enforce_topic and not _is_on_topic(text, known_companies):
        flags.append("off_topic")
        return GuardResult(
            False,
            "I'm scoped to answer questions about the financial documents "
            "that have been uploaded (earnings, revenue, margins, etc.). "
            "Could you rephrase your question around that?",
            "",
            flags,
        )

    cleaned_text, pii_labels = scrub_pii(text)
    if pii_labels:
        flags.append("pii_redacted_in_input")

    return GuardResult(True, "", cleaned_text, flags)
"""
pii_scrubber.py
----------------
Regex-based PII detection & redaction. No external dependencies.

IMPORTANT (rebuilt after reading ingestion.py / extractor.py):
This app's whole job is to surface large financial numbers
(revenue, EPS, market cap, share counts) verbatim from context. An
earlier version of this scrubber matched any bare 9-17 digit number as
an "account number" — that would have silently corrupted real figures
like "150000000000" coming straight out of a 10-K. Fixed here: the
account-number pattern now only fires when a nearby keyword
(account/acct/routing) makes intent unambiguous.
"""

import re
from typing import List, Tuple

PATTERNS = {
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    # Separators (space/dot/dash/parens) are REQUIRED, not optional. The
    # earlier optional-separator version could match a plain 10-12 digit
    # run with no formatting at all — e.g. it matched all 12 digits of
    # "150000000000" (a bare revenue figure) as a "phone number", because
    # the optional country-code group happened to consume exactly the
    # right number of leading digits. Financial figures in this app never
    # carry phone-style separators, so requiring them removes that risk.
    "PHONE": re.compile(r"(?<!\d)(\+?1[\s.-])?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)"),
    "SSN": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    # NOTE: still no-separator-tolerant (real cards are often typed as one
    # 16-digit block). A bare 16-digit financial figure is a much rarer
    # shape than a 10-12 digit one, so this trade-off is left as-is.
    "CREDIT_CARD": re.compile(r"(?<!\d)(?:\d{4}[\s-]?){3}\d{4}(?!\d)"),
    # Only matches when explicitly labeled as an account/routing number —
    # prevents false positives on revenue/share-count figures.
    "ACCOUNT_NUMBER": re.compile(
        r"(?:account|acct|routing)\s*(?:number|no\.?|#)?\s*[:\-]?\s*(\d{6,17})",
        re.I,
    ),
}

REDACTION_TOKEN = "[REDACTED_{label}]"


def find_pii(text: str) -> List[Tuple[str, str]]:
    """Return a list of (label, matched_value) tuples found in text."""
    findings: List[Tuple[str, str]] = []
    if not text:
        return findings
    for label, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append((label, match.group()))
    return findings


def scrub_pii(text: str) -> Tuple[str, List[str]]:
    """
    Replace detected PII with redaction tokens.
    Returns (clean_text, labels_found).
    """
    if not text:
        return text, []

    clean_text = text
    labels_found = set()

    # Apply ACCOUNT_NUMBER first: it's the most context-specific pattern
    # (requires an "account/acct/routing" keyword nearby), so it should
    # get first claim on those digits before PHONE's looser digit-run
    # matching has a chance to mislabel them.
    ordered_labels = ["ACCOUNT_NUMBER", "EMAIL", "SSN", "CREDIT_CARD", "PHONE"]

    for label in ordered_labels:
        pattern = PATTERNS[label]
        def _replace(match, label=label):
            labels_found.add(label)
            return REDACTION_TOKEN.format(label=label)
        clean_text = pattern.sub(_replace, clean_text)

    return clean_text, sorted(labels_found)


def scrub_pii_in_dict(data: dict) -> Tuple[dict, List[str]]:
    """
    Apply scrub_pii to every string value in a dict (one level deep),
    for use on FinancialExtractor.extract_metrics()'s JSON output.
    Non-string values (numbers, None) pass through untouched.
    """
    if not isinstance(data, dict):
        return data, []

    labels_found = set()
    cleaned = {}
    for key, value in data.items():
        if isinstance(value, str):
            clean_value, labels = scrub_pii(value)
            cleaned[key] = clean_value
            labels_found.update(labels)
        else:
            cleaned[key] = value
    return cleaned, sorted(labels_found)


def contains_pii(text: str) -> bool:
    return len(find_pii(text)) > 0
"""
evaluator.py — Answer accuracy evaluation for the Finance Analyser
-------------------------------------------------------------------
Runs each test case through the pipeline, extracts numbers from
the LLM's response, and compares against ground truth.

Scoring:
  EXACT    — number matches exactly              → 1.0
  CLOSE    — within 1% tolerance (rounding)       → 0.8
  PARTIAL  — right ballpark but off by >1%        → 0.3
  WRONG    — completely wrong number              → 0.0
  CORRECT_REFUSAL — model correctly said N/A      → 1.0
  FALSE_REFUSAL   — said N/A when answer exists   → 0.0
  HALLUCINATION   — gave a number for N/A case    → 0.0

Usage:
    python -m app.evaluation.evaluator
    python -m app.evaluation.evaluator --verbose
"""

import json
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class TestResult:
    test_id: str
    question: str
    expected: str
    actual: str
    score: float
    verdict: str
    latency_ms: float
    details: str = ""


@dataclass
class EvalReport:
    results: list[TestResult] = field(default_factory=list)
    total: int = 0
    passed: int = 0
    failed: int = 0
    avg_score: float = 0.0
    avg_latency_ms: float = 0.0

    def summary(self) -> str:
        lines = [
            "",
            "=" * 60,
            "  EVALUATION REPORT",
            "=" * 60,
            f"  Total test cases:   {self.total}",
            f"  Passed (≥0.8):      {self.passed}",
            f"  Failed (<0.8):      {self.failed}",
            f"  Accuracy score:     {self.avg_score:.1%}",
            f"  Avg latency:        {self.avg_latency_ms:.0f} ms",
            "=" * 60,
            "",
        ]

        for r in self.results:
            icon = "✅" if r.score >= 0.8 else "❌"
            lines.append(f"  {icon} {r.test_id}")
            lines.append(f"     Q: {r.question}")
            lines.append(f"     Expected: {r.expected}")
            lines.append(f"     Got:      {r.actual[:120]}")
            lines.append(f"     Verdict:  {r.verdict} (score: {r.score})")
            if r.details:
                lines.append(f"     Detail:   {r.details}")
            lines.append(f"     Latency:  {r.latency_ms:.0f} ms")
            lines.append("")

        return "\n".join(lines)


# ── Number extraction from LLM responses ────────────────────────

def extract_numbers(text: str) -> list[float]:
    """Pull all numbers from a text string.
    Handles: $77.7 billion, 143,756, 42097, $3,434M, 18.5%, etc.
    """
    # Remove currency symbols and common separators
    cleaned = text.replace("$", "").replace("€", "").replace("£", "")

    # Find all number patterns (with commas, decimals)
    raw_numbers = re.findall(r"[\d,]+\.?\d*", cleaned)

    results = []
    for n in raw_numbers:
        try:
            val = float(n.replace(",", ""))
            results.append(val)
        except ValueError:
            continue
    return results


def is_not_available(text: str) -> bool:
    """Check if the response is a 'not available' refusal."""
    lower = text.lower()
    refusal_phrases = [
        "not available", "not found", "not present",
        "not mentioned", "does not appear", "no information",
        "cannot find", "doesn't appear", "not explicitly",
        "not provided", "no data",
    ]
    return any(phrase in lower for phrase in refusal_phrases)


# ── Scoring logic ───────────────────────────────────────────────

def score_qa(expected: str, actual: str) -> tuple[float, str, str]:
    """Compare expected vs actual answer.
    Returns (score, verdict, details).
    """
    # Case 1: expected is NOT_AVAILABLE (hallucination test)
    if expected == "NOT_AVAILABLE":
        if is_not_available(actual):
            return 1.0, "CORRECT_REFUSAL", "Model correctly refused"
        else:
            nums = extract_numbers(actual)
            if nums:
                return 0.0, "HALLUCINATION", f"Gave number(s) {nums} for non-existent data"
            return 0.5, "AMBIGUOUS_REFUSAL", "Didn't clearly refuse or give a number"

    # Case 2: expected is a number
    try:
        expected_num = float(expected.replace(",", ""))
    except ValueError:
        # Expected is text, do substring match
        if expected.lower() in actual.lower():
            return 1.0, "EXACT_TEXT", "Text match found"
        return 0.0, "WRONG", f"Expected '{expected}' not found in response"

    # Check if model refused when it shouldn't have
    if is_not_available(actual):
        return 0.0, "FALSE_REFUSAL", "Model said N/A but answer exists"

    # Extract numbers from the actual response
    actual_nums = extract_numbers(actual)
    if not actual_nums:
        return 0.0, "NO_NUMBER", "No numbers found in response"

    # Check if the expected number appears in the response
    for num in actual_nums:
        if num == expected_num:
            return 1.0, "EXACT", f"Exact match: {num}"

        # Within 1% tolerance (rounding differences)
        if expected_num != 0:
            pct_diff = abs(num - expected_num) / abs(expected_num)
            if pct_diff <= 0.01:
                return 0.8, "CLOSE", f"Within 1%: got {num}, expected {expected_num}"

    # Check closest number
    closest = min(actual_nums, key=lambda x: abs(x - expected_num))
    if expected_num != 0:
        pct_off = abs(closest - expected_num) / abs(expected_num) * 100
        return 0.0, "WRONG_NUMBER", f"Closest: {closest} ({pct_off:.1f}% off)"

    return 0.0, "WRONG", "No matching number found"


def score_metrics(expected_metrics: dict, actual_metrics) -> tuple[float, str, str]:
    """Compare expected vs actual extracted metrics dict."""
    if not isinstance(actual_metrics, dict):
        return 0.0, "PARSE_FAIL", "Metrics extraction didn't return a dict"

    matches = 0
    total_checked = 0
    details_parts = []

    for key, expected_val in expected_metrics.items():
        total_checked += 1
        actual_val = actual_metrics.get(key, "Missing")

        if actual_val == "Missing":
            details_parts.append(f"{key}: MISSING")
            continue

        # Compare as numbers
        try:
            exp_num = float(str(expected_val).replace(",", ""))
            act_num = float(str(actual_val).replace(",", ""))

            if exp_num == act_num:
                matches += 1
                details_parts.append(f"{key}: ✓")
            elif exp_num != 0 and abs(act_num - exp_num) / abs(exp_num) <= 0.01:
                matches += 0.8
                details_parts.append(f"{key}: ~close ({act_num})")
            else:
                details_parts.append(f"{key}: ✗ (got {actual_val}, want {expected_val})")
        except (ValueError, TypeError):
            # String comparison
            if str(actual_val).strip().lower() == str(expected_val).strip().lower():
                matches += 1
                details_parts.append(f"{key}: ✓")
            else:
                details_parts.append(f"{key}: ✗ (got {actual_val})")

    score = matches / total_checked if total_checked > 0 else 0.0
    verdict = "PASS" if score >= 0.8 else "PARTIAL" if score >= 0.5 else "FAIL"
    details = ", ".join(details_parts)

    return round(score, 2), verdict, details


# ── Main evaluator ──────────────────────────────────────────────

def run_evaluation(dataset_path: str = None, verbose: bool = False) -> EvalReport:
    """Load test cases, run through pipeline, score, return report."""

    # Import the pipeline functions
    from app.backend.api import ask_question, extract_financial_metrics

    # Load test dataset
    if dataset_path is None:
        dataset_path = Path(__file__).parent / "test_dataset.json"

    with open(dataset_path) as f:
        dataset = json.load(f)

    test_cases = dataset["test_cases"]
    report = EvalReport()

    print(f"\nRunning {len(test_cases)} evaluation tests...\n")

    for i, tc in enumerate(test_cases, 1):
        test_id = tc["id"]
        question = tc["question"]
        task = tc.get("task", "question_answering")

        # Skip placeholder test cases
        expected_raw = tc.get("expected_answer", "") or ""
        expected_metrics = tc.get("expected_metrics", {})
        if expected_raw == "FILL_IN_FROM_YOUR_PDF":
            if verbose:
                print(f"  ⏭️  {test_id} — skipped (placeholder, fill in ground truth)")
            continue

        print(f"  [{i}/{len(test_cases)}] {test_id}...", end=" ", flush=True)

        # Run the query
        t0 = time.perf_counter()
        try:
            if task == "extract_metrics":
                actual = extract_financial_metrics(question)
            else:
                actual = ask_question(question)
        except Exception as exc:
            actual = f"ERROR: {exc}"
        latency = round((time.perf_counter() - t0) * 1000, 1)

        # Score it
        if task == "extract_metrics":
            sc, verdict, details = score_metrics(expected_metrics, actual)
            expected_display = json.dumps(expected_metrics, indent=None)
            actual_display = json.dumps(actual, indent=None) if isinstance(actual, dict) else str(actual)
        else:
            sc, verdict, details = score_qa(expected_raw, str(actual))
            expected_display = expected_raw
            actual_display = str(actual)

        icon = "✅" if sc >= 0.8 else "❌"
        print(f"{icon} {verdict} ({sc})")

        report.results.append(TestResult(
            test_id=test_id,
            question=question,
            expected=expected_display,
            actual=actual_display,
            score=sc,
            verdict=verdict,
            latency_ms=latency,
            details=details,
        ))

    # Compute totals
    report.total = len(report.results)
    report.passed = sum(1 for r in report.results if r.score >= 0.8)
    report.failed = report.total - report.passed
    report.avg_score = (
        sum(r.score for r in report.results) / report.total
        if report.total > 0 else 0.0
    )
    report.avg_latency_ms = (
        sum(r.latency_ms for r in report.results) / report.total
        if report.total > 0 else 0.0
    )

    return report


# ── CLI entry point ─────────────────────────────────────────────

if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    report = run_evaluation(verbose=verbose)
    print(report.summary())
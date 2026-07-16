"""
guardrails package
-------------------
Rebuilt against the actual pipeline (api.py, generator.py, retriever.py,
extractor.py, ingestion.py). See app/guardrails/README.md for the
exact api.py integration.

Public API:
    check_question(question, known_companies)   -> GuardResult   (input)
    check_answer(answer, retrieved_chunks)       -> OutputGuardResult (strings)
    check_metrics(metrics, retrieved_chunks)     -> OutputGuardResult (dict)
    check_upload(uploaded_file)                  -> GuardResult   (file upload)
    scrub_pii(text) / scrub_pii_in_dict(data)     -> low-level helpers
"""

from .input_guard import check_question, GuardResult as InputGuardResult
from .output_guard import check_answer, check_metrics, OutputGuardResult
from .file_guard import check_upload, GuardResult as FileGuardResult
from .pii_scrubber import scrub_pii, scrub_pii_in_dict, find_pii, contains_pii

__all__ = [
    "check_question",
    "InputGuardResult",
    "check_answer",
    "check_metrics",
    "OutputGuardResult",
    "check_upload",
    "FileGuardResult",
    "scrub_pii",
    "scrub_pii_in_dict",
    "find_pii",
    "contains_pii",
]
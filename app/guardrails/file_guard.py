"""
file_guard.py
--------------
Validates a file BEFORE api.py's upload_document() writes it to disk
and ingests it.

Why this file exists (found while reading api.py):
Streamlit_app.py restricts uploads to type=["pdf", "txt"], but that's a
UI-level hint only — st.file_uploader does not stop a renamed file, and
anyone calling api.upload_document() directly (script, test, future API
endpoint) bypasses it completely. api.py itself does no validation
before doing `open(file_path, "wb").write(...)`. This guard closes that
gap without touching api.py's ingestion logic itself.
"""

import re
from dataclasses import dataclass, field
from typing import List

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB

# Reject filenames containing path traversal or separator characters
_UNSAFE_FILENAME_PATTERN = re.compile(r"(\.\.|[/\\])")


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""
    flags: List[str] = field(default_factory=list)


def check_upload(uploaded_file) -> GuardResult:
    """
    Args:
        uploaded_file: the same object api.upload_document() receives —
            must expose `.name` and `.getbuffer()` (Streamlit's
            UploadedFile satisfies both; anything else must too).

    Returns:
        GuardResult. If allowed=False, reject before calling
        ingestion.ingest_file(...) or writing anything to disk.
    """
    flags: List[str] = []

    if uploaded_file is None or not getattr(uploaded_file, "name", None):
        return GuardResult(False, "No file provided.", flags)

    name = uploaded_file.name

    if _UNSAFE_FILENAME_PATTERN.search(name):
        flags.append("unsafe_filename")
        return GuardResult(False, "Filename contains invalid characters.", flags)

    suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if suffix not in ALLOWED_EXTENSIONS:
        flags.append("disallowed_extension")
        return GuardResult(
            False,
            f"Unsupported file type '{suffix or 'unknown'}'. "
            f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}.",
            flags,
        )

    try:
        size = len(uploaded_file.getbuffer())
    except Exception:
        flags.append("unreadable_buffer")
        return GuardResult(False, "Could not read the uploaded file.", flags)

    if size == 0:
        return GuardResult(False, "The uploaded file is empty.", flags)

    if size > MAX_FILE_SIZE_BYTES:
        flags.append("file_too_large")
        return GuardResult(
            False,
            f"File is too large ({size / 1_048_576:.1f} MB). "
            f"Max allowed is {MAX_FILE_SIZE_BYTES / 1_048_576:.0f} MB.",
            flags,
        )

    return GuardResult(True, "", flags)
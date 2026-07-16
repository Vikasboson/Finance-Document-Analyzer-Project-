"""
Streamlit_app.py — works on both local and Streamlit Cloud
-----------------------------------------------------------
On Streamlit Cloud, secrets from the dashboard are loaded into
os.environ at startup so all existing os.getenv() calls in
generator.py, extractor.py, langfuse_config.py work unchanged.
"""

import os
import re
import uuid
import time

import streamlit as st
import sys
from pathlib import Path

# ── Bridge: Streamlit Cloud secrets → os.environ ─────────────────
# This runs BEFORE any imports that call os.getenv()
# On local, .env.example is loaded by dotenv as usual
# On Cloud, st.secrets fills os.environ instead
try:
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:
    pass  # No secrets configured (local dev) — dotenv handles it

# ── Now safe to import the app ───────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.backend.api import (
    ask_question,
    extract_financial_metrics,
    generate_financial_summary,
    get_available_companies,
    upload_document,
)

# ── Session ──────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
SESSION_ID = st.session_state.session_id


# ══════════════════════════════════════════════════════════════════
#  Scoring functions
# ══════════════════════════════════════════════════════════════════

def _extract_numbers(text: str) -> list[float]:
    raw = re.findall(r"[\d,]+\.?\d*", text.replace("$", "").replace("€", ""))
    nums = []
    for n in raw:
        try:
            nums.append(float(n.replace(",", "")))
        except ValueError:
            pass
    return nums


def _is_refusal(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in [
        "not available", "not found", "not present", "not mentioned",
        "does not appear", "no information", "cannot find",
        "not provided", "no data", "not explicitly",
    ])


def score_qa_response(answer: str) -> dict:
    if _is_refusal(answer):
        return {"score": 0.7, "label": "Acknowledged unavailable", "icon": "🟡",
                "detail": "Model correctly indicated data is not available in context."}

    nums = _extract_numbers(answer)
    word_count = len(answer.split())

    if not nums:
        return {"score": 0.3, "label": "No numbers cited", "icon": "🔴",
                "detail": "Answer doesn't contain specific financial figures."}

    if word_count < 5:
        return {"score": 0.5, "label": "Too brief", "icon": "🟡",
                "detail": f"Found {len(nums)} number(s) but answer is very short."}

    has_citation = any(w in answer.lower() for w in [
        "according", "reported", "stated", "per ", "line item",
        "quarter", "fiscal", "period", "q1", "q2", "q3", "q4",
        "fy", "million", "billion", "revenue", "income", "margin",
    ])

    if has_citation:
        return {"score": 1.0, "label": "Grounded with citation", "icon": "🟢",
                "detail": f"Answer cites {len(nums)} specific number(s) with source context."}

    return {"score": 0.8, "label": "Contains financial data", "icon": "🟢",
            "detail": f"Answer contains {len(nums)} number(s)."}


def score_metrics_response(metrics) -> dict:
    if not isinstance(metrics, dict):
        return {"score": 0.0, "label": "Extraction failed", "icon": "🔴",
                "detail": "Metrics extraction did not return valid JSON."}

    total = len(metrics)
    found = sum(1 for v in metrics.values()
                if str(v).strip().lower() not in ("not available", "n/a", "none", ""))

    if total == 0:
        return {"score": 0.0, "label": "No metrics", "icon": "🔴", "detail": "Empty result."}

    coverage = found / total
    if coverage >= 0.75:
        icon, label = "🟢", f"{found}/{total} metrics extracted"
    elif coverage >= 0.5:
        icon, label = "🟡", f"{found}/{total} metrics extracted"
    else:
        icon, label = "🔴", f"Only {found}/{total} metrics found"

    return {
        "score": round(coverage, 2), "label": label, "icon": icon,
        "detail": ", ".join(
            f"{'✓' if str(v).strip().lower() not in ('not available','n/a','none','') else '✗'} {k}"
            for k, v in metrics.items()
        ),
    }


def score_summary_response(summary: str) -> dict:
    nums = _extract_numbers(summary)
    word_count = len(summary.split())

    if word_count < 20:
        return {"score": 0.3, "label": "Too short", "icon": "🔴",
                "detail": f"Only {word_count} words."}
    if not nums:
        return {"score": 0.4, "label": "No financial figures", "icon": "🟡",
                "detail": "Summary doesn't cite any specific numbers."}
    if len(nums) >= 4:
        return {"score": 1.0, "label": f"Rich summary ({len(nums)} figures)", "icon": "🟢",
                "detail": f"{word_count} words with {len(nums)} financial figures cited."}

    score = min(1.0, 0.5 + len(nums) * 0.1)
    return {"score": round(score, 2), "label": f"{len(nums)} figures cited",
            "icon": "🟢" if score >= 0.7 else "🟡",
            "detail": f"{word_count} words with {len(nums)} financial figure(s)."}


def display_score(score_data: dict, latency_ms: float):
    col1, col2, col3 = st.columns([1, 2, 1])
    col1.metric("Accuracy", f"{score_data['score']:.0%}", delta=score_data["icon"])
    col2.caption(f"**{score_data['label']}** — {score_data['detail']}")
    col3.metric("Latency", f"{latency_ms:.0f}ms")


# ══════════════════════════════════════════════════════════════════
#  Main app
# ══════════════════════════════════════════════════════════════════

st.title("Financial Document Analyzer")

companies = get_available_companies()
if companies:
    st.caption(f"📂 Companies in knowledge base: **{', '.join(sorted(companies))}**")
else:
    st.caption("📂 No documents ingested yet. Upload one below.")

st.divider()

# ── Upload ───────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a financial document", type=["pdf", "txt"])

if uploaded_file is not None:
    if st.button("Upload Document"):
        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            try:
                result = upload_document(uploaded_file, user_id="streamlit_user")
                chunks = result.get("chunks", "?") if isinstance(result, dict) else "?"
                st.success(f"✅ **{uploaded_file.name}** uploaded — {chunks} chunks created.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Upload failed: {e}")

st.divider()

# ── Task ─────────────────────────────────────────────────────────
mode = st.selectbox(
    "Select Task",
    ["Question Answering", "Extract Financial Metrics", "Generate Financial Summary"],
)

question = st.text_input("Enter your question")

if st.button("Submit"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching and generating answer..."):
            t0 = time.perf_counter()
            try:
                if mode == "Question Answering":
                    result = ask_question(question, user_id="streamlit_user",
                                          session_id=SESSION_ID)
                elif mode == "Extract Financial Metrics":
                    result = extract_financial_metrics(question, user_id="streamlit_user",
                                                       session_id=SESSION_ID)
                else:
                    result = generate_financial_summary(question, user_id="streamlit_user",
                                                        session_id=SESSION_ID)
                latency = round((time.perf_counter() - t0) * 1000, 1)

                st.write(result)
                st.divider()

                if mode == "Question Answering":
                    sc = score_qa_response(str(result))
                elif mode == "Extract Financial Metrics":
                    sc = score_metrics_response(result)
                else:
                    sc = score_summary_response(str(result))

                display_score(sc, latency)

            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")

# ── Footer ───────────────────────────────────────────────────────
langfuse_host = os.getenv("LANGFUSE_HOST", "")
if langfuse_host:
    st.divider()
    st.caption(f"📊 [Open Langfuse Dashboard]({langfuse_host})")
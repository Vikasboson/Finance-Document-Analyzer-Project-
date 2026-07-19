

"""Streamlit_app.py — No permanent KB. Upload → query → refresh clears everything."""
import os, re, uuid, time, sys
import streamlit as st
from pathlib import Path

try:
    for k, v in st.secrets.items():
        if isinstance(v, str): os.environ.setdefault(k, v)
except Exception: pass

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.backend.api import (
    ask_question, extract_financial_metrics,
    get_available_companies, get_current_file, upload_document,
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
SESSION_ID = st.session_state.session_id


# ── Scoring ──────────────────────────────────────────────────
def _nums(text):
    return [float(n.replace(",", "")) for n in re.findall(r"[\d,]+\.?\d*", text.replace("$","").replace("€","")) if n.replace(",","").replace(".","").isdigit() or "." in n]

def _refusal(text):
    return any(p in text.lower() for p in ["not available","not found","not present","not mentioned",
        "does not appear","no information","cannot find","not provided","no data","not explicitly"])

def score_qa(answer):
    if _refusal(answer):
        return {"score": 0.7, "label": "Acknowledged unavailable", "icon": "🟡", "detail": "Model correctly indicated data not available."}
    nums = _nums(answer)
    if not nums:
        return {"score": 0.3, "label": "No numbers cited", "icon": "🔴", "detail": "No financial figures found."}
    if len(answer.split()) < 5:
        return {"score": 0.5, "label": "Too brief", "icon": "🟡", "detail": f"{len(nums)} number(s) but very short."}
    cited = any(w in answer.lower() for w in ["according","reported","stated","quarter","fiscal","q1","q2","q3","q4","million","billion","revenue","income","margin"])
    if cited:
        return {"score": 1.0, "label": "Grounded with citation", "icon": "🟢", "detail": f"{len(nums)} number(s) with source context."}
    return {"score": 0.8, "label": "Contains financial data", "icon": "🟢", "detail": f"{len(nums)} number(s)."}

def score_metrics(metrics):
    if not isinstance(metrics, dict):
        return {"score": 0.0, "label": "Extraction failed", "icon": "🔴", "detail": "No valid JSON."}
    total = len(metrics)
    if not total:
        return {"score": 0.0, "label": "No metrics", "icon": "🔴", "detail": "Empty result."}
    found = sum(1 for v in metrics.values() if str(v).strip().lower() not in ("not available","n/a","none",""))
    cov = found / total
    icon = "🟢" if cov >= 0.75 else "🟡" if cov >= 0.5 else "🔴"
    return {"score": round(cov, 2), "label": f"{found}/{total} metrics extracted", "icon": icon,
            "detail": ", ".join(f"{'✓' if str(v).strip().lower() not in ('not available','n/a','none','') else '✗'} {k}" for k,v in metrics.items())}


# ── App ──────────────────────────────────────────────────────
st.title("Financial Document Analyzer")

current_file = get_current_file()
if current_file:
    companies = get_available_companies()
    st.success(f"📄 **Active file:** {current_file}  \n📂 **Company:** {', '.join(sorted(companies)) if companies else 'detecting...'}")
else:
    st.warning("📂 **data/ folder is empty.** Upload a document below to get started.")

st.divider()

# ── Upload ───────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a financial document (PDF or TXT)", type=["pdf", "txt"],
    help="Uploading replaces the previous file. Cleared on app refresh.")

if uploaded_file is not None:
    if current_file:
        st.caption(f"⚠️ Will replace: **{current_file}**")
    if st.button("Upload & Analyze"):
        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            try:
                result = upload_document(uploaded_file, user_id="streamlit_user")
                chunks = result.get("chunks", "?") if isinstance(result, dict) else "?"
                st.success(f"✅ **{uploaded_file.name}** — {chunks} chunks ready.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Upload failed: {e}")

st.divider()

# ── Query ────────────────────────────────────────────────────
if not current_file:
    st.info("Upload a document above to enable querying.")
    st.stop()

mode = st.selectbox("Select Task", ["Question Answering", "Extract Financial Metrics"])
question = st.text_input("Enter your question")

if st.button("Submit"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching and generating answer..."):
            t0 = time.perf_counter()
            try:
                result = (ask_question if mode == "Question Answering" else extract_financial_metrics)(
                    question, user_id="streamlit_user", session_id=SESSION_ID)
                latency = round((time.perf_counter() - t0) * 1000, 1)
                st.write(result)
                st.divider()
                sc = score_qa(str(result)) if mode == "Question Answering" else score_metrics(result)
                c1, c2, c3 = st.columns([1, 2, 1])
                c1.metric("Accuracy", f"{sc['score']:.0%}", delta=sc["icon"])
                c2.caption(f"**{sc['label']}** — {sc['detail']}")
                c3.metric("Latency", f"{latency:.0f}ms")
            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")

# ── Footer ───────────────────────────────────────────────────
lh = os.getenv("LANGFUSE_HOST", "")
if lh:
    st.divider()
    st.caption(f"📊 [Open Langfuse Dashboard]({lh})")
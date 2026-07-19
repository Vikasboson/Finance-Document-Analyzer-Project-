"""api.py — No permanent KB. data/ cleared on startup. Upload → clear → ingest → query."""
import time, chromadb
from pathlib import Path
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.generator import AnswerGenerator
from app.extraction.extractor import FinancialExtractor
from app.rag.ingestion import DocumentIngestion
from app.observability import trace_ops

try:
    from langfuse import observe
except ImportError:
    def observe(*a, **kw): return a[0] if a and callable(a[0]) else lambda fn: fn

try:
    from app.guardrails import check_input, check_output
    _GUARDS = True
except ImportError:
    _GUARDS = False

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

# ── ONE shared client + ONE shared collection ────────────────
shared_client = chromadb.PersistentClient(path="./chroma_datab")
shared_collection = shared_client.get_or_create_collection(
    name="documents", metadata={"hnsw:space": "cosine"}
)


def _clear_data():
    """Empty the data/ folder."""
    for f in DATA_DIR.iterdir():
        if f.is_file(): f.unlink()


def _clear_vectors():
    """Remove all docs from the shared collection. NEVER deletes
    the collection itself — so no stale UUID references."""
    try:
        ids = shared_collection.get()["ids"]
        if ids:
            shared_collection.delete(ids=ids)
    except Exception as e:
        print(f"[clear] {e}")


# ── Startup: wipe data/ and vectors ─────────────────────────
_clear_data()
_clear_vectors()

# All components receive the SAME client — they all call
# get_or_create_collection internally, which returns the SAME
# underlying collection (same UUID) as shared_collection.
ingestion = DocumentIngestion(
    chroma_client=shared_client, collection_name="documents", data_path=str(DATA_DIR)
)
retriever = HybridRetriever(chroma_client=shared_client, collection_name="documents")
generator, extractor = AnswerGenerator(), FinancialExtractor()


# ── Helpers ──────────────────────────────────────────────────
def _retrieve(q):
    with trace_ops.span("retrieval", input={"query": q}) as s:
        r = retriever.retrieve(q, top_k=8)
        s.update(output={"n_chunks": len(r["documents"][0])})
    return r["documents"][0], r["metadatas"][0]


def _gen(llm, method, chunks, *, span, **kw):
    with trace_ops.generation(span, model=llm.model_id) as g:
        res = method(retrieved_chunks=chunks, **kw)
        g.update(output=res)
        return res, trace_ops.attach_usage(llm, g)


# ── Public API ───────────────────────────────────────────────
@observe(name="ask_question")
def ask_question(question: str, *, user_id="anonymous", session_id=None):
    t0, costs = time.perf_counter(), {}
    try:
        if _GUARDS:
            guard = check_input(question)
            if not guard.allowed: return guard.reason
            question = getattr(guard, "cleaned_text", question)
        chunks, metas = _retrieve(question)
        answer, costs = _gen(generator, generator.generate_answer, chunks,
                             span="answer_generation", query=question, metadatas=metas)
        if _GUARDS:
            guarded = check_output(answer, context=chunks)
            answer = getattr(guarded, "text", answer)
        trace_ops.score_trace("error_occurred", 0.0)
        return answer
    except Exception:
        trace_ops.score_trace("error_occurred", 1.0); raise
    finally:
        trace_ops.score_trace("total_latency_ms", round((time.perf_counter()-t0)*1000, 1))
        trace_ops.score_trace("total_cost_usd", costs.get("total_cost", 0))


@observe(name="extract_financial_metrics")
def extract_financial_metrics(question: str, *, user_id="anonymous", session_id=None):
    t0, costs = time.perf_counter(), {}
    try:
        chunks, metas = _retrieve(question)
        metrics, costs = _gen(extractor, extractor.extract_metrics, chunks,
                              span="metrics_extraction", query=question, metadatas=metas)
        if isinstance(metrics, dict):
            t = len(metrics)
            trace_ops.score_trace("metrics_coverage",
                                  sum(1 for v in metrics.values() if v != "Not Available") / t if t else 0)
        trace_ops.score_trace("error_occurred", 0.0)
        return metrics
    except Exception:
        trace_ops.score_trace("error_occurred", 1.0); raise
    finally:
        trace_ops.score_trace("total_latency_ms", round((time.perf_counter()-t0)*1000, 1))
        trace_ops.score_trace("total_cost_usd", costs.get("total_cost", 0))


@observe(name="upload_document")
def upload_document(uploaded_file, *, user_id="anonymous"):
    """Clear previous → save to data/ → ingest → refresh."""
    _clear_data()
    _clear_vectors()
    path = DATA_DIR / uploaded_file.name
    path.write_bytes(uploaded_file.getbuffer())
    result = ingestion.ingest_file(str(path))
    retriever.refresh()
    return result


def get_available_companies():
    return retriever.dense_retriever.known_companies


def get_current_file():
    if DATA_DIR.exists():
        files = [f.name for f in DATA_DIR.iterdir() if f.is_file()]
        return ", ".join(files) if files else None
    return None
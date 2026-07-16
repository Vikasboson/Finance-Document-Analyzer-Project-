"""
api.py — Langfuse v4.14 instrumented backend
"""

import time
import chromadb
from pathlib import Path

from app.rag.hybrid_retriever import HybridRetriever
from app.rag.generator import AnswerGenerator
from app.extraction.extractor import FinancialExtractor
from app.rag.ingestion import DocumentIngestion

from app.observability import trace_ops
from app.observability.langfuse_config import langfuse, is_enabled

# v4 @observe decorator — auto-creates a trace per function call
try:
    from langfuse import observe
except ImportError:
    # Fallback: plain passthrough if langfuse not installed
    def observe(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn

try:
    from app.guardrails import check_input, check_output
    _GUARDS = True
except ImportError:
    _GUARDS = False

# ── Shared ChromaDB client ──────────────────────────────────────
CHROMA_PATH = "./chroma_datab"
COLLECTION  = "documents"
shared_client = chromadb.PersistentClient(path=CHROMA_PATH)

ingestion = DocumentIngestion(chroma_client=shared_client, collection_name=COLLECTION)
ingestion.run()

retriever = HybridRetriever(chroma_client=shared_client, collection_name=COLLECTION)
generator = AnswerGenerator()
extractor = FinancialExtractor()


# ── Shared helpers ──────────────────────────────────────────────

def _retrieve(question):
    with trace_ops.span("retrieval", input={"query": question}) as s:
        results = retriever.retrieve(question)
        chunks = results["documents"][0]
        s.update(output={"n_chunks": len(chunks)})
    return chunks


def _generate(llm_obj, method, chunks, *, span_name, **method_kwargs):
    with trace_ops.generation(span_name, model=llm_obj.model_id) as g:
        result = method(retrieved_chunks=chunks, **method_kwargs)
        g.update(output=result)
        costs = trace_ops.attach_usage(llm_obj, g)
    return result, costs


# ── Public API ──────────────────────────────────────────────────

@observe(name="ask_question")
def ask_question(question: str, *, user_id="anonymous", session_id=None):
    t0 = time.perf_counter()
    costs = {}

    try:
        # Input guard
        if _GUARDS:
            with trace_ops.span("input_guard", input=question) as s:
                guard = check_input(question)
                s.update(output={"allowed": guard.allowed})
            trace_ops.score_trace("input_guard_passed",
                                  1.0 if guard.allowed else 0.0)
            if not guard.allowed:
                return guard.reason
            question = getattr(guard, "cleaned_text", question)

        # Retrieve → Generate
        chunks = _retrieve(question)
        answer, costs = _generate(
            generator, generator.generate_answer, chunks,
            span_name="answer_generation", query=question,
        )

        # Output guard
        if _GUARDS:
            with trace_ops.span("output_guard") as s:
                guarded = check_output(answer, context=chunks)
                s.update(output={"grounded": getattr(guarded, "grounded", None)})
            trace_ops.score_trace("output_grounded",
                                  1.0 if getattr(guarded, "grounded", True) else 0.0)
            answer = getattr(guarded, "text", answer)

        trace_ops.score_trace("error_occurred", 0.0)
        return answer

    except Exception as exc:
        trace_ops.score_trace("error_occurred", 1.0)
        raise
    finally:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        trace_ops.score_trace("total_latency_ms", ms)
        trace_ops.score_trace("total_cost_usd", costs.get("total_cost", 0))


@observe(name="extract_financial_metrics")
def extract_financial_metrics(question: str, *, user_id="anonymous", session_id=None):
    t0 = time.perf_counter()
    costs = {}

    try:
        chunks = _retrieve(question)
        metrics, costs = _generate(
            extractor, extractor.extract_metrics, chunks,
            span_name="metrics_extraction",
        )

        if isinstance(metrics, dict):
            total = len(metrics)
            found = sum(1 for v in metrics.values() if v != "Not Available")
            trace_ops.score_trace("metrics_coverage",
                                  found / total if total else 0.0)

        trace_ops.score_trace("error_occurred", 0.0)
        return metrics

    except Exception:
        trace_ops.score_trace("error_occurred", 1.0)
        raise
    finally:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        trace_ops.score_trace("total_latency_ms", ms)
        trace_ops.score_trace("total_cost_usd", costs.get("total_cost", 0))


@observe(name="generate_financial_summary")
def generate_financial_summary(question: str, *, user_id="anonymous", session_id=None):
    t0 = time.perf_counter()
    costs = {}

    try:
        chunks = _retrieve(question)
        summary, costs = _generate(
            extractor, extractor.generate_summary, chunks,
            span_name="summary_generation",
        )
        trace_ops.score_trace("error_occurred", 0.0)
        return summary

    except Exception:
        trace_ops.score_trace("error_occurred", 1.0)
        raise
    finally:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        trace_ops.score_trace("total_latency_ms", ms)
        trace_ops.score_trace("total_cost_usd", costs.get("total_cost", 0))


@observe(name="upload_document")
def upload_document(uploaded_file, *, user_id="anonymous"):
    t0 = time.perf_counter()

    try:
        data_dir = Path("./data")
        data_dir.mkdir(exist_ok=True)
        file_path = data_dir / uploaded_file.name

        with trace_ops.span("save_file") as s:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            s.update(output={"path": str(file_path)})

        with trace_ops.span("ingest_to_chroma") as s:
            result = ingestion.ingest_file(str(file_path))
            s.update(output=result)

        with trace_ops.span("refresh_retriever") as s:
            retriever.refresh()
            s.update(output={"companies": list(retriever.dense_retriever.known_companies)})

        trace_ops.score_trace("error_occurred", 0.0)
        return result

    except Exception:
        trace_ops.score_trace("error_occurred", 1.0)
        raise
    finally:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        trace_ops.score_trace("total_latency_ms", ms)


def get_available_companies():
    return retriever.dense_retriever.known_companies
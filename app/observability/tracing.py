"""
tracing.py — Langfuse v4.14 compatible
---------------------------------------
Uses the v4 context-based API:
  • start_as_current_observation()  → context-managed spans/generations
  • score_current_trace()           → trace-level scores
  • update()                        → set output, usage, cost on spans

Child observations auto-nest inside parent observations.
"""

import time
from contextlib import contextmanager
from typing import Any, Optional

from app.observability.langfuse_config import langfuse, is_enabled
from app.observability.pricing import compute_cost


# ══════════════════════════════════════════════════════════════════
#  Span (retrieval, guardrails, file I/O)
# ══════════════════════════════════════════════════════════════════

@contextmanager
def span(name: str, *, input: Any = None):
    """Wrap a pipeline step in a Langfuse span.

    Usage:
        with span("retrieval", input={"query": q}) as s:
            results = retriever.retrieve(q)
            s.update(output={"n_chunks": 5})
    """
    if not is_enabled():
        yield _DummySpan()
        return

    t0 = time.perf_counter()
    with langfuse.start_as_current_observation(
        name=name, as_type="span", input=input
    ) as obs:
        try:
            yield obs
        except Exception as exc:
            obs.update(
                level="ERROR",
                status_message=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            ms = round((time.perf_counter() - t0) * 1000, 1)
            obs.update(metadata={"latency_ms": ms})


# ══════════════════════════════════════════════════════════════════
#  Generation (LLM calls — with token counts + cost)
# ══════════════════════════════════════════════════════════════════

@contextmanager
def generation(name: str, *, model: str = "amazon.nova-micro-v1:0",
               input: Any = None):
    """Wrap an LLM call in a Langfuse generation.

    Usage:
        with generation("answer_llm", model=m) as g:
            response = llm.invoke(prompt)
            g.update(output=response.content)
    """
    if not is_enabled():
        yield _DummySpan()
        return

    t0 = time.perf_counter()
    with langfuse.start_as_current_observation(
        name=name, as_type="generation", model=model, input=input
    ) as obs:
        try:
            yield obs
        except Exception as exc:
            obs.update(
                level="ERROR",
                status_message=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            ms = round((time.perf_counter() - t0) * 1000, 1)
            obs.update(metadata={"latency_ms": ms})


# ══════════════════════════════════════════════════════════════════
#  Attach token usage + cost to a generation
# ══════════════════════════════════════════════════════════════════

def attach_usage(llm_obj, obs) -> dict:
    """Read _last_usage from generator/extractor, write tokens + cost
    onto the Langfuse generation observation. Returns cost dict."""
    usage = getattr(llm_obj, "_last_usage", {})
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)
    model = getattr(llm_obj, "model_id", "amazon.nova-micro-v1:0")
    costs = compute_cost(model, inp, out)

    if is_enabled() and not isinstance(obs, _DummySpan):
        obs.update(
            usage_details={"input": inp, "output": out, "total": inp + out},
            cost_details={
                "input": costs["input_cost"],
                "output": costs["output_cost"],
                "total": costs["total_cost"],
            },
        )
    return costs


# ══════════════════════════════════════════════════════════════════
#  Trace-level scores
# ══════════════════════════════════════════════════════════════════

def score_trace(name: str, value: float, *, comment: Optional[str] = None):
    """Attach a score to the current trace (cost, latency, error, etc.)."""
    if not is_enabled():
        return
    try:
        langfuse.score_current_trace(name=name, value=value, comment=comment)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════
#  Dummy for when Langfuse is disabled
# ══════════════════════════════════════════════════════════════════

class _DummySpan:
    """No-op stand-in so callers can still call .update() safely."""
    def update(self, **kwargs):
        pass
    def score(self, **kwargs):
        pass
    def score_trace(self, **kwargs):
        pass
    def end(self, **kwargs):
        pass
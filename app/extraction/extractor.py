"""
extractor.py  (with token usage capture)
-----------------------------------------
Drop-in replacement.  _last_usage is populated after every LLM call.
Return types are unchanged — Streamlit untouched.

FIX: extract_metrics() and generate_summary() now accept an optional
     `query` param so the LLM knows which company/period to focus on,
     and an optional `metadatas` param to label chunks by source.
     Both are backward-compatible — existing callers still work.
"""

import json
import os
import re

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse


class FinancialExtractor:

    def __init__(self):
        load_dotenv(".env")

        self.model_id = "amazon.nova-micro-v1:0"

        self.llm = ChatBedrockConverse(
            model=self.model_id,
            region_name=os.getenv("AWS_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

        self._last_usage: dict = {"input_tokens": 0, "output_tokens": 0}

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _build_context(
        chunks: list[str], metadatas: list[dict] | None = None
    ) -> str:
        """Join chunks, optionally prefixing each with its metadata
        tag so the LLM can distinguish companies and document types."""
        if metadatas:
            parts = []
            for chunk, meta in zip(chunks, metadatas):
                company = meta.get("company", "unknown").title()
                doc_type = meta.get("document_type", "document")
                source = meta.get("source_file", "")
                label = f"[Company: {company} | Type: {doc_type} | Source: {source}]"
                parts.append(f"{label}\n{chunk}")
            return "\n\n---\n\n".join(parts)
        return "\n\n---\n\n".join(chunks)

    @staticmethod
    def _parse_json(raw: str):
        cleaned = re.sub(
            r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE
        ).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return raw

    @staticmethod
    def _extract_usage(response) -> dict:
        out = {"input_tokens": 0, "output_tokens": 0}
        try:
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                um = response.usage_metadata
                out["input_tokens"]  = um.get("input_tokens", 0)
                out["output_tokens"] = um.get("output_tokens", 0)
                return out
            if hasattr(response, "response_metadata"):
                rm = response.response_metadata.get("usage", {})
                out["input_tokens"]  = rm.get("inputTokens",
                                               rm.get("input_tokens", 0))
                out["output_tokens"] = rm.get("outputTokens",
                                               rm.get("output_tokens", 0))
                return out
        except Exception:
            pass
        return out

    # ── Metrics extraction ──────────────────────────────────────

    def extract_metrics(
        self,
        retrieved_chunks: list[str],
        query: str = "",
        metadatas: list[dict] | None = None,
    ):
        context = self._build_context(retrieved_chunks, metadatas)

        # If we know what the user asked, tell the LLM explicitly
        query_instruction = ""
        if query:
            query_instruction = f"""
The user's original question was: "{query}"
Extract metrics ONLY for the company and reporting period relevant to
this question. Ignore data from other companies or other periods.
"""

        prompt = f"""You are a Financial Analyst.

Extract the following financial metrics from the context below.
{query_instruction}
Rules:

1. Return ONLY valid JSON. No markdown fences, no commentary.

2. If a metric is not explicitly stated in the context, return
   "Not Available" — never estimate, calculate, or infer.

3. CRITICAL — distinguish REPORTED RESULTS from FORWARD GUIDANCE:
   - Reported results: "revenue was $X", "net income increased to $X"
   - Forward guidance: "we expect revenue to be $X-$Y", "outlook"
   For each metric, extract the REPORTED (actual) value, not guidance.
   If only guidance is available for a metric, prefix the value with
   "Guidance: " (e.g. "Guidance: $58-61 billion").

4. Each chunk is tagged with [Company: ...]. Only use chunks matching
   the company asked about. Ignore other companies' data entirely.

5. Copy numeric values exactly as written, including units.

Required Metrics:
- Revenue
- Net Income
- Operating Income
- EPS
- Operating Margin
- Gross Margin
- Free Cash Flow
- Capital Expenditure

Context:
{context}

JSON:
"""
        response = self.llm.invoke(prompt)
        self._last_usage = self._extract_usage(response)
        return self._parse_json(response.content)

    # ── Summary generation ──────────────────────────────────────

    def generate_summary(
        self,
        retrieved_chunks: list[str],
        query: str = "",
        metadatas: list[dict] | None = None,
    ):
        context = self._build_context(retrieved_chunks, metadatas)

        query_instruction = ""
        if query:
            query_instruction = f"""
The user's original question was: "{query}"
Focus the summary on the company and period relevant to this question.
"""

        prompt = f"""You are a Financial Analyst.

Produce a concise executive summary from the context below.
{query_instruction}
Rules:

1. Each chunk is tagged with [Company: ...]. Only summarise the
   company the user asked about. If no specific company is indicated,
   summarise whichever company appears most in the context.

2. Clearly separate REPORTED RESULTS from FORWARD GUIDANCE.
   - When stating actuals: "Q1 2026 revenue was $56.3 billion"
   - When stating guidance: "The company expects Q2 revenue of $58-61B"
   Never present a guidance range as if it were an actual result.

3. Only use figures explicitly present in the context — never
   fabricate, round, or approximate.

4. Limit the summary to about 150 words.

Include:
- Key reported financial metrics (revenue, income, margins, EPS)
- Notable year-over-year or quarter-over-quarter changes
- Business highlights
- Forward guidance (clearly labelled as such)

Context:
{context}

Summary:
"""
        response = self.llm.invoke(prompt)
        self._last_usage = self._extract_usage(response)
        return response.content
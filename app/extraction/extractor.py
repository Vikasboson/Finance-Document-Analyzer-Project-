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

5. CRITICAL — distinguish TOTAL COMPANY figures from SEGMENT-LEVEL
   breakdowns. Earnings documents report overall company results first
   (e.g. "This quarter, revenue was $X billion") then segment results
   (e.g. "Revenue from [Segment] was $Y billion" or "Now to [Segment].
   Revenue was $Y billion"). Always extract TOTAL COMPANY figures, NOT
   segment figures, unless the user specifically asks about a segment.

6. MANDATORY — every numeric value MUST include its unit:
   - Dollar amounts: "$77.7 billion" or "$181,519 million" (keep the
     original scale from the document — do NOT strip "billion"/"million")
   - Percentages: "49%" or "69%"
   - Per-share: "$4.13 per share"
   NEVER return a bare number like "181519" — always include the unit.

7. MANDATORY — always include "Company" and "ReportingPeriod" keys.

Return EXACTLY this JSON structure (no extra keys, no missing keys):

Example of correct output:
{{
  "Company": "Microsoft",
  "ReportingPeriod": "Q1 FY26",
  "Revenue": "$77.7 billion",
  "NetIncome": "$34.2 billion",
  "OperatingIncome": "$38.1 billion",
  "EPS": "$4.13 per share",
  "OperatingMargin": "49%",
  "GrossMargin": "69%",
  "FreeCashFlow": "$25.7 billion",
  "CapitalExpenditure": "$34.9 billion"
}}

Context:
{context}

JSON:
"""
        response = self.llm.invoke(prompt)
        self._last_usage = self._extract_usage(response)
        return self._parse_json(response.content)
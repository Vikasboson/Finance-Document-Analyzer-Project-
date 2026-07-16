"""
extractor.py  (with token usage capture)
-----------------------------------------
Drop-in replacement.  _last_usage is populated after every LLM call.
Return types are unchanged — Streamlit untouched.
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

    def extract_metrics(self, retrieved_chunks: list[str]):
        context = "\n\n".join(retrieved_chunks)
        prompt = f"""
You are a Financial Analyst.

Extract the following financial metrics from the context below.

Rules:
- Return ONLY valid JSON.  No markdown fences, no commentary.
- If a metric is missing, return "Not Available" — never estimate.
- Use the most recent period shown unless stated otherwise.
- Copy numeric values exactly as written, including units.

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

    def generate_summary(self, retrieved_chunks: list[str]):
        context = "\n\n".join(retrieved_chunks)
        prompt = f"""
You are a Financial Analyst.

Produce a concise executive summary from the context below.

Include:
- Company's financial performance
- Important financial metrics
- Business highlights
- Future guidance (if available)

Only use figures explicitly present in the context.
Limit the summary to about 150 words.

Context:
{context}

Summary:
"""
        response = self.llm.invoke(prompt)
        self._last_usage = self._extract_usage(response)
        return response.content
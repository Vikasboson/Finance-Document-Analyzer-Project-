"""
generator.py  (with token usage capture)
-----------------------------------------
Drop-in replacement.  After each invoke(), self._last_usage stores
{"input_tokens": N, "output_tokens": N} for the observability layer.

generate_answer() still returns a plain string — Streamlit untouched.
"""

import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse


class AnswerGenerator:

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

    # ── Prompt ──────────────────────────────────────────────────

    def build_prompt(
        self,
        query: str,
        retrieved_chunks: list[str],
        metadatas: list[dict] | None = None,
    ) -> str:
        """Build the LLM prompt.

        When *metadatas* is provided (one dict per chunk), each chunk
        is prefixed with its company and document-type tag so the model
        can tell sources apart even when multiple companies appear.
        """
        if metadatas:
            labelled = []
            for chunk, meta in zip(retrieved_chunks, metadatas):
                company = meta.get("company", "unknown").title()
                doc_type = meta.get("document_type", "document")
                source = meta.get("source_file", "")
                label = f"[Company: {company} | Type: {doc_type} | Source: {source}]"
                labelled.append(f"{label}\n{chunk}")
            context = "\n\n---\n\n".join(labelled)
        else:
            context = "\n\n---\n\n".join(retrieved_chunks)

        return f"""You are a Financial Document Analysis Assistant.

Instructions:

1. Read every retrieved chunk carefully. Each chunk is tagged with its
   company name and document type — ONLY use chunks that match the
   company asked about in the question. Ignore chunks from other
   companies entirely.

2. CRITICAL — distinguish between REPORTED RESULTS and FORWARD GUIDANCE:
   - Reported results use past-tense language: "revenue was", "net
     income increased to", "we recorded", "totaled".
   - Forward guidance uses future-tense or conditional language:
     "we expect", "is expected to be in the range of", "we anticipate",
     "outlook", "guidance", "forecast".
   NEVER substitute a guidance figure for a reported result or vice
   versa. If the question asks for an actual reported number, only
   provide the reported figure. If the question asks about guidance
   or outlook, only provide the forward-looking figure.

3. Financial documents often show multiple periods side by side
   (e.g. Q1 2026 vs Q1 2025, or current year vs prior year).
   Identify exactly which period the question asks about and only
   report that period's value. Do not average, combine, or mix
   values across different periods.

4. Copy numerical values exactly as written in the source text,
   including sign, currency symbol, and units (billions, millions,
   per share, etc.). Do not round or convert.

5. If the answer is present but phrased differently (e.g. "operating
   margin" expressed as a percentage of revenue), you may derive the
   equivalent — but show your working.

6. If the exact figure is not present anywhere in the context, respond:
   "Not available in the provided context." Never fabricate a number.

7. Structure your answer as:
   - A direct, concise answer to the question
   - A brief citation: which metric, which period, which document

Context:
{context}

Question:
{query}

Answer:
"""

    # ── Generate ────────────────────────────────────────────────

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: list[str],
        metadatas: list[dict] | None = None,
    ) -> str:
        prompt = self.build_prompt(query, retrieved_chunks, metadatas)
        response = self.llm.invoke(prompt)
        self._last_usage = self._extract_usage(response)
        return response.content

    # ── Token extraction (works across LangChain versions) ──────

    @staticmethod
    def _extract_usage(response) -> dict:
        """Pull token counts from the LangChain AIMessage.

        Checks two locations:
          1. response.usage_metadata   (LangChain ≥ 0.2)
          2. response.response_metadata["usage"]  (raw Bedrock)
        """
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


if __name__ == "__main__":
    from retriever import DocumentRetriever

    retriever = DocumentRetriever()
    question = "For META what is the expected range of total revenue in Q2 2026?"
    results = retriever.retrieve(question)
    chunks = results["documents"][0]

    gen = AnswerGenerator()
    answer = gen.generate_answer(query=question, retrieved_chunks=chunks)
    print("Answer:", answer)
    print("Tokens:", gen._last_usage)
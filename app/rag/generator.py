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

    def build_prompt(self, query: str, retrieved_chunks: list[str]) -> str:
        context = "\n\n".join(retrieved_chunks)
        return f"""
You are a Financial Document Analysis Assistant.

Instructions:

1. Read every retrieved chunk carefully.

2. Multiple chunks may belong to different companies — only use chunks
   whose company matches the one asked about in the question.  If the
   context does not clearly relate to the company or period asked
   about, say so instead of guessing.

3. Financial statements often show multiple periods side by side
   (e.g. current year vs. prior year).  Identify exactly which period
   the question is asking about and only use that column's value.
   Do not average, combine, or mix values across periods.

4. Copy numerical values exactly as written, including sign, currency
   symbol, and units (e.g. millions).

5. If the answer exists but is phrased differently, infer the
   equivalent meaning.

6. If the exact figure is not present anywhere in the context, respond
   with "Not available in the provided context" — never fabricate.

7. Provide a concise, direct answer and briefly cite which line item
   and period it came from.

Context:
{context}

Question:
{query}

Answer:
"""

    # ── Generate ────────────────────────────────────────────────

    def generate_answer(self, query: str, retrieved_chunks: list[str]) -> str:
        prompt = self.build_prompt(query, retrieved_chunks)
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
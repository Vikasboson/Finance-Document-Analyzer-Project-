import numpy as np
from rank_bm25 import BM25Okapi

from app.rag.retriever import DocumentRetriever


class HybridRetriever:

    def __init__(self, chroma_client=None, collection_name: str = "documents"):
        # Pass the shared client through so everyone uses the same
        # ChromaDB connection.
        self.dense_retriever = DocumentRetriever(
            chroma_client=chroma_client,
            collection_name=collection_name,
        )
        self.collection = self.dense_retriever.collection
        self._load_corpus()

    def _load_corpus(self):
        """Load all documents from Chroma and (re)build the BM25 index."""
        all_data = self.collection.get(include=["documents", "metadatas"])
        self.documents = all_data["documents"]
        self.metadatas = all_data["metadatas"]

        print("Collection count:", self.collection.count())

        if len(self.documents) == 0:
            print("WARNING: No documents in collection. BM25 index empty.")
            self.bm25 = None
            return

        tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def refresh(self):
        """Refresh both the dense retriever's known-company list and
        the BM25 index so newly uploaded documents are searchable."""
        self.dense_retriever.refresh()
        self._load_corpus()

    def bm25_search(self, query: str, top_k: int = 3, company: str = None):
        if self.bm25 is None:
            return []

        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1]

        results = []
        for idx in top_indices:
            # Apply the same company filter as the dense retriever
            # so BM25 results don't contaminate with other companies.
            if company and self.metadatas[idx].get("company") != company:
                continue

            results.append({
                "document": self.documents[idx],
                "metadata": self.metadatas[idx],
                "score": scores[idx],
            })

            if len(results) >= top_k:
                break

        return results

    def _detect_company(self, query: str) -> str | None:
        """Mirror the dense retriever's company detection so BM25
        uses the exact same filter."""
        query_lower = query.lower()
        for company in self.dense_retriever.known_companies:
            if company in query:
                return company
        # known = self.dense_retriever.known_companies
        # matches = [c for c in known if c and c in query_lower]
        # if matches:
        #     return max(matches, key=len)
        return None

    def retrieve(self, query: str, top_k: int = 5):
        # Detect company once, use for both searches
        company = self._detect_company(query)

        # Dense Search (applies company filter internally)
        dense_results = self.dense_retriever.retrieve(query=query, top_k=top_k)
        dense_documents = dense_results["documents"][0]
        dense_metadatas = dense_results["metadatas"][0]

        # BM25 Search (now ALSO filters by company)
        bm25_results = self.bm25_search(query=query, top_k=top_k, company=company)

        merged_documents = []
        merged_metadatas = []
        seen = set()

        for doc, meta in zip(dense_documents, dense_metadatas):
            if doc not in seen:
                seen.add(doc)
                merged_documents.append(doc)
                merged_metadatas.append(meta)

        for result in bm25_results:
            if result["document"] not in seen:
                seen.add(result["document"])
                merged_documents.append(result["document"])
                merged_metadatas.append(result["metadata"])

        return {
            "documents": [merged_documents],
            "metadatas": [merged_metadatas],
        }

    def print_results(self, results):
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
            print("=" * 80)
            print(f"Chunk {i}")
            print(meta)
            print()
            print(doc)
            print()


if __name__ == "__main__":
    retriever = HybridRetriever()
    query = "For META what is the expected range of total revenue in second quarter 2026 in billions?"
    results = retriever.retrieve(query=query, top_k=5)
    retriever.print_results(results)
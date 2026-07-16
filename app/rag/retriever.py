import chromadb
from sentence_transformers import SentenceTransformer


class DocumentRetriever:
    def __init__(
        self,
        chroma_client=None,
        chroma_path: str = "./chroma_datab",
        collection_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        # Accept an external client so api.py can share one instance.
        if chroma_client is not None:
            self.client = chroma_client
        else:
            self.client = chromadb.PersistentClient(path=chroma_path)

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

        self.embedding_model = SentenceTransformer(embedding_model)
        self._load_known_companies()

    def _load_known_companies(self):
        """Build the set of companies actually present in the collection
        dynamically — any company that has been ingested becomes
        automatically detectable in queries."""
        data = self.collection.get(include=["metadatas"])
        self.known_companies = {
            m.get("company")
            for m in data["metadatas"]
            if m and m.get("company")
        }
        print(f"Known companies in collection: {self.known_companies}")

    def refresh(self):
        """Reload known companies after new documents are ingested."""
        self._load_known_companies()

    def retrieve(self, query: str, top_k: int = 3):
        query_embedding = self.embedding_model.encode(
            query, convert_to_numpy=True
        ).tolist()

        query_lower = query.lower()

        # Dynamic company match — longest match wins.
        company = None
        matches = [c for c in self.known_companies if c and c in query_lower]
        if matches:
            company = max(matches, key=len)

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }

        if company:
            query_kwargs["where"] = {"company": company}

        results = self.collection.query(**query_kwargs)

        # Fallback: if filtered query returned nothing, retry without filter.
        if company and not results.get("documents", [[]])[0]:
            query_kwargs.pop("where", None)
            results = self.collection.query(**query_kwargs)

        return results

    def print_results(self, results):
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not documents:
            print("No relevant documents found.")
            return

        for i, (doc, metadata) in enumerate(zip(documents, metadatas), start=1):
            print(f"\n{'=' * 80}")
            print(f"Chunk {i}")
            if metadata:
                print(f"Metadata : {metadata}")
            print("\nContent:")
            print(doc)
        print(f"\n{'=' * 80}")


if __name__ == "__main__":
    retriever = DocumentRetriever()
    query = "For META what is the expected range of total revenue in second quarter 2026 in billions?"
    results = retriever.retrieve(query=query, top_k=3)
    retriever.print_results(results)
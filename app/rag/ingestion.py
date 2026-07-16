import hashlib
import re
from collections import Counter
from pathlib import Path

import chromadb
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class DocumentIngestion:
    def __init__(
        self,
        data_path: str = "./data",
        chroma_client=None,
        chroma_path: str = "./chroma_datab",
        collection_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.data_path = Path(data_path)

        # Accept an external client so api.py can share one instance
        # across ingestion and retrieval. Falls back to creating its
        # own if none is provided (e.g. when run standalone).
        if chroma_client is not None:
            self.client = chroma_client
        else:
            self.client = chromadb.PersistentClient(path=chroma_path)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.embedding_model = SentenceTransformer(embedding_model)

    # ------------------------------------------------------------------
    # Dynamic company / document-type detection
    # ------------------------------------------------------------------

    _LEGAL_SUFFIXES = (
        r"Inc\.|Incorporated|Corporation|Corp\.|Company|LLC|plc|Ltd\.|Limited|L\.P\."
    )

    # Regex design:
    #   - NO comma in the character class so the comma before "Inc."
    #     acts as a separator instead of being consumed.
    #   - Negative lookahead prevents suffix words (Corporation, etc.)
    #     from being captured as part of the company name.
    #   - (?=\s|$) at the end instead of \b, because \b fails after
    #     dots (e.g. "Inc." ends with a non-word char).
    _COMPANY_PATTERN = re.compile(
        r"([A-Z][A-Za-z&\.\-]*"
        r"(?:\s+(?!(?:" + _LEGAL_SUFFIXES + r")(?:\s|$))[A-Z][A-Za-z&\.\-]*){0,4})"
        r"[\s,]+(?:" + _LEGAL_SUFFIXES + r")(?=\s|$)"
    )

    _GENERIC_FILENAME_TOKENS = {
        "earnings", "call", "calls", "transcript", "transcripts",
        "report", "reports", "statement", "statements", "consolidated",
        "financial", "financials", "quarterly", "annual", "results",
        "release", "10q", "10-q", "10k", "10-k", "q1", "q2", "q3", "q4",
    }

    def _clean_fallback_name(self, stem: str) -> str:
        """Strip generic document-type words from the filename so
        'META earnings' becomes 'meta', not 'meta earnings'."""
        tokens = re.split(r"[\s_\-]+", stem.lower())
        filtered = [
            t for t in tokens
            if t not in self._GENERIC_FILENAME_TOKENS
            and not re.fullmatch(r"(fy)?\d{2,4}", t)
        ]
        return " ".join(filtered) if filtered else stem.lower()

    @staticmethod
    def _normalize_company(raw: str) -> str:
        """Clean a raw regex capture into a usable company tag."""
        name = raw.strip().lower()
        name = re.sub(r"\.com|\.net|\.org|\.io", "", name)
        name = name.replace(",", "").replace(".", " ").strip()
        name = re.sub(r"\s+", " ", name)
        return name

    def _detect_company_name(self, documents, fallback_name: str) -> str:
        """Detect the company name from document text using FREQUENCY.

        The company's own name (e.g. "Amazon.com, Inc.") appears far
        more often in its own filing than any other company mentioned
        in passing, so we find ALL matches and pick the most common.
        """
        combined_text = " ".join(doc.page_content for doc in documents)
        matches = self._COMPANY_PATTERN.findall(combined_text)

        if matches:
            counts = Counter(self._normalize_company(m) for m in matches)
            winner, freq = counts.most_common(1)[0]
            print(f"    company candidates: {dict(counts)}")
            if winner:
                return winner

        return self._clean_fallback_name(fallback_name)

    def _detect_document_type(self, filename: str) -> str:
        name = filename.lower()
        if any(k in name for k in ["earnings", "call", "transcript"]):
            return "earnings_call"
        if "10-q" in name or "10q" in name:
            return "10-Q"
        if "10-k" in name or "10k" in name:
            return "10-K"
        if any(k in name for k in [
            "consolidated", "financial_statement", "balance_sheet",
            "income_statement", "cash_flow",
        ]):
            return "financial_statements"
        return "financial_document"

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_documents(self):
        """Load all supported documents from the data directory."""
        documents = []
        for file in self.data_path.iterdir():
            if file.suffix.lower() == ".pdf":
                print(f"Loading PDF: {file.name}")
                documents.extend(
                    self._load_and_tag(file, PyPDFLoader(str(file)).load())
                )
            elif file.suffix.lower() == ".txt":
                print(f"Loading TXT: {file.name}")
                documents.extend(
                    self._load_and_tag(file, TextLoader(str(file)).load())
                )
        print(f"\nLoaded {len(documents)} document pages.")
        return documents

    def load_document(self, file_path: str):
        """Load a single supported document."""
        file = Path(file_path)
        if file.suffix.lower() == ".pdf":
            print(f"Loading PDF: {file.name}")
            docs = PyPDFLoader(str(file)).load()
        elif file.suffix.lower() == ".txt":
            print(f"Loading TXT: {file.name}")
            docs = TextLoader(str(file)).load()
        else:
            raise ValueError(f"Unsupported file type: {file.suffix}")
        return self._load_and_tag(file, docs)

    def _load_and_tag(self, file: Path, docs):
        """Attach company / document_type metadata to every page
        BEFORE chunking so the tags survive the text splitter."""
        company = self._detect_company_name(docs, fallback_name=file.stem)
        document_type = self._detect_document_type(file.name)

        for doc in docs:
            doc.metadata["company"] = company
            doc.metadata["document_type"] = document_type
            doc.metadata["topic"] = document_type.replace("_", " ").title()
            doc.metadata["source_file"] = file.name

        print(f"  -> detected company='{company}', document_type='{document_type}'")
        return docs

    # ------------------------------------------------------------------
    # Chunking + Embedding + Storage
    # ------------------------------------------------------------------

    def chunk_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
        )
        chunks = splitter.split_documents(documents)
        chunks = [c for c in chunks if len(c.page_content.strip()) > 20]
        print(f"Created {len(chunks)} chunks.")
        return chunks

    def create_embeddings(self, chunks):
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        print("Embeddings generated.")
        return texts, embeddings, chunks

    def store_in_chroma(self, texts, embeddings, chunks):
        """Store with deterministic hash-based IDs and upsert so
        re-ingestion is idempotent and uploads never collide."""
        ids = []
        for text, chunk in zip(texts, chunks):
            source = chunk.metadata.get(
                "source_file", chunk.metadata.get("source", "")
            )
            page = chunk.metadata.get("page", "")
            raw_id = f"{source}|{page}|{text}"
            ids.append(hashlib.md5(raw_id.encode("utf-8")).hexdigest())

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[chunk.metadata for chunk in chunks],
        )
        print("Documents stored in ChromaDB successfully.")

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def ingest_file(self, file_path: str):
        """Ingest a single uploaded document."""
        documents = self.load_document(file_path)
        chunks = self.chunk_documents(documents)
        texts, embeddings, chunks = self.create_embeddings(chunks)
        self.store_in_chroma(texts, embeddings, chunks)
        return {
            "status": "success",
            "file_name": Path(file_path).name,
            "chunks": len(chunks),
        }

    ## the deleted files from data##
    def remove_orphaned_documents(self):
        """Delete chunks whose source file no longer exists in ./data.

        Runs at the start of every run() call so that a file deleted from
        disk between restarts also disappears from ChromaDB — otherwise
        its chunks (and its company in known_companies) linger forever,
        since ingestion only ever upserts and never previously deleted.
        """
        if not self.data_path.exists():
            existing_files = set()
        else:
            existing_files = {f.name for f in self.data_path.iterdir()}

        all_data = self.collection.get(include=["metadatas"])
        orphaned_sources = {
            m.get("source_file") for m in all_data["metadatas"]
            if m.get("source_file") and m.get("source_file") not in existing_files
        }

        for source in orphaned_sources:
            self.collection.delete(where={"source_file": source})
            print(f"Removed orphaned chunks for deleted file: {source}")

        return orphaned_sources

    def run(self):
        """Run the complete ingestion pipeline over the data directory."""
        self.remove_orphaned_documents() # for deleting the files not present in data from chroma database

        if not self.data_path.exists():
            print(f"Data directory {self.data_path} not found. Skipping ingestion.")
            return

        documents = self.load_documents()
        if not documents:
            print("No documents found in data directory. Skipping.")
            return

        chunks = self.chunk_documents(documents)
        texts, embeddings, chunks = self.create_embeddings(chunks)
        self.store_in_chroma(texts, embeddings, chunks)
        print("\nIngestion pipeline completed successfully.")


if __name__ == "__main__":
    ingestion = DocumentIngestion()
    ingestion.run()
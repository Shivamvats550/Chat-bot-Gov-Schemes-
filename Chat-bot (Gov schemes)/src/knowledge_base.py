import csv
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Custom lightweight class to represent a single document structure in our RAG system
class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}


# Define paths relative to the project directory structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "gov_schemes.csv")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
INDEX_PATH = os.path.join(VECTOR_DB_DIR, "index.faiss")
DOCS_PATH = os.path.join(VECTOR_DB_DIR, "documents.pkl")


class GovernmentSchemesDB:
    """
    A clean vector database wrapper using FAISS and Hugging Face Sentence Transformers.
    It loads government schemes from a CSV file, generates semantic vector embeddings,
    saves the FAISS index to disk for fast caching, and runs semantic search queries.
    """

    def __init__(self, csv_path: str = CSV_PATH):
        self.csv_path = csv_path
        self.schemes = []      # List of raw scheme rows from the CSV file
        self.documents = []    # List of Document objects
        self.index = None      # FAISS index for quick vector search
        self.model = None      # Hugging Face SentenceTransformer model

        # 1. Load CSV raw rows into memory
        self._load_csv()

        # 2. Load the SentenceTransformer embedding model (runs locally on CPU)
        # 'all-MiniLM-L6-v2' is a fast, highly-optimized model that outputs 384-dimensional embeddings.
        print("Loading Hugging Face SentenceTransformer model ('all-MiniLM-L6-v2')...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # 3. Load cached FAISS index or build it if not found
        self._init_vector_db()

    def _load_csv(self) -> None:
        """Reads scheme data from the CSV file, cleans up text fields, and stores in self.schemes."""
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")

        with open(self.csv_path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scheme = {
                    "scheme_name": (row.get("scheme_name") or "").strip(),
                    "details": (row.get("details") or "").strip(),
                    "benefits": (row.get("benefits") or "").strip(),
                    "eligibility": (row.get("eligibility") or "").strip(),
                    "application": (row.get("application") or "").strip(),
                    "documents": (row.get("documents") or "").strip(),
                    "schemeCategory": (row.get("schemeCategory") or "").strip(),
                }
                # Ensure we only include entries with a valid scheme name
                if scheme["scheme_name"]:
                    self.schemes.append(scheme)
        print(f"Loaded {len(self.schemes)} schemes from CSV.")

    def _init_vector_db(self) -> None:
        """Loads cached index from disk if available; otherwise, generates embeddings and saves the cache."""
        if os.path.isfile(INDEX_PATH) and os.path.isfile(DOCS_PATH):
            print("Cached FAISS index found. Loading from disk...")
            # Read the serialized FAISS vector index
            self.index = faiss.read_index(INDEX_PATH)
            
            # Load stored document page contents and metadata
            with open(DOCS_PATH, "rb") as f:
                docs_data = pickle.load(f)
            self.documents = [Document(d["page_content"], d["metadata"]) for d in docs_data]
            print(f"FAISS index loaded successfully with {len(self.documents)} documents.")
        else:
            print("Cached FAISS index not found. Building new vector database...")
            
            # Convert raw CSV scheme dictionaries to structured Document objects
            self.documents = []
            for s in self.schemes:
                fields = [
                    f"Scheme Name: {s['scheme_name']}",
                    f"Category: {s['schemeCategory']}",
                    f"Details: {s['details']}",
                    f"Benefits: {s['benefits']}",
                    f"Eligibility: {s['eligibility']}",
                    f"Application Process: {s['application']}",
                    f"Required Documents: {s['documents']}"
                ]
                content = "\n".join(field for field in fields if field.strip())
                
                # Sanitize content (removes surrogates that crash Hugging Face tokenizers)
                content = "".join(ch for ch in content if not (0xD800 <= ord(ch) <= 0xDFFF))
                
                self.documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "scheme_name": s["scheme_name"],
                            "category": s["schemeCategory"]
                        }
                    )
                )

            # Generate vector representations for all documents using SentenceTransformer
            texts = [doc.page_content for doc in self.documents]
            print(f"Generating embeddings for {len(texts)} documents. This may take 1-2 minutes...")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            embeddings = np.array(embeddings).astype("float32")

            # Create a FAISS Flat Inner Product index for Cosine Similarity search
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings)  # Normalize embeddings to enable Inner Product as Cosine Similarity
            index.add(embeddings)
            self.index = index

            # Save the generated FAISS index and documents metadata cache to disk
            os.makedirs(VECTOR_DB_DIR, exist_ok=True)
            faiss.write_index(self.index, INDEX_PATH)
            
            docs_data = [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in self.documents
            ]
            with open(DOCS_PATH, "wb") as f:
                pickle.dump(docs_data, f)
            print("Vector database built and saved to disk.")

    def search(self, query: str, k: int = 4) -> list[Document]:
        """
        Calculates the embedding of the query and searches the FAISS index 
        for the top-k most semantically similar documents.
        """
        if not query or not query.strip() or self.index is None:
            return []

        # Encode and normalize query vector
        query_vector = self.model.encode([query])
        query_vector = np.array(query_vector).astype("float32")
        faiss.normalize_L2(query_vector)

        # Query the FAISS index for top-k nearest neighbors
        distances, indices = self.index.search(query_vector, k)

        # Retrieve documents corresponding to returned indices
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.documents):
                results.append(self.documents[idx])
        return results

import shutil
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from embedding.embedding_dataclass import EmbeddingResult
from vectorstore.fake_vector_store import FakeVectorStore
from vectorstore.chroma_vector_store import ChromaVectorStore
from vectorstore.faiss_vector_store import FaissVectorStore

# ============================================================
# clear() - every BaseVectorStore implementation must support wiping
# itself empty (used by build_index(..., fresh=True), see
# ingestion/build_index.py / retrieval/query_pipeline.py). Throwaway
# persist paths here - never touches the production chroma_data/
# faiss_data directories.
# ============================================================

SCRATCH_DIR = PROJECT_ROOT / "tests" / "_scratch_vector_store_clear"
if SCRATCH_DIR.exists():
    shutil.rmtree(SCRATCH_DIR)
SCRATCH_DIR.mkdir(parents=True)

embeddings = [
    EmbeddingResult(
        document=Document(page_content="Document one", metadata={"chunk_id": "chunk-001"}),
        vector=[0.1, 0.2, 0.3],
    ),
    EmbeddingResult(
        document=Document(page_content="Document two", metadata={"chunk_id": "chunk-002"}),
        vector=[0.4, 0.5, 0.6],
    ),
]

# ---------- FakeVectorStore ----------

fake_store = FakeVectorStore()
fake_store.add(embeddings)
assert len(fake_store.get_all_documents()) == 2
fake_store.clear()
assert len(fake_store.get_all_documents()) == 0
print("FakeVectorStore.clear() empties the store: PASS")

# Store remains usable after clear() - re-adding works.
fake_store.add(embeddings)
assert len(fake_store.get_all_documents()) == 2
print("FakeVectorStore is usable after clear(): PASS")

# ---------- ChromaVectorStore ----------

chroma_store = ChromaVectorStore(
    collection_name="test_clear_collection",
    persist_path=str(SCRATCH_DIR / "chroma_data"),
)
chroma_store.add(embeddings)
assert chroma_store.collection.count() == 2
chroma_store.clear()
assert chroma_store.collection.count() == 0
print("ChromaVectorStore.clear() empties the collection: PASS")

chroma_store.add(embeddings)
assert chroma_store.collection.count() == 2
print("ChromaVectorStore is usable after clear(): PASS")

# clear() on an already-empty collection is a safe no-op.
chroma_store.clear()
chroma_store.clear()
assert chroma_store.collection.count() == 0
print("ChromaVectorStore.clear() is safe to call repeatedly/on empty: PASS")

# ---------- FaissVectorStore ----------

faiss_store = FaissVectorStore(index_dir=str(SCRATCH_DIR / "faiss_data"))
faiss_store.add(embeddings)
assert len(faiss_store.get_all_documents()) == 2
faiss_store.clear()
assert len(faiss_store.get_all_documents()) == 0
assert faiss_store.index is None
assert not faiss_store.index_path.exists()
print("FaissVectorStore.clear() empties the index and removes persisted files: PASS")

faiss_store.add(embeddings)
assert len(faiss_store.get_all_documents()) == 2
print("FaissVectorStore is usable after clear(): PASS")

# clear() before anything was ever added (index still None) shouldn't error.
empty_faiss_store = FaissVectorStore(index_dir=str(SCRATCH_DIR / "faiss_data_never_used"))
empty_faiss_store.clear()
print("FaissVectorStore.clear() is safe to call before any add(): PASS")

try:
    shutil.rmtree(SCRATCH_DIR)
except PermissionError:
    # Windows keeps a file handle open on Chroma's sqlite/hnsw files
    # until the process exits - harmless leftover in a scratch dir,
    # not a real cleanup failure.
    pass

print("\nAll vector store clear() tests passed.")

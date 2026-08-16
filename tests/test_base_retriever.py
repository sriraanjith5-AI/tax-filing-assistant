import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from embedding.embedding_dataclass import EmbeddingResult
from vectorstore.fake_vector_store import FakeVectorStore
from retrieval.vector_retriever import VectorRetriever

embeddings = [
    EmbeddingResult(
        document=Document(
            page_content="Document one",
            metadata={"chunk_id": "chunk-001"}
        ),
        vector=[1.0, 0.0, 0.0]
    ),
    EmbeddingResult(
        document=Document(
            page_content="Document two",
            metadata={"chunk_id": "chunk-002"}
        ),
        vector=[0.0, 1.0, 0.0]
    ),
    EmbeddingResult(
        document=Document(
            page_content="Document three",
            metadata={"chunk_id": "chunk-003"}
        ),
        vector=[0.8, 0.6, 0.0]
    )
]

vector_store=FakeVectorStore()
vector_store.add(embeddings)

retriever = VectorRetriever(vector_store)

results = retriever.retrieve(
    query_vector=[1.0, 0.0, 0.0],
    top_k=2
)

assert len(results) == 2
assert results[0].document.metadata["chunk_id"] == "chunk-001"
assert results[1].document.metadata["chunk_id"] == "chunk-003"
assert results[0].score >= results[1].score
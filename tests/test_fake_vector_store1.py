import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from embedding.embedding_dataclass import EmbeddingResult
from vectorstore.fake_vector_store import FakeVectorStore

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

vector_store = FakeVectorStore()

response1 = vector_store.add(embeddings)

assert response1.total_stored_chunks == 3

print("First add:")
print(f"Total received: {response1.total_received_chunks}")
print(f"Successful stores: {response1.total_stored_chunks}")
print(f"Skipped duplicates: {response1.total_skipped_chunks}")
print(f"Failed stores: {response1.total_failed_chunks}")

query_vector = [1.0, 0.0, 0.0]

results = vector_store.search(
    query_vector=query_vector,
    top_k=2
)

assert len(results) == 2

assert results[0].document.metadata["chunk_id"] == "chunk-001"
assert results[1].document.metadata["chunk_id"] == "chunk-003"

assert results[0].score > results[1].score

print("\nSearch Results:")

for index, result in enumerate(results):

    print(
        f"Rank {index + 1} | "
        f"chunk_id={result.document.metadata['chunk_id']} | "
        f"score={result.score:.4f} | "
        f"content={result.document.page_content}"
    )
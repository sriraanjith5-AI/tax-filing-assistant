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
        vector=[0.1, 0.2, 0.3]
    ),
    EmbeddingResult(
        document=Document(
            page_content="Document two",
            metadata={"chunk_id": "chunk-002"}
        ),
        vector=[0.4, 0.5, 0.6]
    ),
    EmbeddingResult(
        document=Document(
            page_content="Document three",
            metadata={"chunk_id": "chunk-003"}
        ),
        vector=[0.7, 0.8, 0.9]
    )
]

new_embedding = EmbeddingResult(
    document=Document(
        page_content="Document four",
        metadata={"chunk_id": "chunk-004"}
    ),
    vector=[1.0, 1.1, 1.2]
)


vector_store = FakeVectorStore()

response1 = vector_store.add(embeddings)

response2 = vector_store.add(embeddings)


response3 = vector_store.add(
    embeddings + [new_embedding]
)

print("First add:")
print(f"Total received: {response1.total_received_chunks}")
print(f"Successful stores: {response1.total_stored_chunks}")
print(f"Skipped duplicates: {response1.total_skipped_chunks}")
print(f"Failed stores: {response1.total_failed_chunks}")

print("\nSecond add:")
print(f"Total received: {response2.total_received_chunks}")
print(f"Successful stores: {response2.total_stored_chunks}")
print(f"Skipped duplicates: {response2.total_skipped_chunks}")
print(f"Failed stores: {response2.total_failed_chunks}")

print("\nThird add:")
print(f"Total received: {response3.total_received_chunks}")
print(f"Successful stores: {response3.total_stored_chunks}")
print(f"Skipped duplicates: {response3.total_skipped_chunks}")
print(f"Failed stores: {response3.total_failed_chunks}")
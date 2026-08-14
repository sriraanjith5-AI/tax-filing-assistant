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


vector_store = FakeVectorStore()

response = vector_store.add(embeddings)

print(f"Total received: {response.total_received}")
print(f"Successful stores: {response.successful_stores}")
print(f"Skipped duplicates: {response.skipped_duplicates}")
print(f"Failed stores: {response.failed_stores}")

assert response.total_received == 3
assert response.successful_stores == 3
assert response.skipped_duplicates == 0
assert response.failed_stores == 0
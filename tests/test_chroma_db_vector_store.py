import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from embedding.embedding_dataclass import EmbeddingResult
from vectorstore.chroma_vector_store import ChromaVectorStore
from langchain_core.documents import Document

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

""" new_embedding = EmbeddingResult(
    document=Document(
        page_content="Document four",
        metadata={"chunk_id": "chunk-004"}
    ),
    vector=[1.0, 1.1, 1.2]
) """


chroma = ChromaVectorStore()
response=chroma.add(embeddings)
if response.total_received_chunks != 0:
    print("Embeddings added successfully.")
    print(f"Total stored chunks: {response.total_stored_chunks}") 
    print(f"Total skipped chunks: {response.total_skipped_chunks}") 
    print(f"Total failed chunks: {response.total_failed_chunks}") 
    print(f"Count is: {chroma.collection.count()}")

else:
    print("Failed to add embeddings.")

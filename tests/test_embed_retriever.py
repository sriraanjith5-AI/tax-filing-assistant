import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from embedding.embedding_dataclass import EmbeddingConfig, EmbeddingResult
from vectorstore.fake_vector_store import FakeVectorStore
from retrieval.vector_retriever import VectorRetriever
from embedding.huggingface_embedder import HuggingFaceEmbedder

documents = [
    Document(
        page_content="The standard deduction is available to eligible taxpayers.",
        metadata={"chunk_id": "chunk-001"}
    ),
    Document(
        page_content="Employers must withhold federal income tax from employee wages.",
        metadata={"chunk_id": "chunk-002"}
    ),
    Document(
        page_content="Form W-4 is used by employees to provide withholding information.",
        metadata={"chunk_id": "chunk-003"}
    )
]
embedding_config = EmbeddingConfig()
embedder = HuggingFaceEmbedder(embedding_config)

embedding_response = embedder.embed(documents)

assert embedding_response.embed_status == "SUCCESS"
assert len(embedding_response.successful_embeddings) == 3

vector_store=FakeVectorStore()
store_response = vector_store.add(
    embedding_response.successful_embeddings
)
assert store_response.total_stored_chunks == 3

retriever = VectorRetriever(vector_store)


query_embedder = HuggingFaceEmbedder(embedding_config)
query="How is federal income tax withholding calculated?"
query_vector = query_embedder.embed_query(query)

results = retriever.retrieve(
    query_vector,
    top_k=2
)

assert len(results) == 2
assert results[0].score >= results[1].score
assert results[0].document is not None
assert results[0].document.metadata.get("chunk_id") is not None

#assert len(results) == 2
#assert results[0].document.metadata["chunk_id"] == "chunk-001"
#assert results[1].document.metadata["chunk_id"] == "chunk-003"
#assert results[0].score >= results[1].score
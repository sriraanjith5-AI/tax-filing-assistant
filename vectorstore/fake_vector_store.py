from typing import List

from embedding.embedding_dataclass import EmbeddingResult
from vectorstore.base_vector_store import BaseVectorStore
from vectorstore.vectorstore_dataclass import VectorStoreResponse


class FakeVectorStore(BaseVectorStore):

    def __init__(self):
        self.store = {}

    def exists(self, chunk_id: str) -> bool:
        return chunk_id in self.store

    def add(self,embeddings: List[EmbeddingResult]) -> VectorStoreResponse:
        successful_stores = 0
        skipped_duplicates = 0
        failed_stores = 0

        for result in embeddings:

            chunk_id = result.document.metadata.get("chunk_id")

            if chunk_id is None:
                failed_stores += 1
                continue

            if self.exists(chunk_id):
                skipped_duplicates += 1
                continue

            self.store[chunk_id] = result
            successful_stores += 1

        return VectorStoreResponse(
            total_received_chunks =len(embeddings),
            total_stored_chunks=successful_stores,
            total_skipped_chunks=skipped_duplicates,
            total_failed_chunks=failed_stores
        )
    def search(self, query_vector, top_k):
        raise NotImplementedError("Search is not implemented yet.")
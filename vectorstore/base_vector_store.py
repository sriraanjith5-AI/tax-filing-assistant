from abc import ABC, abstractmethod
from typing import List

from embedding.embedding_dataclass import EmbeddingResult
from vectorstore.vectorstore_dataclass import VectorStoreResponse,SearchResult


class BaseVectorStore(ABC):

    @abstractmethod
    def add(self, embeddings: List[EmbeddingResult]) -> VectorStoreResponse:
        pass

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int) -> List[SearchResult]:
        pass
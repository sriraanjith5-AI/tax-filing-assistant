from abc import ABC, abstractmethod
from typing import List

from embedding.embedding_dataclass import EmbeddingResult


class BaseVectorStore(ABC):

    @abstractmethod
    def add(self, embeddings: List[EmbeddingResult]) -> None:
        pass

    @abstractmethod
    def search(self, query: str, top_k: int):
        pass
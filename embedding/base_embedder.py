from abc import ABC, abstractmethod
from langchain_core.documents import Document
from embedding.embedding_dataclass import EmbeddingResponse
from typing import List, Tuple

class BaseEmbedder(ABC):
    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def embed(self,documents: List[Document]) -> EmbeddingResponse:
        """Embeds input Document chunks into vectors.

        Returns a (successful_results, failed_documents) tuple so callers can
        see exactly which documents failed to embed, not just how many.
        """
        pass
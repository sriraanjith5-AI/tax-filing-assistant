from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document

from embedding.embedding_dataclass import EmbeddingResult
from vectorstore.vectorstore_dataclass import VectorStoreResponse,SearchResult


class BaseVectorStore(ABC):

    @abstractmethod
    def add(self, embeddings: List[EmbeddingResult]) -> VectorStoreResponse:
        pass

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int) -> List[SearchResult]:
        pass

    @abstractmethod
    def get_all_documents(self) -> List[Document]:
        """
        Every document currently held by the store, used by
        build_default_retriever (retrieval/default_retriever.py) to
        index the BM25 leg of the hybrid retriever over the same
        corpus the embedding leg searches.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Removes every document/vector currently in the store, leaving it
        empty. Used by build_index(..., fresh=True) so a caller (e.g. the
        /ask endpoint's startup warm-up - see retrieval/query_pipeline.py)
        can guarantee exactly one copy of each chunk after re-ingesting,
        instead of relying on chunk_id-based skip/reconcile logic (which
        only catches drift it can still recognize - e.g. it silently
        missed chunks whose chunk_id changed because their `source` path
        was cased differently across process runs, until
        BaseDocumentLoader.normalize_source() fixed that root cause).
        """
        pass
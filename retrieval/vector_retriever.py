from retrieval.base_retriever import BaseRetriever
from vectorstore.vectorstore_dataclass import SearchResult
from typing import List

class VectorRetriever(BaseRetriever):
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self,query_vector,top_k: int) -> List[SearchResult]:
        return self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k)


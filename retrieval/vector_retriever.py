from retrieval.base_retriever import BaseRetriever
from vectorstore.vectorstore_dataclass import SearchResult
from typing import List
from utils.trace import record_stage, summarize_results

class VectorRetriever(BaseRetriever):
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self,query_vector,top_k: int,query_text: str = None) -> List[SearchResult]:
        # query_text is accepted (and ignored) so VectorRetriever is
        # interchangeable with other BaseRetriever implementations
        # (BM25Retriever, HybridRetriever, CrossEncoderReranker) that
        # do need it.
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k)
        record_stage("embedding_search", top_k=top_k, results=summarize_results(results))
        return results


from typing import List

from retrieval.base_retriever import BaseRetriever
from vectorstore.vectorstore_dataclass import SearchResult
from utils.trace import record_stage, summarize_results


class HybridRetriever(BaseRetriever):
    """
    Fuses embedding-based (semantic) and BM25 (lexical) candidates
    via Reciprocal Rank Fusion (RRF) before handing the merged
    candidate set off to reranking / truncation.

    RRF score for a document = sum, over each retriever's rank list,
    of 1 / (rrf_k + rank). A document that both retrievers agree on
    (even at different ranks) gets boosted; a document unique to one
    list still survives if it's ranked well there. This combines
    embeddings' strength at topical/semantic matches with BM25's
    strength at exact lexical distinctions, without either one
    silently dropping candidates the other would have caught.
    """

    def __init__(
        self,
        embedding_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        rrf_k: int = 60,
        fetch_k: int = 20,
    ):
        self.embedding_retriever = embedding_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.fetch_k = fetch_k

    def retrieve(
        self,
        query_vector,
        top_k: int,
        query_text: str = None,
    ) -> List[SearchResult]:

        if query_text is None:
            raise ValueError(
                "HybridRetriever.retrieve() requires query_text for "
                "BM25 lexical scoring."
            )

        candidate_k = max(top_k, self.fetch_k)

        embedding_results = self.embedding_retriever.retrieve(
            query_vector,
            top_k=candidate_k,
            query_text=query_text,
        )

        bm25_results = self.bm25_retriever.retrieve(
            query_vector,
            top_k=candidate_k,
            query_text=query_text,
        )

        rrf_scores = {}
        doc_lookup = {}

        for result_list in (embedding_results, bm25_results):
            for rank, result in enumerate(result_list, start=1):

                key = self._doc_key(result.document)

                doc_lookup[key] = result.document

                rrf_scores[key] = (
                    rrf_scores.get(key, 0.0)
                    + 1.0 / (self.rrf_k + rank)
                )

        ranked_keys = sorted(
            rrf_scores,
            key=lambda key: rrf_scores[key],
            reverse=True,
        )

        # top_k may be smaller than candidate_k (a direct standalone
        # call) or equal to it (called as the base_retriever inside
        # CrossEncoderReranker, which wants the full wide set) -
        # either way, return at most top_k fused results.
        result_count = min(top_k, candidate_k)

        results = [
            SearchResult(
                document=doc_lookup[key],
                score=rrf_scores[key],
            )
            for key in ranked_keys[:result_count]
        ]
        record_stage(
            "rrf_fusion",
            rrf_k=self.rrf_k,
            fetch_k=candidate_k,
            embedding_candidate_count=len(embedding_results),
            bm25_candidate_count=len(bm25_results),
            fused=summarize_results(results),
        )
        return results

    @staticmethod
    def _doc_key(document):
        chunk_id = document.metadata.get("chunk_id")
        return chunk_id if chunk_id is not None else id(document)

from typing import List

from retrieval.base_retriever import BaseRetriever
from vectorstore.vectorstore_dataclass import SearchResult


class QueryExpansionRetriever(BaseRetriever):
    """
    Retrieves for the original query AND LLM-generated paraphrases
    of it (via QueryExpander), then merges every retrieval run with
    Reciprocal Rank Fusion before handing the merged candidate set
    off to reranking / truncation.
    """

    def __init__(
        self,
        embedding_retriever: BaseRetriever,
        embedder,
        query_expander,
        rrf_k: int = 60,
        fetch_k: int = 20,
    ):
        self.embedding_retriever = embedding_retriever
        self.embedder = embedder
        self.query_expander = query_expander
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
                "QueryExpansionRetriever.retrieve() requires "
                "query_text to generate paraphrases."
            )

        candidate_k = max(top_k, self.fetch_k)

        query_variants = [query_text] + self.query_expander.expand(
            query_text
        )

        rrf_scores = {}
        doc_lookup = {}

        for variant in query_variants:

            variant_vector = (
                query_vector
                if variant == query_text
                else self.embedder.embed_query(variant)
            )

            results = self.embedding_retriever.retrieve(
                variant_vector,
                top_k=candidate_k,
                query_text=variant,
            )

            for rank, result in enumerate(results, start=1):

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

        result_count = min(top_k, candidate_k)

        return [
            SearchResult(
                document=doc_lookup[key],
                score=rrf_scores[key],
            )
            for key in ranked_keys[:result_count]
        ]

    @staticmethod
    def _doc_key(document):
        chunk_id = document.metadata.get("chunk_id")
        return chunk_id if chunk_id is not None else id(document)

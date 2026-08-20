from config import (
    RERANK_MODEL_STRONG,
    RERANK_FETCH_K,
    RERANK_SCORE_THRESHOLD,
)
from retrieval.vector_retriever import VectorRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.cross_encoder_reranker import CrossEncoderReranker


def build_default_retriever(vector_store):
    """
    Builds the project's default retriever: BM25 lexical search fused
    with embedding search (HybridRetriever), reranked with the
    stronger L-12 cross-encoder, with weak candidates dropped via a
    score threshold instead of always padding to a fixed top_k.

    This is the "combined" variant from
    evaluation/results/ragas_precision_experiments_summary_*.csv -
    the best precision AND recall of every variant tried
    (avg precision 0.9066, avg recall 1.0000 vs. baseline embedding
    + L-6 rerank's 0.8541 / 0.9222), at ~970ms avg retrieval latency.

    Suggestion #2 (LLM query expansion) was tried and dropped - it
    regressed both metrics on its own. See retrieval/query_expander.py
    and retrieval/query_expansion_retriever.py if revisiting it.

    Parameters
    ----------
    vector_store : BaseVectorStore
        Must already be populated (add() called) - BM25Retriever
        indexes every document currently in vector_store.store.
    """

    vector_retriever = VectorRetriever(vector_store)

    all_documents = [
        result.document for result in vector_store.store.values()
    ]

    bm25_retriever = BM25Retriever(all_documents)

    hybrid_retriever = HybridRetriever(
        embedding_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        fetch_k=RERANK_FETCH_K,
    )

    return CrossEncoderReranker(
        base_retriever=hybrid_retriever,
        model_name=RERANK_MODEL_STRONG,
        fetch_k=RERANK_FETCH_K,
        score_threshold=RERANK_SCORE_THRESHOLD,
    )

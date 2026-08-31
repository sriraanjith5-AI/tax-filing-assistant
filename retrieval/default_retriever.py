from config import (
    RERANK_MODEL_STRONG,
    RERANK_FETCH_K,
    RERANK_SCORE_THRESHOLD,
    CONTEXT_EXPANSION_WINDOW,
)
from retrieval.vector_retriever import VectorRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.cross_encoder_reranker import CrossEncoderReranker
from retrieval.context_expander import ContextExpandingRetriever


def build_retriever(kind: str, vector_store):
    """
    Builds one of the project's retriever variants over a populated
    vector store, so a caller (e.g. the comparison UI) can pick a
    search mechanism by name instead of wiring retrievers by hand.

    Parameters
    ----------
    kind : str
        One of:
        - "vector"                   : embedding similarity only (VectorRetriever)
        - "bm25"                      : lexical keyword search only (BM25Retriever)
        - "hybrid"                    : BM25 + embedding fused via RRF (HybridRetriever)
        - "hybrid_reranked"           : "hybrid", then cross-encoder reranked -
                                          this is the project's default (see
                                          build_default_retriever below).
        - "hybrid_reranked_expanded"  : "hybrid_reranked", then each
                                          surviving chunk has its
                                          +/- CONTEXT_EXPANSION_WINDOW
                                          neighbors merged in by document
                                          position (ContextExpandingRetriever)
                                          so chunk boundaries don't cut off
                                          context the generator needs.

    vector_store : BaseVectorStore
        Must already be populated (add() called) - BM25/hybrid variants
        index every document returned by vector_store.get_all_documents().
    """

    vector_retriever = VectorRetriever(vector_store)

    if kind == "vector":
        return vector_retriever

    all_documents = vector_store.get_all_documents()
    bm25_retriever = BM25Retriever(all_documents)

    if kind == "bm25":
        return bm25_retriever

    hybrid_retriever = HybridRetriever(
        embedding_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        fetch_k=RERANK_FETCH_K,
    )

    if kind == "hybrid":
        return hybrid_retriever

    reranked_retriever = CrossEncoderReranker(
        base_retriever=hybrid_retriever,
        model_name=RERANK_MODEL_STRONG,
        fetch_k=RERANK_FETCH_K,
        score_threshold=RERANK_SCORE_THRESHOLD,
    )

    if kind == "hybrid_reranked":
        return reranked_retriever

    if kind == "hybrid_reranked_expanded":
        return ContextExpandingRetriever(
            base_retriever=reranked_retriever,
            vector_store=vector_store,
            window=CONTEXT_EXPANSION_WINDOW,
        )

    raise ValueError(
        f"Unknown retriever kind '{kind}'. Valid options: vector, bm25, "
        "hybrid, hybrid_reranked, hybrid_reranked_expanded."
    )


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
    regressed both metrics on its own. See
    experiments/retrieval/query_expander.py and
    experiments/retrieval/query_expansion_retriever.py if revisiting it.

    Equivalent to build_retriever("hybrid_reranked", vector_store) -
    kept as a separate named entrypoint since it's the shipped default.

    Parameters
    ----------
    vector_store : BaseVectorStore
        Must already be populated (add() called) - BM25Retriever
        indexes every document returned by vector_store.get_all_documents().
    """

    return build_retriever("hybrid_reranked", vector_store)

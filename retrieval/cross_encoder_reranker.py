from typing import List

import numpy as np
from sentence_transformers import CrossEncoder

from retrieval.base_retriever import BaseRetriever
from vectorstore.vectorstore_dataclass import SearchResult
from utils.trace import record_stage, page_display


class CrossEncoderReranker(BaseRetriever):
    """
    Two-stage retriever: retrieve wide with the underlying embedding
    retriever, then rerank with a cross-encoder before truncating.

    Embedding similarity (stage 1) is good at finding topically
    related chunks but weak at fine-grained distinctions (e.g.
    "Form W-4 Step 3" vs. "Form W-4P Step 2"). A cross-encoder scores
    the (query, chunk) pair jointly instead of comparing two separate
    vectors, so it is much better at those distinctions.

    Retrieving wide in stage 1 keeps recall high (same effect as
    raising top_k); reranking in stage 2 pushes irrelevant chunks
    out of the final top_k, which is what improves precision.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        fetch_k: int = 15,
        score_threshold: float = None,
    ):
        """
        Parameters
        ----------
        base_retriever : BaseRetriever
            The stage-1 retriever (e.g. VectorRetriever) used to
            fetch the wide candidate set.

        model_name : str
            HuggingFace cross-encoder model used for stage-2
            reranking.

        fetch_k : int
            Number of candidates pulled from base_retriever before
            reranking. Should be >= the top_k requested from
            retrieve().

        score_threshold : float, optional
            If set, the cross-encoder's raw score is squashed to a
            0-1 probability via sigmoid, and only candidates scoring
            >= score_threshold are kept (still capped at top_k).
            This trades a fixed "always return top_k" for "return
            however many are actually relevant" - if only 2 of 5
            candidates clear the bar, only 2 are returned, which is
            what actually drives precision up instead of padding the
            result with borderline chunks. Without a threshold, the
            reranker always returns exactly top_k (or fewer if there
            aren't enough candidates).
        """

        self.base_retriever = base_retriever
        self.fetch_k = fetch_k
        self.score_threshold = score_threshold
        self.model_name = model_name
        self.cross_encoder = CrossEncoder(model_name)

    def retrieve(
        self,
        query_vector,
        top_k: int,
        query_text: str = None,
    ) -> List[SearchResult]:
        """
        Parameters
        ----------
        query_vector
            Passed through to the base retriever for stage-1 search.

        top_k : int
            Number of results to return AFTER reranking.

        query_text : str
            The raw query string. Required for reranking, since the
            cross-encoder scores (query_text, chunk_text) pairs
            directly rather than comparing vectors.
        """

        if query_text is None:
            raise ValueError(
                "CrossEncoderReranker.retrieve() requires query_text "
                "for cross-encoder scoring."
            )

        # ========================================================
        # STAGE 1: RETRIEVE WIDE
        # ========================================================

        candidate_k = max(top_k, self.fetch_k)

        candidates = self.base_retriever.retrieve(
            query_vector,
            top_k=candidate_k,
            query_text=query_text,
        )

        if not candidates:
            return []

        # ========================================================
        # STAGE 2: CROSS-ENCODER RERANK
        # ========================================================

        pairs = [
            (query_text, candidate.document.page_content)
            for candidate in candidates
        ]

        rerank_scores = self.cross_encoder.predict(pairs)

        if self.score_threshold is not None:
            # Squash raw cross-encoder logits to a 0-1 probability so
            # the threshold has a meaningful, model-independent scale.
            rerank_probs = 1.0 / (1.0 + np.exp(-np.asarray(rerank_scores)))
        else:
            rerank_probs = rerank_scores

        reranked = sorted(
            zip(candidates, rerank_scores, rerank_probs),
            key=lambda triple: triple[1],
            reverse=True,
        )

        # ========================================================
        # STAGE 3: FILTER (OPTIONAL) + TRUNCATE TO TOP_K
        # ========================================================

        if self.score_threshold is not None:
            filtered = [
                triple for triple in reranked
                if triple[2] >= self.score_threshold
            ]
            # Never return zero results just because nothing cleared
            # the bar - fall back to the single best candidate.
            reranked = filtered if filtered else reranked[:1]

        kept = reranked[:top_k]
        kept_ids = {id(candidate) for candidate, _, _ in kept}

        # Every scored candidate, including ones the threshold dropped -
        # this is what makes "why wasn't chunk X in the answer?" an
        # answerable question (dropped by the score bar vs. never
        # reached stage 1 at all are very different failure modes).
        record_stage(
            "cross_encoder_rerank",
            model_name=self.model_name,
            fetch_k=self.fetch_k,
            score_threshold=self.score_threshold,
            candidates=[
                {
                    "chunk_id": candidate.document.metadata.get("chunk_id"),
                    "source": candidate.document.metadata.get("source"),
                    "page_display": page_display(candidate.document.metadata),
                    "raw_score": round(float(raw_score), 4),
                    "probability": round(float(prob), 4) if self.score_threshold is not None else None,
                    "kept": id(candidate) in kept_ids,
                }
                for candidate, raw_score, prob in reranked
            ],
        )

        return [
            SearchResult(
                document=candidate.document,
                score=float(rerank_score),
            )
            for candidate, rerank_score, _ in kept
        ]

import math

from evaluation.evidence_matching import (
    extract_evidence_items,
    find_matching_evidence,
)


class RetrievalMetrics:
    """
    Classical, non-LLM information-retrieval metrics for the
    retrieval stage: Recall@K, MRR, and NDCG@K.

    These are deterministic and free (no judge LLM call) - unlike
    ragas' LLMContextPrecisionWithReference / LLMContextRecall
    (evaluation/ragas_retrieval_evaluator.py), which use an LLM to
    judge semantic relevance against a reference ANSWER. These
    metrics instead compare retrieved chunk text directly against
    the golden dataset's stable evidence TEXT (same substring-match
    logic RAGEvaluator uses for hit_at_k/recall_at_k/reciprocal_rank -
    see evaluation/evidence_matching.py), producing binary relevance
    judgments per retrieved chunk. They are NOT inputs to ragas and
    ragas does not consume them - the two approaches are run side by
    side, not composed.

    Relevance is binary per retrieved chunk: 1 if it contains at
    least one evidence item's text, else 0. Multiple evidence items
    per question are supported (Recall@K = fraction of distinct
    evidence items covered within the top K), but the golden dataset
    currently ships one evidence string per question, so in practice
    Recall@K reduces to Hit@K for this dataset - it becomes
    meaningfully different from Hit@K once a question's ground truth
    has more than one evidence passage.
    """

    def compute(
        self,
        retrieved_results,
        retrieval_ground_truth,
        k_values=(1, 3, 5, 10),
        ndcg_k=10,
    ):
        """
        Parameters
        ----------
        retrieved_results : list[SearchResult]
            Retriever output, already rank-ordered (rank 1 = index 0).
            Should have at least max(k_values, ndcg_k) results for
            every K to be measured meaningfully - fewer than K
            retrieved results just means Recall@K/NDCG@K are computed
            over however many were actually returned.

        retrieval_ground_truth : dict
            The golden dataset entry's "retrieval_ground_truth" field.

        k_values : tuple[int]
            Cutoffs to compute Recall@K for.

        ndcg_k : int
            Cutoff to compute NDCG@K for.

        Returns
        -------
        dict with keys "recall_at_{k}" for each k in k_values, plus
        "mrr" and f"ndcg_at_{ndcg_k}". Returns all-zero if there is
        no valid ground-truth evidence.
        """

        evidence_items = extract_evidence_items(retrieval_ground_truth)

        result = {f"recall_at_{k}": 0.0 for k in k_values}
        result["mrr"] = 0.0
        result[f"ndcg_at_{ndcg_k}"] = 0.0

        if not evidence_items or not retrieved_results:
            return result

        # ------------------------------------------------------
        # Binary relevance per retrieved chunk (rank-ordered), and
        # which evidence indexes each rank covers.
        # ------------------------------------------------------

        relevance = []
        covered_indexes_by_rank = []

        for retrieved_result in retrieved_results:

            document = retrieved_result.document

            if document is None:
                relevance.append(0)
                covered_indexes_by_rank.append(set())
                continue

            matched_indexes = find_matching_evidence(
                chunk_content=document.page_content,
                evidence_items=evidence_items,
            )

            relevance.append(1 if matched_indexes else 0)
            covered_indexes_by_rank.append(set(matched_indexes))

        total_evidence_items = len(evidence_items)

        # ------------------------------------------------------
        # Recall@K: fraction of distinct evidence items covered by
        # the top K retrieved chunks.
        # ------------------------------------------------------

        for k in k_values:

            covered = set()

            for indexes in covered_indexes_by_rank[:k]:
                covered.update(indexes)

            result[f"recall_at_{k}"] = len(covered) / total_evidence_items

        # ------------------------------------------------------
        # MRR: 1 / rank of the first relevant chunk (0 if none).
        # ------------------------------------------------------

        for rank, is_relevant in enumerate(relevance, start=1):
            if is_relevant:
                result["mrr"] = 1.0 / rank
                break

        # ------------------------------------------------------
        # NDCG@K: binary relevance, so
        #   DCG  = sum_{i=1}^{K} rel_i / log2(i + 1)
        #   IDCG = the same sum for the ideal ordering (all
        #          relevant chunks first), i.e. sum over the top
        #          min(#relevant, K) ranks.
        # ------------------------------------------------------

        top_k_relevance = relevance[:ndcg_k]

        dcg = sum(
            rel / math.log2(rank + 1)
            for rank, rel in enumerate(top_k_relevance, start=1)
        )

        ideal_relevant_count = min(sum(relevance), ndcg_k)

        idcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_relevant_count + 1)
        )

        result[f"ndcg_at_{ndcg_k}"] = (dcg / idcg) if idcg > 0 else 0.0

        return result

from evaluation.evaluation_dataclass import EvaluationResult


class RAGEvaluator:

    def evaluate_retrieval(
        self,
        question_id,
        query,
        expected_answer,
        retrieved_results
    ):

        retrieved_count = len(retrieved_results)

        if retrieved_count == 0:
            return EvaluationResult(
                question_id=question_id,
                query=query,
                expected_answer=expected_answer,
                actual_answer="",
                retrieved_count=0,
                top_score=0.0,
                retrieval_success=False,
                answer_score=0.0,
                evaluation_status="RETRIEVAL_FAILED"
            )

        top_score = retrieved_results[0].score

        return EvaluationResult(
            question_id=question_id,
            query=query,
            expected_answer=expected_answer,
            actual_answer="",
            retrieved_count=retrieved_count,
            top_score=top_score,
            retrieval_success=True,
            answer_score=0.0,
            evaluation_status="RETRIEVAL_SUCCESS"
        )
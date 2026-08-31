from dataclasses import dataclass


@dataclass
class EvaluationResult:

    question_id: str
    query: str
    expected_answer: str
    actual_answer: str

    retrieved_count: int
    top_score: float

    retrieval_success: bool

    answer_score: float

    evaluation_status: str

    hit_at_k: float
    recall_at_k: float
    reciprocal_rank: float
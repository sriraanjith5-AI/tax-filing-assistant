from dataclasses import dataclass, field
from typing import List

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

    # --- Accuracy metrics (relevance against the golden dataset) ---
    page_hit: bool = False
    matched_keywords: List[str] = field(default_factory=list)
    missing_keywords: List[str] = field(default_factory=list)
    keyword_coverage: float = 0.0
    accuracy_status: str = "NOT_EVALUATED"

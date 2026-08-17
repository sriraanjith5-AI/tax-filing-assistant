from dataclasses import dataclass
from typing import List


@dataclass
class GoldenEvidence:
    source: str
    page: int
    text: str


@dataclass
class GoldenCase:
    case_id: str
    query: str
    expected_answer: str
    evidence: List[GoldenEvidence]
    category: str
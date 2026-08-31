from dataclasses import dataclass


@dataclass
class GenerationResult:
    answer: str
    latency_ms: float

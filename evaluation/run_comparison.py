import json
import statistics
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config import (
    EVAL_DIR,
    TOP_K,
    EMBEDDING_MODEL,
    RERANK_MODEL_STRONG,
    RERANK_FETCH_K,
    RERANK_SCORE_THRESHOLD,
    GENERATION_MODEL,
)
from ingestion.build_index import build_index
from ingestion.registry import build_retriever
from evaluation.retrieval_metrics import RetrievalMetrics

GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset_with_retrieval_ground_truth.json"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_RUNS_CSV = RESULTS_DIR / "comparison_runs.csv"

CLASSICAL_METRIC_KEYS = (
    "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
    "mrr", "ndcg_at_10",
)


@dataclass
class RunConfig:
    chunker: str
    vector_store: str
    retriever: str
    chunk_size: int
    chunk_overlap: int
    loader: str = "pypdf"
    include_ragas: bool = False
    include_generation: bool = False


@dataclass
class RunResult:
    run_id: str
    config: RunConfig
    status: str = "pending"  # pending -> running -> done | failed
    classical_metrics: Optional[dict] = None
    ragas_metrics: Optional[dict] = None
    generation_metrics: Optional[dict] = None
    sample_answers: Optional[list] = None
    avg_latency_ms: Optional[float] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "config": asdict(self.config),
            "status": self.status,
            "classical_metrics": self.classical_metrics,
            "ragas_metrics": self.ragas_metrics,
            "generation_metrics": self.generation_metrics,
            "sample_answers": self.sample_answers,
            "avg_latency_ms": self.avg_latency_ms,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def _load_golden_dataset() -> list:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["entries"]


def _average_classical_metrics(rows: list) -> dict:
    if not rows:
        return {key: 0.0 for key in CLASSICAL_METRIC_KEYS}
    return {
        key: statistics.mean(row[key] for row in rows)
        for key in CLASSICAL_METRIC_KEYS
    }


def _append_run_to_csv(result: RunResult) -> None:
    # Every execution - success or failure (called from execute_run's
    # `finally`) - gets one row, covering both the parameters the UI
    # let the user choose (config.*) and the ones fixed in config.py at
    # the time of this run (top_k/embedding/reranker settings). The
    # latter aren't swept via the UI, but they were still parameters
    # used for this execution - recording their actual value here means
    # a later change to config.py can't make old rows ambiguous about
    # what actually ran.
    is_reranked = result.config.retriever == "hybrid_reranked"
    row = {
        "run_id": result.run_id,
        "timestamp": result.finished_at,
        "loader": result.config.loader,
        "chunker": result.config.chunker,
        "chunk_size": result.config.chunk_size,
        "chunk_overlap": result.config.chunk_overlap,
        "vector_store": result.config.vector_store,
        "retriever": result.config.retriever,
        "top_k": TOP_K,
        "embedding_model": EMBEDDING_MODEL,
        "rerank_model": RERANK_MODEL_STRONG if is_reranked else None,
        "rerank_fetch_k": RERANK_FETCH_K if is_reranked else None,
        "rerank_score_threshold": RERANK_SCORE_THRESHOLD if is_reranked else None,
        "include_ragas": result.config.include_ragas,
        "include_generation": result.config.include_generation,
        "generation_model": GENERATION_MODEL if result.config.include_generation else None,
        **(result.classical_metrics or {}),
        "avg_context_precision": (result.ragas_metrics or {}).get("avg_context_precision"),
        "avg_context_recall": (result.ragas_metrics or {}).get("avg_context_recall"),
        "avg_faithfulness": (result.generation_metrics or {}).get("avg_faithfulness"),
        "avg_answer_relevancy": (result.generation_metrics or {}).get("avg_answer_relevancy"),
        "avg_answer_correctness": (result.generation_metrics or {}).get("avg_answer_correctness"),
        "avg_latency_ms": result.avg_latency_ms,
        "status": result.status,
        "error": result.error,
    }
    # Read-merge-rewrite rather than a pure append: the row schema has
    # already changed once (new columns added after earlier rows were
    # written), and appending raw would misalign those older rows
    # against the new header. pandas.concat aligns on column name, so
    # any future schema change self-heals (missing columns -> NaN)
    # instead of silently corrupting the file. The run count is small
    # (a handful of comparisons per session), so rewriting the whole
    # file each time is cheap.
    new_row_df = pd.DataFrame([row])
    existing = load_run_history()
    combined = pd.concat([existing, new_row_df], ignore_index=True) if not existing.empty else new_row_df
    combined.to_csv(COMPARISON_RUNS_CSV, index=False)


def load_run_history() -> pd.DataFrame:
    if not COMPARISON_RUNS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(COMPARISON_RUNS_CSV)


def execute_run(run_id: str, config: RunConfig, run_store: dict) -> None:
    """
    Runs one chunker/vector_store/retriever configuration end-to-end
    (ingest -> retrieve -> score) and records the result. Designed to
    be handed to FastAPI's BackgroundTasks - run_store[run_id] is
    mutated in place so a polling endpoint can report progress/results.
    """

    result: RunResult = run_store[run_id]
    result.status = "running"
    result.started_at = datetime.now(timezone.utc).isoformat()

    try:
        store, embedder, _add_response = build_index(
            config.chunker,
            config.vector_store,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            loader_name=config.loader,
        )
        retriever = build_retriever(config.retriever, store)

        golden_dataset = _load_golden_dataset()

        ragas_evaluator = None
        if config.include_ragas:
            from evaluation.ragas_retrieval_evaluator import RagasRetrievalEvaluator
            ragas_evaluator = RagasRetrievalEvaluator()

        generator = None
        generation_evaluator = None
        if config.include_generation:
            from llm.generator import OpenAIGenerator
            from evaluation.generation_metrics import RagasGenerationEvaluator
            generator = OpenAIGenerator()
            generation_evaluator = RagasGenerationEvaluator()

        classical_rows = []
        ragas_samples = []
        generation_samples = []
        sample_answers = []
        latencies_ms = []

        for case in golden_dataset:
            query = case["query"]
            query_vector = embedder.embed_query(query)

            start = time.perf_counter()
            retrieved_results = retriever.retrieve(
                query_vector, top_k=TOP_K, query_text=query,
            )
            latencies_ms.append((time.perf_counter() - start) * 1000)

            classical_rows.append(
                RetrievalMetrics().compute(
                    retrieved_results, case["retrieval_ground_truth"],
                )
            )

            if ragas_evaluator is not None:
                ragas_samples.append(
                    ragas_evaluator.build_sample(
                        query=query,
                        retrieved_results=retrieved_results,
                        expected_answer=case["expected_answer"],
                    )
                )

            if generator is not None:
                generation_result = generator.generate(
                    query,
                    [r.document.page_content for r in retrieved_results if r.document is not None],
                )
                generation_samples.append(
                    generation_evaluator.build_sample(
                        query=query,
                        retrieved_results=retrieved_results,
                        answer=generation_result.answer,
                        expected_answer=case["expected_answer"],
                    )
                )
                if len(sample_answers) < 3:
                    sample_answers.append({
                        "query": query,
                        "answer": generation_result.answer,
                        "expected_answer": case["expected_answer"],
                    })

        result.classical_metrics = _average_classical_metrics(classical_rows)
        result.avg_latency_ms = statistics.mean(latencies_ms) if latencies_ms else None

        if ragas_evaluator is not None:
            ragas_df = ragas_evaluator.evaluate(ragas_samples).to_pandas()
            result.ragas_metrics = {
                "avg_context_precision": float(ragas_df["llm_context_precision_with_reference"].mean()),
                "avg_context_recall": float(ragas_df["context_recall"].mean()),
            }

        if generation_evaluator is not None:
            generation_df = generation_evaluator.evaluate(generation_samples).to_pandas()
            result.generation_metrics = {
                "avg_faithfulness": float(generation_df["faithfulness"].mean()),
                "avg_answer_relevancy": float(generation_df["answer_relevancy"].mean()),
                "avg_answer_correctness": float(generation_df["answer_correctness"].mean()),
            }
            result.sample_answers = sample_answers

        result.status = "done"

    except Exception as exc:
        result.status = "failed"
        result.error = f"{exc}\n{traceback.format_exc()}"

    finally:
        result.finished_at = datetime.now(timezone.utc).isoformat()
        _append_run_to_csv(result)

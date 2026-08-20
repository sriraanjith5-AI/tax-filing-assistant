import sys
import json
import time
import statistics
from pathlib import Path
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from embedding.huggingface_embedder import HuggingFaceEmbedder
from config import (
    PDF_DIR,
    EVAL_DIR,
    RERANK_MODEL,
    RERANK_MODEL_STRONG,
    RERANK_FETCH_K,
    RERANK_SCORE_THRESHOLD,
    TOP_K,
)
from ingestion.pdfloader import PDFLoader
from ingestion.cleaner_pipeline import CleanerPipeline
from cleaners.header_cleaner import HeaderCleaner
from cleaners.footer_cleaner import FooterCleaner
from cleaners.whitespace_cleaner import WhitespaceCleaner
from chunking.recursive_chunker import RecursiveChunker
from utils.logger import logging
from embedding.embedding_dataclass import EmbeddingConfig
from vectorstore.fake_vector_store import FakeVectorStore
from retrieval.vector_retriever import VectorRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.cross_encoder_reranker import CrossEncoderReranker
from evaluation.ragas_retrieval_evaluator import RagasRetrievalEvaluator


logger = logging.getLogger(__name__)


# ============================================================
# RESULTS DIR
# ============================================================

RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# 1. LOAD -> CLEAN -> CHUNK -> EMBED -> STORE (once, reused
#    across every variant)
# ============================================================

pdf_loader = PDFLoader(PDF_DIR)
loaded_docs = pdf_loader.load_documents()

if not loaded_docs:
    raise RuntimeError("No documents were loaded from PDF_DIR.")

pipeline = CleanerPipeline(
    cleaners=[
        HeaderCleaner(),
        FooterCleaner(),
        WhitespaceCleaner(),
    ]
)

clean_doc = pipeline.clean(loaded_docs)

chunking = RecursiveChunker()
final_chunks = chunking.chunk(loaded_docs)

embedding_config = EmbeddingConfig()
embedder = HuggingFaceEmbedder(embedding_config)

embedding_response = embedder.embed(final_chunks)

assert embedding_response.embed_status == "SUCCESS"
assert len(embedding_response.successful_embeddings) == len(final_chunks)

vector_store = FakeVectorStore()
store_response = vector_store.add(embedding_response.successful_embeddings)

assert store_response.total_stored_chunks == len(final_chunks)

vector_retriever = VectorRetriever(vector_store)

all_documents = [
    result.document for result in vector_store.store.values()
]

bm25_retriever = BM25Retriever(all_documents)


# ============================================================
# 2. LOAD GOLDEN DATASET
# ============================================================

golden_dataset_path = (
    EVAL_DIR / "golden_dataset_with_retrieval_ground_truth.json"
)

with open(golden_dataset_path, "r", encoding="utf-8") as file:
    golden_dataset = json.load(file)


# ============================================================
# 3. QUERY EMBEDDINGS (once, reused across every variant)
# ============================================================

query_vectors = {}

for test_case in golden_dataset["entries"]:
    query_vectors[test_case["id"]] = embedder.embed_query(
        test_case["query"]
    )


# ============================================================
# 4. DEFINE VARIANTS
# ============================================================
# baseline           : current default (embedding -> rerank L-6,
#                       fixed top_k) - suggestion already shipped.
# stronger_reranker   : suggestion #5, bigger cross-encoder (L-12)
# score_threshold      : suggestion #4, threshold instead of fixed k
# hybrid_bm25         : suggestion #6, BM25 + embedding fused -> rerank
# combined            : hybrid_bm25 + stronger_reranker + score_threshold
#
# Suggestion #2 (LLM query expansion) is dropped from this run - it
# regressed both precision AND recall in the prior comparison
# (evaluation/results/ragas_precision_experiments_summary_20260820_193520.csv):
# avg precision 0.8481 -> 0.8408, avg recall 0.9333 -> 0.8833.

variants = {
    "baseline": CrossEncoderReranker(
        base_retriever=vector_retriever,
        model_name=RERANK_MODEL,
        fetch_k=RERANK_FETCH_K,
    ),
    "stronger_reranker": CrossEncoderReranker(
        base_retriever=vector_retriever,
        model_name=RERANK_MODEL_STRONG,
        fetch_k=RERANK_FETCH_K,
    ),
    "score_threshold": CrossEncoderReranker(
        base_retriever=vector_retriever,
        model_name=RERANK_MODEL,
        fetch_k=RERANK_FETCH_K,
        score_threshold=RERANK_SCORE_THRESHOLD,
    ),
    "hybrid_bm25": CrossEncoderReranker(
        base_retriever=HybridRetriever(
            embedding_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            fetch_k=RERANK_FETCH_K,
        ),
        model_name=RERANK_MODEL,
        fetch_k=RERANK_FETCH_K,
    ),
    "combined": CrossEncoderReranker(
        base_retriever=HybridRetriever(
            embedding_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            fetch_k=RERANK_FETCH_K,
        ),
        model_name=RERANK_MODEL_STRONG,
        fetch_k=RERANK_FETCH_K,
        score_threshold=RERANK_SCORE_THRESHOLD,
    ),
}


# ============================================================
# 5. RUN RAGAS EVALUATION FOR EACH VARIANT
# ============================================================

ragas_evaluator = RagasRetrievalEvaluator()

per_variant_dataframes = {}

for variant_name, active_retriever in variants.items():

    print(f"\n{'=' * 80}")
    print(f"RUNNING RAGAS EVALUATION | variant={variant_name}")
    print(f"{'=' * 80}\n")

    samples = []
    latencies_ms = []

    for test_case in golden_dataset["entries"]:

        query_id = test_case["id"]
        query = test_case["query"]
        expected_answer = test_case["expected_answer"]
        query_vector = query_vectors[query_id]

        # ----------------------------------------------------------
        # Wall-clock retrieval latency for this variant. This times
        # ONLY active_retriever.retrieve() - the retrieval-side cost
        # (embedding search, BM25 lookup, cross-encoder reranking,
        # any LLM calls a variant makes) - not the downstream ragas
        # judge call, which is identical infra cost across variants
        # and would otherwise swamp the comparison.
        # ----------------------------------------------------------

        retrieval_start = time.perf_counter()

        retrieved_results = active_retriever.retrieve(
            query_vector,
            top_k=TOP_K,
            query_text=query,
        )

        latencies_ms.append(
            (time.perf_counter() - retrieval_start) * 1000
        )

        sample = ragas_evaluator.build_sample(
            query=query,
            retrieved_results=retrieved_results,
            expected_answer=expected_answer,
        )

        sample["question_id"] = query_id

        samples.append(sample)

    result = ragas_evaluator.evaluate(samples)

    result_df = result.to_pandas()
    result_df.insert(0, "question_id", [s["question_id"] for s in samples])
    result_df.insert(1, "variant", variant_name)
    result_df["retrieval_latency_ms"] = latencies_ms

    csv_path = (
        RESULTS_DIR
        / f"ragas_evaluation_run_{variant_name}_{timestamp}.csv"
    )
    result_df.to_csv(csv_path, index=False)

    print(f"\nvariant={variant_name} results written to: {csv_path}")
    print(result)

    per_variant_dataframes[variant_name] = result_df


# ============================================================
# 6. BUILD COMPARISON TABLE
# ============================================================

metric_columns = [
    "llm_context_precision_with_reference",
    "context_recall",
]

baseline_variant = "baseline"

comparison_df = per_variant_dataframes[baseline_variant][
    ["question_id", "user_input", "reference"]
].copy()

for variant_name in variants:
    df_v = per_variant_dataframes[variant_name]
    for metric in metric_columns:
        comparison_df[f"{metric}_{variant_name}"] = df_v[metric].values
    comparison_df[f"retrieval_latency_ms_{variant_name}"] = (
        df_v["retrieval_latency_ms"].values
    )

for variant_name in variants:
    if variant_name == baseline_variant:
        continue
    for metric in metric_columns:
        base_col = f"{metric}_{baseline_variant}"
        new_col = f"{metric}_{variant_name}"
        delta_col = f"{metric}_delta_{variant_name}"
        improved_col = f"{metric}_improved_{variant_name}"

        comparison_df[delta_col] = (
            comparison_df[new_col] - comparison_df[base_col]
        )
        comparison_df[improved_col] = comparison_df[delta_col] > 0

comparison_csv_path = (
    RESULTS_DIR / f"ragas_precision_experiments_comparison_{timestamp}.csv"
)
comparison_df.to_csv(comparison_csv_path, index=False)


# ============================================================
# 7. SUMMARY
# ============================================================

print(f"\n{'=' * 80}")
print("PRECISION EXPERIMENTS SUMMARY")
print(f"{'=' * 80}\n")

summary_rows = []

for variant_name in variants:
    df_v = per_variant_dataframes[variant_name]
    avg_precision = df_v["llm_context_precision_with_reference"].mean()
    avg_recall = df_v["context_recall"].mean()

    latencies = df_v["retrieval_latency_ms"].tolist()
    avg_latency_ms = statistics.mean(latencies)
    median_latency_ms = statistics.median(latencies)
    p95_latency_ms = statistics.quantiles(
        latencies, n=20
    )[18] if len(latencies) >= 20 else max(latencies)

    summary_rows.append((
        variant_name,
        avg_precision,
        avg_recall,
        avg_latency_ms,
        median_latency_ms,
        p95_latency_ms,
    ))

    print(
        f"{variant_name:20s} | avg precision={avg_precision:.4f} | "
        f"avg recall={avg_recall:.4f} | "
        f"avg latency={avg_latency_ms:8.1f} ms | "
        f"median latency={median_latency_ms:8.1f} ms | "
        f"p95 latency={p95_latency_ms:8.1f} ms"
    )

summary_df = pd.DataFrame(
    summary_rows,
    columns=[
        "variant",
        "avg_precision",
        "avg_recall",
        "avg_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
    ],
)
summary_csv_path = (
    RESULTS_DIR / f"ragas_precision_experiments_summary_{timestamp}.csv"
)
summary_df.to_csv(summary_csv_path, index=False)

print(f"\nSummary CSV written to: {summary_csv_path}")
print(f"Per-question comparison CSV written to: {comparison_csv_path}")

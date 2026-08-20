import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from embedding.huggingface_embedder import HuggingFaceEmbedder
from config import PDF_DIR, EVAL_DIR
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
from evaluation.ragas_retrieval_evaluator import RagasRetrievalEvaluator


logger = logging.getLogger(__name__)


# ============================================================
# TOP_K VALUES TO COMPARE
# ============================================================

TOP_K_VALUES = [3, 5]


# ============================================================
# RESULTS DIR
# ============================================================

RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# 1. LOAD -> CLEAN -> CHUNK -> EMBED -> STORE (once, reused
#    across every top_k value being compared)
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

retriever = VectorRetriever(vector_store)


# ============================================================
# 2. LOAD GOLDEN DATASET
# ============================================================

golden_dataset_path = (
    EVAL_DIR / "golden_dataset_with_retrieval_ground_truth.json"
)

with open(golden_dataset_path, "r", encoding="utf-8") as file:
    golden_dataset = json.load(file)


# ============================================================
# 3. QUERY EMBEDDINGS (once, reused across every top_k value)
# ============================================================

query_ids = []
query_vectors = {}

for test_case in golden_dataset["entries"]:
    query_ids.append(test_case["id"])
    query_vectors[test_case["id"]] = embedder.embed_query(
        test_case["query"]
    )


# ============================================================
# 4. RUN RAGAS EVALUATION FOR EACH TOP_K
# ============================================================

ragas_evaluator = RagasRetrievalEvaluator()

per_topk_dataframes = {}

for top_k in TOP_K_VALUES:

    print(f"\n{'=' * 80}")
    print(f"RUNNING RAGAS EVALUATION | top_k={top_k}")
    print(f"{'=' * 80}\n")

    samples = []

    for test_case in golden_dataset["entries"]:

        query_id = test_case["id"]
        query = test_case["query"]
        expected_answer = test_case["expected_answer"]

        retrieved_results = retriever.retrieve(
            query_vectors[query_id],
            top_k=top_k,
        )

        sample = ragas_evaluator.build_sample(
            query=query,
            retrieved_results=retrieved_results,
            expected_answer=expected_answer,
        )

        # Carry the golden-dataset id through so rows can be
        # matched up across different top_k runs regardless of
        # any duplicate queries.
        sample["question_id"] = query_id

        samples.append(sample)

    result = ragas_evaluator.evaluate(samples)

    result_df = result.to_pandas()
    result_df.insert(0, "question_id", [s["question_id"] for s in samples])
    result_df.insert(1, "top_k", top_k)

    csv_path = RESULTS_DIR / f"ragas_evaluation_run_topk{top_k}_{timestamp}.csv"
    result_df.to_csv(csv_path, index=False)

    print(f"\ntop_k={top_k} results written to: {csv_path}")
    print(result)

    per_topk_dataframes[top_k] = result_df


# ============================================================
# 5. BUILD COMPARISON TABLE
# ============================================================

metric_columns = [
    "llm_context_precision_with_reference",
    "context_recall",
]

baseline_k = TOP_K_VALUES[0]
comparison_df = per_topk_dataframes[baseline_k][
    ["question_id", "user_input", "reference"]
].copy()

for top_k in TOP_K_VALUES:
    df_k = per_topk_dataframes[top_k]
    for metric in metric_columns:
        comparison_df[f"{metric}_k{top_k}"] = df_k[metric].values

# ------------------------------------------------------------
# Deltas + improvement flags, relative to the smallest top_k
# ------------------------------------------------------------

other_ks = TOP_K_VALUES[1:]

for top_k in other_ks:
    for metric in metric_columns:
        base_col = f"{metric}_k{baseline_k}"
        new_col = f"{metric}_k{top_k}"
        delta_col = f"{metric}_delta_k{baseline_k}_to_k{top_k}"
        improved_col = f"{metric}_improved_k{baseline_k}_to_k{top_k}"

        comparison_df[delta_col] = (
            comparison_df[new_col] - comparison_df[base_col]
        )
        comparison_df[improved_col] = comparison_df[delta_col] > 0

comparison_csv_path = (
    RESULTS_DIR / f"ragas_topk_comparison_{timestamp}.csv"
)
comparison_df.to_csv(comparison_csv_path, index=False)

print(f"\n{'=' * 80}")
print("TOP_K COMPARISON SUMMARY")
print(f"{'=' * 80}\n")

for top_k in TOP_K_VALUES:
    df_k = per_topk_dataframes[top_k]
    print(
        f"top_k={top_k} | "
        f"avg precision={df_k['llm_context_precision_with_reference'].mean():.4f} | "
        f"avg recall={df_k['context_recall'].mean():.4f}"
    )

for top_k in other_ks:
    for metric in metric_columns:
        improved_col = f"{metric}_improved_k{baseline_k}_to_k{top_k}"
        improved_count = comparison_df[improved_col].sum()
        print(
            f"\n{metric}: improved on {improved_count}/{len(comparison_df)} "
            f"questions going from top_k={baseline_k} to top_k={top_k}"
        )

print(f"\nComparison CSV written to: {comparison_csv_path}")

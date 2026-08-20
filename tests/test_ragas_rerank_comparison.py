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
from retrieval.cross_encoder_reranker import CrossEncoderReranker
from evaluation.ragas_retrieval_evaluator import RagasRetrievalEvaluator


logger = logging.getLogger(__name__)


# ============================================================
# VARIANTS TO COMPARE
# ============================================================
# "no_rerank"  : embedding-only retrieval, top_k=5
# "reranked"   : embedding retrieves 15 candidates, cross-encoder
#                reranks, top 5 kept

TOP_K = 5
FETCH_K = 15


# ============================================================
# RESULTS DIR
# ============================================================

RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# 1. LOAD -> CLEAN -> CHUNK -> EMBED -> STORE (once, reused
#    across both variants)
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
reranking_retriever = CrossEncoderReranker(
    base_retriever=vector_retriever,
    fetch_k=FETCH_K,
)


# ============================================================
# 2. LOAD GOLDEN DATASET
# ============================================================

golden_dataset_path = (
    EVAL_DIR / "golden_dataset_with_retrieval_ground_truth.json"
)

with open(golden_dataset_path, "r", encoding="utf-8") as file:
    golden_dataset = json.load(file)


# ============================================================
# 3. QUERY EMBEDDINGS (once, reused across both variants)
# ============================================================

query_vectors = {}

for test_case in golden_dataset["entries"]:
    query_vectors[test_case["id"]] = embedder.embed_query(
        test_case["query"]
    )


# ============================================================
# 4. RUN RAGAS EVALUATION FOR EACH VARIANT
# ============================================================

ragas_evaluator = RagasRetrievalEvaluator()

variants = {
    "no_rerank": vector_retriever,
    "reranked": reranking_retriever,
}

per_variant_dataframes = {}

for variant_name, active_retriever in variants.items():

    print(f"\n{'=' * 80}")
    print(f"RUNNING RAGAS EVALUATION | variant={variant_name}")
    print(f"{'=' * 80}\n")

    samples = []

    for test_case in golden_dataset["entries"]:

        query_id = test_case["id"]
        query = test_case["query"]
        expected_answer = test_case["expected_answer"]
        query_vector = query_vectors[query_id]

        if variant_name == "reranked":
            retrieved_results = active_retriever.retrieve(
                query_vector,
                top_k=TOP_K,
                query_text=query,
            )
        else:
            retrieved_results = active_retriever.retrieve(
                query_vector,
                top_k=TOP_K,
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

    csv_path = (
        RESULTS_DIR
        / f"ragas_evaluation_run_{variant_name}_{timestamp}.csv"
    )
    result_df.to_csv(csv_path, index=False)

    print(f"\nvariant={variant_name} results written to: {csv_path}")
    print(result)

    per_variant_dataframes[variant_name] = result_df


# ============================================================
# 5. BUILD COMPARISON TABLE
# ============================================================

metric_columns = [
    "llm_context_precision_with_reference",
    "context_recall",
]

baseline_variant = "no_rerank"
compared_variant = "reranked"

comparison_df = per_variant_dataframes[baseline_variant][
    ["question_id", "user_input", "reference"]
].copy()

for variant_name in variants:
    df_v = per_variant_dataframes[variant_name]
    for metric in metric_columns:
        comparison_df[f"{metric}_{variant_name}"] = df_v[metric].values

for metric in metric_columns:
    base_col = f"{metric}_{baseline_variant}"
    new_col = f"{metric}_{compared_variant}"
    delta_col = f"{metric}_delta"
    improved_col = f"{metric}_improved"

    comparison_df[delta_col] = comparison_df[new_col] - comparison_df[base_col]
    comparison_df[improved_col] = comparison_df[delta_col] > 0

comparison_csv_path = (
    RESULTS_DIR / f"ragas_rerank_comparison_{timestamp}.csv"
)
comparison_df.to_csv(comparison_csv_path, index=False)

print(f"\n{'=' * 80}")
print("RERANK COMPARISON SUMMARY")
print(f"{'=' * 80}\n")

for variant_name in variants:
    df_v = per_variant_dataframes[variant_name]
    print(
        f"{variant_name} | "
        f"avg precision={df_v['llm_context_precision_with_reference'].mean():.4f} | "
        f"avg recall={df_v['context_recall'].mean():.4f}"
    )

for metric in metric_columns:
    improved_col = f"{metric}_improved"
    improved_count = comparison_df[improved_col].sum()
    print(
        f"\n{metric}: improved on {improved_count}/{len(comparison_df)} "
        f"questions with reranking vs. no_rerank"
    )

print(f"\nComparison CSV written to: {comparison_csv_path}")

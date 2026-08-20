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
from retrieval.default_retriever import build_default_retriever
from evaluation.retrieval_metrics import RetrievalMetrics


logger = logging.getLogger(__name__)


# ============================================================
# This test needs no LLM judge - RetrievalMetrics is deterministic
# substring matching against the golden dataset's evidence text
# (same approach RAGEvaluator/hit_at_k already uses), so it's free
# and fast to run, unlike the ragas test scripts.
# ============================================================

K_VALUES = (1, 3, 5, 10)
NDCG_K = 10
FETCH_TOP_K = max(K_VALUES + (NDCG_K,))  # retrieve enough for every cutoff


# ============================================================
# RESULTS DIR
# ============================================================

RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# 1. LOAD -> CLEAN -> CHUNK -> EMBED -> STORE
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

# Project default retriever (BM25 + embedding hybrid, L-12 rerank,
# score threshold) - see retrieval/default_retriever.py
retriever = build_default_retriever(vector_store)


# ============================================================
# 2. LOAD GOLDEN DATASET
# ============================================================

golden_dataset_path = (
    EVAL_DIR / "golden_dataset_with_retrieval_ground_truth.json"
)

with open(golden_dataset_path, "r", encoding="utf-8") as file:
    golden_dataset = json.load(file)


# ============================================================
# 3. RUN RETRIEVAL + COMPUTE METRICS
# ============================================================

metrics_calculator = RetrievalMetrics()

rows = []

for test_case in golden_dataset["entries"]:

    question_id = test_case["id"]
    query = test_case["query"]
    retrieval_ground_truth = test_case["retrieval_ground_truth"]

    query_vector = embedder.embed_query(query)

    retrieved_results = retriever.retrieve(
        query_vector,
        top_k=FETCH_TOP_K,
        query_text=query,
    )

    metrics = metrics_calculator.compute(
        retrieved_results=retrieved_results,
        retrieval_ground_truth=retrieval_ground_truth,
        k_values=K_VALUES,
        ndcg_k=NDCG_K,
    )

    row = {
        "question_id": question_id,
        "user_input": query,
        "retrieved_count": len(retrieved_results),
    }
    row.update(metrics)
    rows.append(row)

    print(
        f"{question_id:8s} | "
        + " | ".join(
            f"R@{k}={metrics[f'recall_at_{k}']:.2f}" for k in K_VALUES
        )
        + f" | MRR={metrics['mrr']:.2f}"
        + f" | NDCG@{NDCG_K}={metrics[f'ndcg_at_{NDCG_K}']:.2f}"
    )


# ============================================================
# 4. SAVE + SUMMARIZE
# ============================================================

results_df = pd.DataFrame(rows)

csv_path = RESULTS_DIR / f"retrieval_metrics_{timestamp}.csv"
results_df.to_csv(csv_path, index=False)

print(f"\n{'=' * 80}")
print("RETRIEVAL METRICS SUMMARY (classical IR metrics, non-LLM)")
print(f"{'=' * 80}\n")

metric_columns = [f"recall_at_{k}" for k in K_VALUES] + [
    "mrr",
    f"ndcg_at_{NDCG_K}",
]

for column in metric_columns:
    print(f"avg {column:16s} = {results_df[column].mean():.4f}")

print(f"\nPer-question metrics written to: {csv_path}")

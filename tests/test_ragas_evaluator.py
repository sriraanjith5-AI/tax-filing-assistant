import sys
import json
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from embedding.huggingface_embedder import HuggingFaceEmbedder
from config import PDF_DIR, EVAL_DIR, TOP_K
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
from evaluation.ragas_retrieval_evaluator import RagasRetrievalEvaluator


logger = logging.getLogger(__name__)


# ============================================================
# 0. RESULTS FILE SETUP
# ============================================================

RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_output_path = RESULTS_DIR / f"ragas_evaluation_run_{timestamp}.csv"


# ============================================================
# 1. LOAD -> CLEAN -> CHUNK -> EMBED -> STORE
# ============================================================
# Same pipeline as test_evaluator.py, so both evaluators run
# against an identical retrieval setup.

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

# Default retriever: BM25 + embedding fused (HybridRetriever),
# reranked with the stronger L-12 cross-encoder, weak candidates
# dropped via score_threshold. This is the "combined" variant from
# evaluation/results/ragas_precision_experiments_summary_*.csv - best
# precision AND recall of everything tried (avg precision 0.9066,
# avg recall 1.0000 vs. plain embedding+L-6 rerank's 0.8541/0.9222).

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
# 3. RUN RETRIEVAL + BUILD RAGAS SAMPLES
# ============================================================

ragas_evaluator = RagasRetrievalEvaluator()

top_k = TOP_K
samples = []

for test_case in golden_dataset["entries"]:

    query = test_case["query"]
    expected_answer = test_case["expected_answer"]

    query_vector = embedder.embed_query(query)

    retrieved_results = retriever.retrieve(
        query_vector,
        top_k=top_k,
        query_text=query,
    )

    sample = ragas_evaluator.build_sample(
        query=query,
        retrieved_results=retrieved_results,
        expected_answer=expected_answer,
    )

    samples.append(sample)

    logger.info(
        "Built ragas sample for %s | retrieved_contexts=%s",
        test_case["id"],
        len(sample["retrieved_contexts"]),
    )


# ============================================================
# 4. RUN RAGAS EVALUATION
# ============================================================

print(f"\nRunning ragas evaluation over {len(samples)} samples "
      f"(judge model: {ragas_evaluator.judge_llm.langchain_llm.model_name})...\n")

result = ragas_evaluator.evaluate(samples)

print(result)

result_df = result.to_pandas()
result_df.to_csv(csv_output_path, index=False)

print(f"\nRagas evaluation results written to: {csv_output_path}")

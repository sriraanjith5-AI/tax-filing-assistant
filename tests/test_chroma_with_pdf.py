import sys
from pathlib import Path
from datetime import datetime

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
from vectorstore.chroma_vector_store import ChromaVectorStore

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

chroma = ChromaVectorStore()
response=chroma.add(embedding_response.successful_embeddings)
if response.total_received_chunks != 0:
    print("Embeddings added successfully.")
    print(f"Total stored chunks: {response.total_stored_chunks}") 
    print(f"Total skipped chunks: {response.total_skipped_chunks}") 
    print(f"Total failed chunks: {response.total_failed_chunks}") 
    print(f"Count is: {chroma.collection.count()}")

else:
    print("Failed to add embeddings.")

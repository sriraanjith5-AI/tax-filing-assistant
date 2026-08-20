from pathlib import Path

# ----------------------------
# Project Paths
# ----------------------------

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"

PDF_DIR = DATA_DIR / "pdf"

CSV_DIR = DATA_DIR / "csv"

EXCEL_DIR = DATA_DIR / "excel"

VECTOR_DB_DIR = BASE_DIR / "vectorstore"

EVAL_DIR = BASE_DIR / "evaluation"


# ----------------------------
# Chunking
# ----------------------------

CHUNK_SIZE = 700

CHUNK_OVERLAP = 120

CHUNK_MIN_SIZE = 200

# ----------------------------
# Embedding Model
# ----------------------------

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 2


# ----------------------------
# Retrieval
# ----------------------------

TOP_K = 5


# ----------------------------
# Reranking
# ----------------------------
# CrossEncoderReranker is the default retriever: VectorRetriever
# fetches RERANK_FETCH_K wide candidates by embedding similarity,
# then the cross-encoder reranks and truncates to TOP_K.
# Confirmed via evaluation/results/ragas_rerank_comparison_*.csv:
# avg precision 0.7587 -> 0.8508, avg recall 0.8889 -> 0.9222.

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_FETCH_K = 15

# Stronger (larger) cross-encoder, tried as a precision experiment
# against RERANK_MODEL. See evaluation/results/ragas_precision_
# experiments_*.csv for the comparison.
RERANK_MODEL_STRONG = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Minimum sigmoid(cross-encoder score) a candidate must clear to be
# kept, when using CrossEncoderReranker's score_threshold option
# instead of a fixed top_k cutoff.
RERANK_SCORE_THRESHOLD = 0.5

# Number of LLM-generated paraphrases QueryExpander produces per
# query for QueryExpansionRetriever.
QUERY_EXPANSION_VARIANTS = 2


# ----------------------------
# Ragas Evaluation (LLM judge)
# ----------------------------

# Model used as the ragas judge for LLM-based context metrics
# (LLMContextPrecisionWithReference, LLMContextRecall).
# Requires OPENAI_API_KEY to be set (e.g. in a .env file).
RAGAS_JUDGE_MODEL = "gpt-4o-mini"
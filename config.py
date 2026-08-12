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


# ----------------------------
# Retrieval
# ----------------------------

TOP_K = 5
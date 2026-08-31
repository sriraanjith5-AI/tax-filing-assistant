from pathlib import Path

from config import CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_DB_DIR
from chunking.base_chunker import BaseChunker
from chunking.recursive_chunker import RecursiveChunker
from chunking.semantic_chunker import SemanticChunker
from ingestion.base_loader import BaseDocumentLoader
from ingestion.pdfloader import PDFLoader
from vectorstore.base_vector_store import BaseVectorStore
from vectorstore.chroma_vector_store import ChromaVectorStore, DEFAULT_COLLECTION_NAME
from vectorstore.faiss_vector_store import FaissVectorStore
from retrieval.default_retriever import build_retriever

CHUNKERS = {
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
}

VECTOR_STORES = ("chroma", "faiss")

RETRIEVERS = ("vector", "bm25", "hybrid", "hybrid_reranked", "hybrid_reranked_expanded")

# Every loader but "pypdf" is imported lazily inside make_loader() - each
# pulls in its own dependency tree (unstructured: spacy/numba/pdfminer;
# pymupdf4llm/pdfplumber are lighter but still needless for callers who
# never touch them) - so importing them eagerly here would slow down
# every `import ingestion.registry` regardless of which loader is
# actually used.
#
# Relative speed on this project's corpus (71-page IRS PDF, CPU only):
# pypdf ~12s, unstructured(fast) ~36s, pdfplumber ~23s, pymupdf4llm ~114s.
# Retrieval-quality differences between them were measured as within
# noise on the golden dataset - see evaluation/results/loader_comparison_*.csv.
#
# Docling (layout+table ML models) was tried and dropped: CPU-only
# conversion took minutes per document with no reliable upper bound (one
# page stalled a full-corpus run for 90+ minutes under load), which isn't
# a workable cost for this project's default ingestion path.
LOADERS = ("pypdf", "pymupdf4llm", "pdfplumber", "unstructured")


def make_loader(name: str, pdf_directory: Path) -> BaseDocumentLoader:
    if name == "pypdf":
        return PDFLoader(pdf_directory)
    if name == "pymupdf4llm":
        from ingestion.pymupdf4llm_loader import PyMuPDF4LLMLoader
        return PyMuPDF4LLMLoader(pdf_directory)
    if name == "pdfplumber":
        from ingestion.pdfplumber_loader import PDFPlumberLoader
        return PDFPlumberLoader(pdf_directory)
    if name == "unstructured":
        from ingestion.unstructured_loader import UnstructuredFastLoader
        return UnstructuredFastLoader(pdf_directory)
    raise ValueError(f"Unknown loader '{name}'. Valid options: {LOADERS}")


def make_chunker(
    name: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> BaseChunker:
    if name not in CHUNKERS:
        raise ValueError(f"Unknown chunker '{name}'. Valid options: {list(CHUNKERS)}")

    if name == "recursive":
        return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # chunk_size/chunk_overlap don't apply to SemanticChunker (boundaries
    # come from embedding-similarity percentile, not a fixed length) -
    # silently ignored rather than erroring, so the UI can leave the
    # fields at their default without special-casing the request.
    return SemanticChunker()


def variant_key(
    chunker_name: str,
    chunk_size: int,
    chunk_overlap: int,
    loader_name: str = "pypdf",
) -> str:
    """Encodes every parameter that changes chunk content, so two runs
    that differ in loader, chunker, or chunk_size/chunk_overlap get
    isolated vector store collections/indexes instead of reconciling
    (deleting) each other's chunks on every run. Different loaders
    extract different text for the "same" page (see
    ingestion/pymupdf4llm_loader.py vs ingestion/pdfloader.py), so
    they're just as content-changing as a different chunker.

    loader_name defaults to "pypdf" (the project's original loader) so
    existing pypdf-variant collections/indexes keep the same key as
    before this parameter was added - no forced re-ingestion for callers
    that don't pass it.
    """
    loader_prefix = "" if loader_name == "pypdf" else f"{loader_name}__"
    if chunker_name == "semantic":
        return f"{loader_prefix}semantic"
    return f"{loader_prefix}{chunker_name}__cs{chunk_size}_co{chunk_overlap}"


def make_vector_store(name: str, variant: str) -> BaseVectorStore:
    if name == "chroma":
        return ChromaVectorStore(collection_name=f"{DEFAULT_COLLECTION_NAME}__{variant}")
    if name == "faiss":
        return FaissVectorStore(index_dir=str(VECTOR_DB_DIR / "faiss_data" / variant))
    raise ValueError(f"Unknown vector store '{name}'. Valid options: {VECTOR_STORES}")


__all__ = [
    "CHUNKERS",
    "VECTOR_STORES",
    "RETRIEVERS",
    "LOADERS",
    "make_loader",
    "make_chunker",
    "variant_key",
    "make_vector_store",
    "build_retriever",
]

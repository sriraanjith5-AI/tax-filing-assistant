from config import PDF_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from ingestion.cleaner_pipeline import CleanerPipeline
from ingestion.reconciler import reconcile_source
from ingestion.registry import make_chunker, make_vector_store, make_loader, variant_key
from cleaners.header_cleaner import HeaderCleaner
from cleaners.footer_cleaner import FooterCleaner
from cleaners.whitespace_cleaner import WhitespaceCleaner
from embedding.huggingface_embedder import HuggingFaceEmbedder
from embedding.embedding_dataclass import EmbeddingConfig
from utils.logger import logging

logger = logging.getLogger(__name__)


def build_index(
    chunker_name: str,
    vectorstore_name: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    loader_name: str = "pypdf",
    fresh: bool = False,
):
    """
    End-to-end ingestion pipeline: Load -> Clean -> Chunk -> Embed ->
    Store -> Reconcile, wiring together the project's existing pieces
    for a given loader/chunker/vector-store configuration. Used by the
    comparison UI (evaluation/run_comparison.py) to build a fresh (or
    incrementally-updated) index per config before evaluating it.

    loader_name : one of ingestion.registry.LOADERS - "pypdf" (default,
    langchain's PyPDFLoader - fast, plain text-stream extraction),
    "pymupdf4llm", "pdfplumber", or "unstructured" (each parses more
    carefully - headings/lists/tables - at extra ingest cost; see their
    docstrings under ingestion/). Different loaders extract different
    text for the same PDF, so they get isolated vector store
    collections/indexes (variant_key includes loader_name) rather than
    reconciling against each other's chunks.

    Content-addressed chunk_id + store.add()'s skip-on-existing
    behavior mean repeat runs over an unchanged corpus are cheap - only
    genuinely new/changed chunks get inserted; reconcile_source removes
    anything stale per source afterwards. This keeps re-runs fast, but
    it's a best-effort reconciliation - it only catches drift it can
    still recognize as "the same source" (see
    ingestion/base_loader.py::normalize_source's docstring for one class
    of drift it used to miss entirely).

    fresh : if True, wipes the target collection/index before
    re-ingesting, so the run starts from a guaranteed-empty store and
    the result can never contain more than one copy of any chunk -
    stronger than reconcile's best-effort cleanup, at the cost of
    re-embedding the whole corpus every call instead of only what
    changed. Used by the /ask endpoint's startup warm-up (see
    retrieval/query_pipeline.py) so every app restart starts from a
    known-clean index; left off (default) for the comparison UI, which
    deliberately reuses cached embeddings across many experiment runs.

    Returns (vector_store, embedder, add_response) so callers can build
    a retriever over the populated store and reuse the same embedder
    for query embedding.
    """

    chunker = make_chunker(chunker_name, chunk_size, chunk_overlap)
    variant = variant_key(chunker_name, chunk_size, chunk_overlap, loader_name)
    store = make_vector_store(vectorstore_name, variant)

    if fresh:
        store.clear()

    loaded_docs = make_loader(loader_name, PDF_DIR).load_documents()
    if not loaded_docs:
        raise RuntimeError("No documents were loaded from PDF_DIR.")

    cleaner_pipeline = CleanerPipeline(
        cleaners=[HeaderCleaner(), FooterCleaner(), WhitespaceCleaner()]
    )
    clean_docs = cleaner_pipeline.clean(loaded_docs)

    chunks = chunker.chunk(clean_docs)
    if not chunks:
        raise RuntimeError("Chunking produced no chunks.")

    embedder = HuggingFaceEmbedder(EmbeddingConfig())
    embedding_response = embedder.embed(chunks)

    add_response = store.add(embedding_response.successful_embeddings)

    sources = {chunk.metadata["source"] for chunk in chunks}
    for source in sources:
        source_chunk_ids = {
            chunk.metadata["chunk_id"]
            for chunk in chunks
            if chunk.metadata["source"] == source
        }
        deleted = reconcile_source(store, source, source_chunk_ids)
        if deleted:
            logger.info(f"Reconciled {source}: removed {deleted} stale chunk(s).")

    return store, embedder, add_response

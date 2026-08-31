from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from langchain_core.documents import Document


class BaseDocumentLoader(ABC):
    """
    Parses PDFs in a directory into one langchain Document per page.

    Every downstream stage (cleaners, chunkers, chunk-position metadata,
    ContextExpandingRetriever's neighbor lookup) depends on this
    one-Document-per-page shape and on metadata["source"]/metadata["page"]
    being present - see chunking/recursive_chunker.py and
    chunking/semantic_chunker.py, which read doc.metadata['source']/['page']
    directly, and retrieval/context_expander.py, whose neighbor index is
    keyed on (source, page, chunk_number). A loader implementation must
    preserve that contract even if its underlying parser doesn't naturally
    work page-by-page (see ingestion/unstructured_loader.py, which groups
    Unstructured's fine-grained elements back into per-page Documents).
    """

    def __init__(self, pdf_directory: Path):
        self.pdf_directory = pdf_directory

    @staticmethod
    def normalize_source(pdf_file: Path) -> str:
        """Canonical string to store as metadata['source'].

        chunk_id is content-addressed as sha256(source|chunker|content)
        (see BaseChunker.generate_chunk_id) - on Windows, the same file
        can be reported with different drive-letter/path casing
        ("C:\\..." vs "c:\\...") depending on how the process/shell that
        launched ingestion resolved its working directory, which silently
        produces a *different* chunk_id for genuinely identical content.
        That doesn't error - it just quietly re-embeds and re-stores a
        duplicate, bloating the vector store and cluttering every
        downstream retrieval with doubled-up context (found by tracing
        why a generation answer looked wrong despite the right chunk
        being retrieved - see evaluation/results/loader_comparison_*).
        Lower-casing here (safe - Windows paths are case-insensitive)
        makes chunk_id stable across invocations regardless of casing.
        """
        return str(pdf_file).lower()

    @abstractmethod
    def load_documents(self) -> List[Document]:
        """Returns one Document per PDF page, in file/page order, with at
        least metadata['source'] (the PDF path) and metadata['page']
        (0-indexed) set."""
        pass

from pathlib import Path
from typing import List

from langchain_core.documents import Document

from ingestion.base_loader import BaseDocumentLoader
from utils.logger import logging

logger = logging.getLogger(__name__)


class PyMuPDF4LLMLoader(BaseDocumentLoader):
    """
    Parses PDFs via pymupdf4llm (PyMuPDF's Markdown-oriented extraction) -
    one Document per page, text as Markdown (headings/lists/tables
    rendered with Markdown syntax where PyMuPDF's heuristics detect them),
    instead of PyPDFLoader's plain text-stream extraction.

    Pure PyMuPDF heuristics, no ML layout/table model - ~1.6s/page
    measured on this corpus (~2 min for the full 71-page document, with
    no per-page cost blowups observed) while still picking up basic
    table structure via Markdown table syntax. Slowest of this project's
    loaders in absolute terms, but reliably bounded.

    ignore_images/ignore_graphics=True: we only want page text for
    chunking, and leaving these on can trigger pymupdf4llm's OCR fallback
    for image regions (rapidocr) - unnecessary cost for a RAG text
    pipeline that never uses the extracted images.
    """

    def load_documents(self) -> List[Document]:
        import pymupdf4llm

        documents = []
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        logger.info(f"\n Found {len(pdf_files)} PDF files\n")

        for pdf_file in pdf_files:
            logger.info(f"\n pdf file name is {pdf_file}\n")
            pages = pymupdf4llm.to_markdown(
                str(pdf_file),
                page_chunks=True,
                ignore_images=True,
                ignore_graphics=True,
            )
            for page in pages:
                page_number = page["metadata"]["page_number"]  # 1-indexed
                total_pages = page["metadata"].get("page_count", len(pages))
                documents.append(
                    Document(
                        page_content=page.get("text", ""),
                        metadata={
                            "source": self.normalize_source(pdf_file),
                            "page": page_number - 1,
                            "total_pages": total_pages,
                        },
                    )
                )
            logger.info(f"Loaded {len(pages)} pages")

        logger.info(f"\nTotal Pages Loaded : {len(documents)}\n")
        return documents

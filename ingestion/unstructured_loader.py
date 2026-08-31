from pathlib import Path
from typing import List

from langchain_core.documents import Document

from ingestion.base_loader import BaseDocumentLoader
from utils.logger import logging

logger = logging.getLogger(__name__)


class UnstructuredFastLoader(BaseDocumentLoader):
    """
    Parses PDFs via Unstructured's "fast" strategy (partition_pdf(...,
    strategy="fast")) - pdfminer.six-based text/layout extraction, no
    OCR/vision model, unlike Unstructured's "hi_res"/"ocr_only" strategies.
    ~0.5s/page measured on this corpus (36s for the full 71-page document).

    Unstructured partitions a PDF into many fine-grained elements (Title,
    NarrativeText, ListItem, Table, Header/Footer, ...) rather than
    pymupdf4llm's whole-page blocks, each tagged with
    element.metadata.page_number (1-indexed) - these are grouped back
    into one Document per page here, joined in reading order, to match
    the project's one-Document-per-page contract (see
    ingestion/base_loader.py).
    """

    def load_documents(self) -> List[Document]:
        from unstructured.partition.pdf import partition_pdf

        documents = []
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        logger.info(f"\n Found {len(pdf_files)} PDF files\n")

        for pdf_file in pdf_files:
            logger.info(f"\n pdf file name is {pdf_file}\n")
            elements = partition_pdf(filename=str(pdf_file), strategy="fast")

            page_texts: dict = {}
            max_page = 0
            for element in elements:
                page_number = getattr(element.metadata, "page_number", None) or 1
                page_texts.setdefault(page_number, []).append(str(element))
                max_page = max(max_page, page_number)

            for page_number in range(1, max_page + 1):
                documents.append(
                    Document(
                        page_content="\n".join(page_texts.get(page_number, [])),
                        metadata={
                            "source": self.normalize_source(pdf_file),
                            "page": page_number - 1,
                            "total_pages": max_page,
                        },
                    )
                )
            logger.info(f"Loaded {max_page} pages")

        logger.info(f"\nTotal Pages Loaded : {len(documents)}\n")
        return documents

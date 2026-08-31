from pathlib import Path
from typing import List

from langchain_core.documents import Document

from ingestion.base_loader import BaseDocumentLoader
from utils.logger import logging

logger = logging.getLogger(__name__)


class PDFPlumberLoader(BaseDocumentLoader):
    """
    Parses PDFs via pdfplumber - one Document per page, text via
    extract_text() (character-clustering-based layout analysis; no ML
    models). Pure Python, CPU-light, no OCR - reads layout more carefully
    than PyPDFLoader's raw text-stream extraction, at modest extra cost
    (~23s for the full 71-page document on this corpus).
    """

    def load_documents(self) -> List[Document]:
        import pdfplumber

        documents = []
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        logger.info(f"\n Found {len(pdf_files)} PDF files\n")

        for pdf_file in pdf_files:
            logger.info(f"\n pdf file name is {pdf_file}\n")
            with pdfplumber.open(str(pdf_file)) as pdf:
                total_pages = len(pdf.pages)
                for page_index, page in enumerate(pdf.pages):
                    documents.append(
                        Document(
                            page_content=page.extract_text() or "",
                            metadata={
                                "source": self.normalize_source(pdf_file),
                                "page": page_index,
                                "total_pages": total_pages,
                            },
                        )
                    )
            logger.info(f"Loaded {total_pages} pages")

        logger.info(f"\nTotal Pages Loaded : {len(documents)}\n")
        return documents

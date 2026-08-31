from langchain_community.document_loaders import PyPDFLoader
import logging

from ingestion.base_loader import BaseDocumentLoader

logger = logging.getLogger(__name__)


class PDFLoader(BaseDocumentLoader):
    """Parses PDFs via langchain_community's PyPDFLoader (pypdf under the
    hood) - one Document per page, text extracted in reading order. The
    project's original/default loader; see ingestion/registry.py's
    LOADERS for the alternatives (pymupdf4llm, pdfplumber, unstructured)."""

    def load_documents(self) -> list:
        documents=[]
        pdf_files=list(self.pdf_directory.glob("*.pdf"))
        logger.info(f"\n Found {len(pdf_files)} PDF files\n")
        logger.info(f"\n Type of var pdf_files is {type(pdf_files)}\n")

        for pdf_file in pdf_files:
            logger.info(f"\n pdf file name is {pdf_file}\n")
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = self.normalize_source(pdf_file)
            logger.info(f"Loaded {len(docs)} pages")
            logger.info(f"Type of var - docs is {type(docs)}")
            documents.extend(docs)
        logger.info(f"\nTotal Pages Loaded : {len(documents)}\n")
        return documents
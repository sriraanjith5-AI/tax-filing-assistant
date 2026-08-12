from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
import logging

logger = logging.getLogger(__name__)

logger.info("Loading IncomeTax.pdf")


class PDFLoader:
    def __init__(self, pdf_directory: Path):
        self.pdf_directory = pdf_directory

    def load_documents(self):
        documents=[]
        pdf_files=list(self.pdf_directory.glob("*.pdf"))
        logger.info(f"\n Found {len(pdf_files)} PDF files\n")
        logger.info(f"\n Type of var pdf_files is {type(pdf_files)}\n")

        for pdf_file in pdf_files:
            logger.info(f"\n pdf file name is {pdf_file}\n")
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} pages")
            logger.info(f"Type of var - docs is {type(docs)}")
            documents.extend(docs)
        logger.info(f"\nTotal Pages Loaded : {len(documents)}\n")
        return documents
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import PDF_DIR
from ingestion.pdfloader import PDFLoader
from ingestion.cleaner_pipeline import CleanerPipeline
#from cleaners.dummy_cleaner import DummyCleaner
from cleaners.header_cleaner import HeaderCleaner
from cleaners.footer_cleaner import FooterCleaner
from cleaners.whitespace_cleaner import WhitespaceCleaner
from chunking.recursive_chunker import RecursiveChunker
from utils.logger import logging

logger = logging.getLogger(__name__)

logger.info(f"\n PDF_DIR is {PDF_DIR}\n")
logger.info(list(PDF_DIR.iterdir()))
pdf_loader = PDFLoader(PDF_DIR)
loaded_docs=pdf_loader.load_documents()
logger.info(f"\n len of loaded_docs is {len(loaded_docs)}\n")
logger.info(f"first 100 char of loaded_docs are{loaded_docs[0].page_content[:500]}")

logger.info("testing pdfloader.py is completed")

print("========== BEFORE ==========")
print(loaded_docs[0].page_content[:500])


#pipeline = CleanerPipeline( cleaners=[ DummyCleaner() ] )
pipeline = CleanerPipeline( cleaners=[ HeaderCleaner(),
                                       FooterCleaner(),
                                       WhitespaceCleaner() ] 
                                       )

clean_doc=pipeline.clean(loaded_docs)
print("Cleaning is completed")  
#print(logger.info(clean_doc[0].page_content[-100:]))
print("========== AFTER ==========")
print(loaded_docs[0].page_content[:500])

assert len(clean_doc) == len(loaded_docs)

assert id(clean_doc[0]) == id(loaded_docs[0])

print("Entire Process is completed")  

logger.info("Chunking Process Started")
chunking=RecursiveChunker()
final_chunks=chunking.chunk(loaded_docs[:2])
logger.info(f"Length of Final Chunks is {len(final_chunks)}")

for index,chunk in enumerate(final_chunks):
    logger.info(
        "Chunk %s | length=%s | source=%s | page=%s | total_chunks=%s",
        index + 1,
        len(chunk.page_content),
        chunk.metadata.get("source"),
        chunk.metadata.get("page"),
        chunk.metadata.get("total_chunks")
    )
logger.info("Chunking Process Completed")

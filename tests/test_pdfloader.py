import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from embedding.huggingface_embedder import HuggingFaceEmbedder
from langchain_core.documents import Document
from config import PDF_DIR
from ingestion.pdfloader import PDFLoader
from ingestion.cleaner_pipeline import CleanerPipeline
#from cleaners.dummy_cleaner import DummyCleaner
from cleaners.header_cleaner import HeaderCleaner
from cleaners.footer_cleaner import FooterCleaner
from cleaners.whitespace_cleaner import WhitespaceCleaner
from chunking.recursive_chunker import RecursiveChunker
from utils.logger import logging
from embedding.embedding_dataclass import EmbeddingConfig
from vectorstore.fake_vector_store import FakeVectorStore
from retrieval.vector_retriever import VectorRetriever

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

pipeline = CleanerPipeline( cleaners=[ HeaderCleaner(),
                                       FooterCleaner(),
                                       WhitespaceCleaner() ] 
                                       )

clean_doc=pipeline.clean(loaded_docs)
print("Cleaning is completed")  

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
        "Chunk %s | length=%s | source=%s | page=%s | chunk_number=%s| chunk_id=%s | total_chunks=%s",
        index + 1,
        len(chunk.page_content),
        chunk.metadata.get("source"),
        chunk.metadata.get("page"),
        chunk.metadata.get("chunk_number"),
        chunk.metadata.get("chunk_id"),
        chunk.metadata.get("total_chunks")
    )

logger.info("Chunking Process Completed")

embedding_config = EmbeddingConfig()
embedder = HuggingFaceEmbedder(embedding_config)
embedding_response = embedder.embed(final_chunks)

assert embedding_response.embed_status == "SUCCESS"
assert len(embedding_response.successful_embeddings) == len(final_chunks)

vector_store=FakeVectorStore()
store_response = vector_store.add(
    embedding_response.successful_embeddings
)
assert store_response.total_received_chunks == len(final_chunks)
assert store_response.total_stored_chunks == len(final_chunks)
assert store_response.total_skipped_chunks == 0
assert store_response.total_failed_chunks == 0


retriever = VectorRetriever(vector_store)

query="How is federal income tax withholding calculated?"
query_vector = embedder.embed_query(query)

results = retriever.retrieve(
    query_vector,
    top_k=2
)

for result in results:
    assert result.document is not None
    assert result.document.metadata.get("chunk_id") is not None
    assert result.score is not None

print("\n========== RETRIEVAL RESULTS ==========")

for rank, result in enumerate(results, start=1):
    print(
        f"\nRank       : {rank}"
        f"\nChunk ID   : {result.document.metadata.get('chunk_id')}"
        f"\nPage       : {result.document.metadata.get('page')}"
        f"\nScore      : {result.score:.4f}"
        f"\nContent    : {result.document.page_content[:300]}"
    )
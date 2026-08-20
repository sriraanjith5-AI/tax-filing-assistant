import sys,json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from embedding.huggingface_embedder import HuggingFaceEmbedder
from langchain_core.documents import Document
from config import PDF_DIR,EVAL_DIR
from ingestion.pdfloader import PDFLoader
from ingestion.cleaner_pipeline import CleanerPipeline
from cleaners.header_cleaner import HeaderCleaner
from cleaners.footer_cleaner import FooterCleaner
from cleaners.whitespace_cleaner import WhitespaceCleaner
from chunking.recursive_chunker import RecursiveChunker
from utils.logger import logging
from embedding.embedding_dataclass import EmbeddingConfig
from vectorstore.fake_vector_store import FakeVectorStore
from retrieval.vector_retriever import VectorRetriever
from evaluation.rag_evaluator import RAGEvaluator


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
evaluator=RAGEvaluator()

results=[]
# NOTE: golden_dataset.json moved to experiments/evaluation/ - this is a
# superseded evaluator version, kept for reference only (see
# evaluation/RETRIEVAL_EVALUATION_JOURNAL.md for the current setup).
golden_dataset_path = PROJECT_ROOT / "experiments" / "evaluation" / "golden_dataset.json"

with open(golden_dataset_path, "r", encoding="utf-8") as file:
    golden_dataset = json.load(file)

for test_case in golden_dataset["entries"]:

    question_id = test_case["id"]
    query = test_case["query"]
    expected_answer = test_case["expected_answer"]

    query_vector = embedder.embed_query(query)

    retrieved_results = retriever.retrieve(
        query_vector,
        top_k=3
    )
    print("\n========================================")
    print(f"Question ID : {question_id}")
    print(f"Query       : {query}")
    print("========================================")

    print("\n========== RETRIEVAL RESULTS ==========")

    for rank, retrieved_result in enumerate(
        retrieved_results,
            start=1
):
        print(f"\nRank       : {rank}")
        print(
            f"Chunk ID   : "
            f"{retrieved_result.document.metadata.get('chunk_id')}"
        )
        print(
            f"Page       : "
            f"{retrieved_result.document.metadata.get('page')}"
        )
        print(
            f"Score      : "
            f"{retrieved_result.score:.4f}"
        )
        print(
            f"Content    : "
            f"{retrieved_result.document.page_content[:500]}"
        )

    for retrieved_result in retrieved_results:
        assert retrieved_result.document is not None
        assert retrieved_result.document.metadata.get("chunk_id") is not None
        assert retrieved_result.score >= 0
    
    evaluation_result = evaluator.evaluate_retrieval(
        question_id=question_id,
        query=query,
        expected_answer=expected_answer,
        retrieved_results=retrieved_results
    )

    print("\n========== EVALUATION RESULT ==========")

    print(f"Question ID       : {evaluation_result.question_id}")
    print(f"Retrieval Success : {evaluation_result.retrieval_success}")
    print(f"Retrieved Count   : {evaluation_result.retrieved_count}")
    print(f"Top Score         : {evaluation_result.top_score:.4f}")
    print(f"Status            : {evaluation_result.evaluation_status}")

    assert evaluation_result.question_id == question_id
    assert evaluation_result.query == query
    assert evaluation_result.expected_answer == expected_answer
    assert evaluation_result.retrieved_count == len(retrieved_results)

    results.append(evaluation_result)

print("\n========== FINAL EVALUATION SUMMARY ==========")

total_questions = len(results)

successful_retrievals = sum(
    1
    for result in results
    if result.retrieval_success
)

print(f"Total Questions      : {total_questions}")
print(f"Successful Retrieval : {successful_retrievals}")
print(
    f"Failed Retrieval     : "
    f"{total_questions - successful_retrievals}"
)

if total_questions > 0:
    retrieval_rate = (
        successful_retrievals / total_questions
    ) * 100

    print(
        f"Retrieval Success Rate : "
        f"{retrieval_rate:.2f}%"
    )


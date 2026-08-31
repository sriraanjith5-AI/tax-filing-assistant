import sys
import json
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from embedding.huggingface_embedder import HuggingFaceEmbedder
from config import PDF_DIR, EVAL_DIR
from ingestion.pdfloader import PDFLoader
from ingestion.cleaner_pipeline import CleanerPipeline
from cleaners.header_cleaner import HeaderCleaner
from cleaners.footer_cleaner import FooterCleaner
from cleaners.whitespace_cleaner import WhitespaceCleaner
from chunking.recursive_chunker import RecursiveChunker
from utils.logger import logging
from embedding.embedding_dataclass import EmbeddingConfig
from vectorstore.fake_vector_store import FakeVectorStore
from retrieval.default_retriever import build_default_retriever
from experiments.evaluation.rag_evaluator import RAGEvaluator
from config import TOP_K


logger = logging.getLogger(__name__)


# ============================================================
# 0. REDIRECT PRINT OUTPUT TO A FILE
# ============================================================
#
# Everything written with print() below (retrieval results,
# per-question evaluation, final summary) is duplicated to both
# the console/debug console AND a timestamped .txt file under
# EVAL_DIR / "results".
#
# If you only want the file (no console output at all), delete
# `sys.__stdout__` from the `_targets` list in the Tee class below.

class Tee:
    """Writes everything given to it to multiple streams at once."""

    def __init__(self, *targets):
        self._targets = targets

    def write(self, data):
        for target in self._targets:
            target.write(data)
            target.flush()

    def flush(self):
        for target in self._targets:
            target.flush()


RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file_path = RESULTS_DIR / f"evaluation_run_{timestamp}.txt"

output_file = open(output_file_path, "w", encoding="utf-8")

_original_stdout = sys.stdout
sys.stdout = Tee(_original_stdout, output_file)

print(f"Writing evaluation output to: {output_file_path}\n")


try:

    # ============================================================
    # 1. LOAD DOCUMENTS
    # ============================================================

    logger.info(
        f"\nPDF_DIR is {PDF_DIR}\n"
    )

    logger.info(
        list(PDF_DIR.iterdir())
    )


    pdf_loader = PDFLoader(PDF_DIR)

    loaded_docs = pdf_loader.load_documents()


    logger.info(
        f"\nlen of loaded_docs is {len(loaded_docs)}\n"
    )


    if not loaded_docs:

        raise RuntimeError(
            "No documents were loaded from PDF_DIR."
        )


    logger.info(
        f"First 500 chars of loaded document:\n"
        f"{loaded_docs[0].page_content[:500]}"
    )


    logger.info(
        "Testing pdfloader.py is completed"
    )


    # ============================================================
    # 2. CLEAN DOCUMENTS
    # ============================================================

    print(
        "\n========== BEFORE CLEANING =========="
    )

    print(
        loaded_docs[0].page_content[:500]
    )


    pipeline = CleanerPipeline(
        cleaners=[
            HeaderCleaner(),
            FooterCleaner(),
            WhitespaceCleaner(),
        ]
    )


    clean_doc = pipeline.clean(
        loaded_docs
    )


    print(
        "\nCleaning is completed"
    )


    print(
        "\n========== AFTER CLEANING =========="
    )

    print(
        clean_doc[0].page_content[:500]
    )


    assert len(clean_doc) == len(loaded_docs)

    assert id(clean_doc[0]) == id(loaded_docs[0])


    print(
        "\nEntire Cleaning Process is completed"
    )


    # ============================================================
    # 3. CHUNK DOCUMENTS
    # ============================================================

    logger.info(
        "Chunking Process Started"
    )


    chunking = RecursiveChunker()


    # IMPORTANT:
    # Evaluate ALL loaded documents.
    #
    # Do NOT use:
    #
    # loaded_docs[:2]
    #
    # because the evaluation should operate on the complete
    # document collection.

    final_chunks = chunking.chunk(
        loaded_docs
    )


    logger.info(
        f"Length of Final Chunks is "
        f"{len(final_chunks)}"
    )


    for index, chunk in enumerate(
        final_chunks
    ):

        logger.info(
            "Chunk %s | length=%s | source=%s | page=%s | "
            "chunk_number=%s | chunk_id=%s | total_chunks=%s",

            index + 1,

            len(chunk.page_content),

            chunk.metadata.get(
                "source"
            ),

            chunk.metadata.get(
                "page"
            ),

            chunk.metadata.get(
                "chunk_number"
            ),

            chunk.metadata.get(
                "chunk_id"
            ),

            chunk.metadata.get(
                "total_chunks"
            ),
        )


    logger.info(
        "Chunking Process Completed"
    )


    # ============================================================
    # 4. CREATE EMBEDDINGS
    # ============================================================

    embedding_config = EmbeddingConfig()


    embedder = HuggingFaceEmbedder(
        embedding_config
    )


    embedding_response = embedder.embed(
        final_chunks
    )


    assert (
        embedding_response.embed_status
        == "SUCCESS"
    )


    assert (
        len(
            embedding_response.successful_embeddings
        )
        == len(final_chunks)
    )


    # ============================================================
    # 5. STORE EMBEDDINGS
    # ============================================================

    vector_store = FakeVectorStore()


    store_response = vector_store.add(
        embedding_response.successful_embeddings
    )


    assert (
        store_response.total_received_chunks
        == len(final_chunks)
    )


    assert (
        store_response.total_stored_chunks
        == len(final_chunks)
    )


    assert (
        store_response.total_skipped_chunks
        == 0
    )


    assert (
        store_response.total_failed_chunks
        == 0
    )


    # ============================================================
    # 6. CREATE RETRIEVER + EVALUATOR
    # ============================================================

    # Default retriever: BM25 + embedding fused (HybridRetriever),
    # reranked with the stronger L-12 cross-encoder, weak candidates
    # dropped via score_threshold. This is the "combined" variant from
    # evaluation/results/ragas_precision_experiments_summary_*.csv -
    # best precision AND recall of everything tried (avg precision
    # 0.9066, avg recall 1.0000 vs. plain embedding+L-6 rerank's
    # 0.8541/0.9222).

    retriever = build_default_retriever(vector_store)


    evaluator = RAGEvaluator()


    results = []


    # ============================================================
    # 7. LOAD GOLDEN DATASET
    # ============================================================

    golden_dataset_path = (
        EVAL_DIR
        / "golden_dataset_with_retrieval_ground_truth.json"
    )


    with open(
        golden_dataset_path,
        "r",
        encoding="utf-8",
    ) as file:

        golden_dataset = json.load(
            file
        )


    # ============================================================
    # 8. RUN RETRIEVAL EVALUATION
    # ============================================================

    for test_case in golden_dataset["entries"]:

        question_id = test_case["id"]

        query = test_case["query"]

        expected_answer = test_case[
            "expected_answer"
        ]

        retrieval_ground_truth = test_case[
            "retrieval_ground_truth"
        ]


        # ========================================================
        # QUERY EMBEDDING
        # ========================================================

        query_vector = embedder.embed_query(
            query
        )


        # ========================================================
        # RETRIEVAL
        # ========================================================

        top_k = TOP_K


        retrieved_results = retriever.retrieve(
            query_vector,
            top_k=top_k,
            query_text=query,
        )


        # ========================================================
        # QUESTION HEADER
        # ========================================================

        print(
            "\n" + "=" * 80
        )

        print(
            f"Question ID : {question_id}"
        )

        print(
            f"Query       : {query}"
        )

        print(
            "=" * 80
        )


        # ========================================================
        # RETRIEVAL RESULTS
        # ========================================================

        print(
            "\n---------- RETRIEVAL RESULTS ----------"
        )


        for rank, retrieved_result in enumerate(
            retrieved_results,
            start=1,
        ):

            metadata = (
                retrieved_result
                .document
                .metadata
            )

            content = (
                retrieved_result
                .document
                .page_content
            )


            print(
                f"\n[Rank {rank}]"
            )


            print(
                f"Score        : "
                f"{retrieved_result.score:.4f}"
            )


            print(
                f"Chunk ID      : "
                f"{metadata.get('chunk_id')}"
            )


            print(
                f"Source        : "
                f"{metadata.get('source')}"
            )


            print(
                f"Page          : "
                f"{metadata.get('page')}"
            )


            print(
                f"Chunk No      : "
                f"{metadata.get('chunk_number')}"
            )


            print(
                f"Total Chunks  : "
                f"{metadata.get('total_chunks')}"
            )


            print(
                "Content       :"
            )


            print(
                content[:500]
                .replace("\n", " ")
            )


        # ========================================================
        # RETRIEVAL ASSERTIONS
        # ========================================================

        for retrieved_result in retrieved_results:

            assert (
                retrieved_result.document
                is not None
            )


            assert (
                retrieved_result.document.metadata.get(
                    "chunk_id"
                )
                is not None
            )


            assert (
                retrieved_result.score >= 0
            )


        # ========================================================
        # EVALUATION
        # ========================================================

        evaluation_result = (
            evaluator.evaluate_retrieval(

                question_id=question_id,

                query=query,

                expected_answer=expected_answer,

                retrieved_results=retrieved_results,

                retrieval_ground_truth=(
                    retrieval_ground_truth
                ),

                k=top_k,
            )
        )


        # ========================================================
        # EVALUATION RESULT
        # ========================================================

        print(
            "\n---------- EVALUATION RESULT ----------"
        )


        print(
            f"Question ID       : "
            f"{evaluation_result.question_id}"
        )


        print(
            f"Retrieval Success  : "
            f"{evaluation_result.retrieval_success}"
        )


        print(
            f"Retrieved Count    : "
            f"{evaluation_result.retrieved_count}"
        )


        print(
            f"Top Score          : "
            f"{evaluation_result.top_score:.4f}"
        )


        print(
            f"Hit@{top_k}            : "
            f"{evaluation_result.hit_at_k:.4f}"
        )


        print(
            f"Recall@{top_k}         : "
            f"{evaluation_result.recall_at_k:.4f}"
        )


        print(
            f"MRR                : "
            f"{evaluation_result.reciprocal_rank:.4f}"
        )


        print(
            f"Status             : "
            f"{evaluation_result.evaluation_status}"
        )


        # ========================================================
        # EVALUATION ASSERTIONS
        # ========================================================

        assert (
            evaluation_result.question_id
            == question_id
        )


        assert (
            evaluation_result.query
            == query
        )


        assert (
            evaluation_result.expected_answer
            == expected_answer
        )


        assert (
            evaluation_result.retrieved_count
            == len(retrieved_results)
        )


        # ========================================================
        # STORE RESULT
        # ========================================================

        results.append(
            evaluation_result
        )


    # ============================================================
    # 9. FINAL EVALUATION SUMMARY
    # ============================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "FINAL RETRIEVAL EVALUATION SUMMARY"
    )

    print(
        "=" * 80
    )


    total_questions = len(
        results
    )


    # ============================================================
    # RETRIEVAL SUCCESS
    # ============================================================

    successful_retrievals = sum(
        1
        for result in results
        if result.retrieval_success
    )


    failed_retrievals = (
        total_questions
        - successful_retrievals
    )


    print(
        f"Total Questions       : "
        f"{total_questions}"
    )


    print(
        f"Successful Retrieval  : "
        f"{successful_retrievals}"
    )


    print(
        f"Failed Retrieval      : "
        f"{failed_retrievals}"
    )


    # ============================================================
    # AGGREGATE METRICS
    # ============================================================

    if total_questions > 0:

        retrieval_rate = (
            successful_retrievals
            / total_questions
        ) * 100


        average_hit_at_k = (
            sum(
                result.hit_at_k
                for result in results
            )
            / total_questions
        )


        average_recall_at_k = (
            sum(
                result.recall_at_k
                for result in results
            )
            / total_questions
        )


        mean_reciprocal_rank = (
            sum(
                result.reciprocal_rank
                for result in results
            )
            / total_questions
        )


        print(
            f"\nRetrieval Success Rate : "
            f"{retrieval_rate:.2f}%"
        )


        print(
            f"Average Hit@{top_k}      : "
            f"{average_hit_at_k:.4f}"
        )


        print(
            f"Average Recall@{top_k}   : "
            f"{average_recall_at_k:.4f}"
        )


        print(
            f"MRR                     : "
            f"{mean_reciprocal_rank:.4f}"
        )


    print(
        "=" * 80
    )

finally:

    # ============================================================
    # RESTORE STDOUT + CLOSE THE OUTPUT FILE
    # ============================================================

    sys.stdout = _original_stdout
    output_file.close()

    print(f"\nEvaluation output written to: {output_file_path}")

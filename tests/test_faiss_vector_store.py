import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from embedding.embedding_dataclass import EmbeddingResult
from chunking.recursive_chunker import RecursiveChunker
from vectorstore.faiss_vector_store import FaissVectorStore
from ingestion.reconciler import reconcile_source

chunker = RecursiveChunker()
test_dir = tempfile.mkdtemp(prefix="faiss_test_")

try:
    store = FaissVectorStore(index_dir=test_dir)

    source = "doc.pdf"

    v1_docs = [
        Document(page_content="Alpha content about filing status.", metadata={"source": source, "page": 1}),
        Document(page_content="Beta content about deductions.", metadata={"source": source, "page": 2}),
    ]
    v1_chunks = chunker.chunk(v1_docs)
    assert len(v1_chunks) == 2

    v1_embeddings = [
        EmbeddingResult(document=c, vector=[float(i), 0.0, 0.0])
        for i, c in enumerate(v1_chunks)
    ]
    response1 = store.add(v1_embeddings)
    assert response1.total_stored_chunks == 2
    assert store.index.ntotal == 2

    # search round-trip
    results = store.search(query_vector=[0.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2

    id_alpha = v1_chunks[0].metadata["chunk_id"]
    id_beta_v1 = v1_chunks[1].metadata["chunk_id"]

    assert set(store.get_ids_by_source(source)) == {id_alpha, id_beta_v1}

    # re-add identical content -> skipped, not duplicated
    response_dup = store.add(v1_embeddings)
    assert response_dup.total_skipped_chunks == 2
    assert response_dup.total_stored_chunks == 0
    assert store.index.ntotal == 2

    # v2: page 1 unchanged, page 2 edited
    v2_docs = [
        Document(page_content="Alpha content about filing status.", metadata={"source": source, "page": 1}),
        Document(page_content="Beta content about deductions, revised for 2026.", metadata={"source": source, "page": 2}),
    ]
    v2_chunks = chunker.chunk(v2_docs)
    id_beta_v2 = v2_chunks[1].metadata["chunk_id"]
    assert id_beta_v2 != id_beta_v1

    v2_embeddings = [
        EmbeddingResult(document=c, vector=[float(i), 1.0, 0.0])
        for i, c in enumerate(v2_chunks)
    ]
    response2 = store.add(v2_embeddings)
    assert response2.total_skipped_chunks == 1
    assert response2.total_stored_chunks == 1

    deleted_count = reconcile_source(store, source, {id_alpha, id_beta_v2})
    assert deleted_count == 1

    remaining_ids = set(store.get_ids_by_source(source))
    assert remaining_ids == {id_alpha, id_beta_v2}
    assert store.index.ntotal == 2

    # persistence round-trip: reopen from disk and confirm state survives
    reopened = FaissVectorStore(index_dir=test_dir)
    assert reopened.index.ntotal == 2
    assert set(reopened.get_ids_by_source(source)) == {id_alpha, id_beta_v2}

    print("FaissVectorStore add/search/delete/reconcile/persistence round-trip: PASS")

finally:
    del store
    shutil.rmtree(test_dir, ignore_errors=True)

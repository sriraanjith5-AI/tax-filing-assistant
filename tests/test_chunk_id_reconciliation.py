import shutil
import sys
import tempfile
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from embedding.embedding_dataclass import EmbeddingResult
from chunking.recursive_chunker import RecursiveChunker
from vectorstore.chroma_vector_store import ChromaVectorStore
from ingestion.reconciler import reconcile_source

# ============================================================
# 1. chunk_id is content-addressed, not positional
# ============================================================

chunker = RecursiveChunker()

# Same source + same content -> same id, regardless of what
# page/chunk_number end up being once it's chunked (those are no
# longer part of the hash, only the metadata).
id_a = chunker.generate_chunk_id("policy.pdf", "Standard deduction is $14,600.")
id_b = chunker.generate_chunk_id("policy.pdf", "Standard deduction is $14,600.")
assert id_a == id_b

# Same content, different source -> different id (scoped per source).
id_other_source = chunker.generate_chunk_id("other.pdf", "Standard deduction is $14,600.")
assert id_other_source != id_a

# Same source + content, different chunker (name) -> different id,
# so a future semantic chunker can't collide with recursive-chunker ids
# for identical text spans.
class FakeSemanticChunker(RecursiveChunker):
    pass

id_semantic = FakeSemanticChunker().generate_chunk_id("policy.pdf", "Standard deduction is $14,600.")
assert id_semantic != id_a

print("chunk_id is content-addressed: PASS")


# ============================================================
# 2. insert-new / delete-stale reconciliation on an edited source
# ============================================================

test_dir = tempfile.mkdtemp(prefix="chroma_reconcile_test_")
collection_name = f"test_reconciliation_{uuid.uuid4().hex[:8]}"

try:
    store = ChromaVectorStore(collection_name=collection_name, persist_path=test_dir)

    source = "doc.pdf"

    # v1: two pages, two chunks.
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

    id_alpha = v1_chunks[0].metadata["chunk_id"]
    id_beta_v1 = v1_chunks[1].metadata["chunk_id"]

    # v2: page 1 unchanged, page 2 edited.
    v2_docs = [
        Document(page_content="Alpha content about filing status.", metadata={"source": source, "page": 1}),
        Document(page_content="Beta content about deductions, revised for 2026.", metadata={"source": source, "page": 2}),
    ]
    v2_chunks = chunker.chunk(v2_docs)
    assert len(v2_chunks) == 2

    id_alpha_v2 = v2_chunks[0].metadata["chunk_id"]
    id_beta_v2 = v2_chunks[1].metadata["chunk_id"]

    # Unchanged page keeps its id; edited page gets a new one.
    assert id_alpha_v2 == id_alpha
    assert id_beta_v2 != id_beta_v1

    v2_embeddings = [
        EmbeddingResult(document=c, vector=[float(i), 1.0, 0.0])
        for i, c in enumerate(v2_chunks)
    ]
    response2 = store.add(v2_embeddings)
    # Alpha's id already exists (skipped, no re-embed); only the edited
    # Beta chunk is genuinely new.
    assert response2.total_skipped_chunks == 1
    assert response2.total_stored_chunks == 1

    deleted_count = reconcile_source(store, source, {id_alpha_v2, id_beta_v2})
    assert deleted_count == 1

    remaining_ids = set(store.get_ids_by_source(source))
    assert remaining_ids == {id_alpha, id_beta_v2}
    assert id_beta_v1 not in remaining_ids

    print("insert-new/delete-stale reconciliation: PASS")

finally:
    del store
    shutil.rmtree(test_dir, ignore_errors=True)

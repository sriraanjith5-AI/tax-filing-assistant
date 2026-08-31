import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from embedding.embedding_dataclass import EmbeddingResult
from vectorstore.fake_vector_store import FakeVectorStore
from vectorstore.vectorstore_dataclass import SearchResult
from retrieval.base_retriever import BaseRetriever
from retrieval.context_expander import ContextExpandingRetriever


class StubRetriever(BaseRetriever):
    """Returns a fixed, pre-set list of SearchResult - stands in for
    whatever upstream retriever (e.g. CrossEncoderReranker) actually
    picked the surviving chunks, so this test can exercise
    ContextExpandingRetriever's neighbor-merging in isolation."""

    def __init__(self, results):
        self.results = results

    def retrieve(self, query_vector, top_k, query_text=None):
        return self.results


def _doc(source, page, chunk_number, total_chunks, text):
    chunk_id = f"{source}-p{page}-c{chunk_number}"
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "page": page,
            "chunk_number": chunk_number,
            "total_chunks": total_chunks,
            "chunk_id": chunk_id,
        },
    )


# A single source/page split into 4 chunks - the whole corpus the
# position index gets built from.
all_chunks = [
    _doc("w4.pdf", 1, 1, 4, "Step 1: Enter personal information."),
    _doc("w4.pdf", 1, 2, 4, "Step 2: Multiple jobs or spouse works."),
    _doc("w4.pdf", 1, 3, 4, "Step 3: Claim dependents."),
    _doc("w4.pdf", 1, 4, 4, "Step 4: Other adjustments."),
]

vector_store = FakeVectorStore()
vector_store.add([
    EmbeddingResult(document=doc, vector=[0.0])
    for doc in all_chunks
])


# ------------------------------------------------------------------
# A middle chunk pulls in both neighbors.
# ------------------------------------------------------------------

middle_hit = SearchResult(document=all_chunks[1], score=0.9)  # chunk_number=2
retriever = ContextExpandingRetriever(
    base_retriever=StubRetriever([middle_hit]),
    vector_store=vector_store,
    window=1,
)
results = retriever.retrieve(query_vector=None, top_k=1, query_text="q")

assert len(results) == 1
merged = results[0].document
assert results[0].score == 0.9
assert "Step 1" in merged.page_content
assert "Step 2" in merged.page_content
assert "Step 3" in merged.page_content
assert "Step 4" not in merged.page_content
assert merged.metadata["context_expanded"] is True
assert merged.metadata["expanded_chunk_ids"] == [
    "w4.pdf-p1-c1", "w4.pdf-p1-c2", "w4.pdf-p1-c3",
]
# The surviving chunk's own identity (chunk_id/chunk_number/etc.) is
# preserved on the merged Document, since it's still "that chunk" as
# far as scoring/ranking/dedup upstream is concerned.
assert merged.metadata["chunk_id"] == "w4.pdf-p1-c2"


# ------------------------------------------------------------------
# An edge chunk (chunk_number=1) only has one neighbor - lo clips at 1.
# ------------------------------------------------------------------

edge_hit = SearchResult(document=all_chunks[0], score=0.5)  # chunk_number=1
retriever_edge = ContextExpandingRetriever(
    base_retriever=StubRetriever([edge_hit]),
    vector_store=vector_store,
    window=1,
)
edge_results = retriever_edge.retrieve(query_vector=None, top_k=1, query_text="q")
edge_merged = edge_results[0].document

assert "Step 1" in edge_merged.page_content
assert "Step 2" in edge_merged.page_content
assert "Step 3" not in edge_merged.page_content


# ------------------------------------------------------------------
# window=0 disables expansion entirely - passthrough, same object.
# ------------------------------------------------------------------

retriever_disabled = ContextExpandingRetriever(
    base_retriever=StubRetriever([middle_hit]),
    vector_store=vector_store,
    window=0,
)
disabled_results = retriever_disabled.retrieve(query_vector=None, top_k=1, query_text="q")
assert disabled_results[0].document is all_chunks[1]
assert "context_expanded" not in disabled_results[0].document.metadata


# ------------------------------------------------------------------
# Missing positional metadata (e.g. a test fixture without
# chunk_number) - returned untouched instead of erroring.
# ------------------------------------------------------------------

bare_doc = Document(page_content="no positional metadata", metadata={"chunk_id": "bare-1"})
retriever_bare = ContextExpandingRetriever(
    base_retriever=StubRetriever([SearchResult(document=bare_doc, score=0.1)]),
    vector_store=vector_store,
    window=1,
)
bare_results = retriever_bare.retrieve(query_vector=None, top_k=1, query_text="q")
assert bare_results[0].document is bare_doc

# ------------------------------------------------------------------
# Two adjacent surviving chunks (chunk_number 2 and 3) have overlapping
# +/-1 windows ([1,2,3] and [2,3,4]) - these must merge into a single
# Document instead of shipping chunks 2/3 to the generator twice.
# ------------------------------------------------------------------

hit_2 = SearchResult(document=all_chunks[1], score=0.7)   # chunk_number=2
hit_3 = SearchResult(document=all_chunks[2], score=0.9)   # chunk_number=3, higher score
retriever_overlap = ContextExpandingRetriever(
    base_retriever=StubRetriever([hit_2, hit_3]),
    vector_store=vector_store,
    window=1,
)
overlap_results = retriever_overlap.retrieve(query_vector=None, top_k=2, query_text="q")

# Merged into exactly one result, not two.
assert len(overlap_results) == 1
merged_overlap = overlap_results[0].document
assert "Step 1" in merged_overlap.page_content
assert "Step 2" in merged_overlap.page_content
assert "Step 3" in merged_overlap.page_content
assert "Step 4" in merged_overlap.page_content
# Each chunk's text appears exactly once - not duplicated across two cards.
assert merged_overlap.page_content.count("Step 2") == 1
assert merged_overlap.page_content.count("Step 3") == 1
assert merged_overlap.metadata["context_expanded"] is True
assert merged_overlap.metadata["expanded_chunk_ids"] == [
    "w4.pdf-p1-c1", "w4.pdf-p1-c2", "w4.pdf-p1-c3", "w4.pdf-p1-c4",
]
# Score is the max of the merged members' scores (the stronger evidence).
assert overlap_results[0].score == 0.9


# ------------------------------------------------------------------
# Non-overlapping hits on the same page (chunk_number 1 and 4, window=1)
# stay as two separate results - their windows ([1,2] via clip and
# [3,4]) don't actually intersect.
# ------------------------------------------------------------------

hit_1 = SearchResult(document=all_chunks[0], score=0.6)   # chunk_number=1
hit_4 = SearchResult(document=all_chunks[3], score=0.4)   # chunk_number=4
retriever_no_overlap = ContextExpandingRetriever(
    base_retriever=StubRetriever([hit_1, hit_4]),
    vector_store=vector_store,
    window=1,
)
no_overlap_results = retriever_no_overlap.retrieve(query_vector=None, top_k=2, query_text="q")
assert len(no_overlap_results) == 2

print("All ContextExpandingRetriever tests passed.")

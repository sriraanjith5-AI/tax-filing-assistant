import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from chunking.semantic_chunker import SemanticChunker

docs = [
    Document(
        page_content=(
            "The standard deduction reduces taxable income. "
            "For 2026, the standard deduction for single filers is $14,600. "
            "Married filing jointly filers get double that amount. "
            "Form W-4 tells your employer how much tax to withhold from your paycheck. "
            "Filing status affects your tax bracket and standard deduction amount. "
            "You can itemize deductions instead of taking the standard deduction if it "
            "results in a larger deduction overall."
        ),
        metadata={"source": "guide.pdf", "page": 1},
    )
]

chunker = SemanticChunker()
chunks = chunker.chunk(docs)

assert len(chunks) >= 1, "SemanticChunker should produce at least one chunk"

for i, chunk in enumerate(chunks, start=1):
    assert chunk.metadata["source"] == "guide.pdf"
    assert chunk.metadata["page"] == 1
    assert chunk.metadata["chunk_number"] == i
    assert chunk.metadata["total_chunks"] == len(chunks)
    assert chunk.metadata["chunk_id"] == chunker.generate_chunk_id(
        "guide.pdf", chunk.page_content
    )

print(f"SemanticChunker produced {len(chunks)} chunk(s) with a valid metadata contract: PASS")

# Same content, different chunker name -> different chunk_id (isolated
# from RecursiveChunker even over identical text).
from chunking.recursive_chunker import RecursiveChunker
recursive = RecursiveChunker()
semantic_id = chunker.generate_chunk_id("guide.pdf", "Sample text.")
recursive_id = recursive.generate_chunk_id("guide.pdf", "Sample text.")
assert semantic_id != recursive_id
print("SemanticChunker ids are isolated from RecursiveChunker ids: PASS")

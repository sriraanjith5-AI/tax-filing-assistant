from dataclasses import dataclass
from langchain_core.documents import Document

@dataclass
class VectorStoreResponse:
    total_received_chunks: int
    total_stored_chunks: int
    total_skipped_chunks: int
    total_failed_chunks: int

@dataclass
class SearchResult:
    document: Document
    score: float

from dataclasses import dataclass
from typing import List
from langchain_core.documents import Document

@dataclass
class EmbeddingResult:
    document: Document
    vector: List[float]

@dataclass
class EmbeddingResponse:
    embed_status: str
    total_no_documents: int
    successful_documents: List[EmbeddingResult]
    failed_documents: List[Document]

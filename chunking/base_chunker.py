import hashlib
from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class BaseChunker(ABC):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def generate_chunk_id(self, source: str, content: str) -> str:
        # Content-addressed, not positional: identity depends only on
        # source + chunker + normalized text, never on page/chunk_number.
        # This keeps ids stable when upstream edits shift chunk boundaries
        # (inserting/removing text no longer reassigns unrelated chunks'
        # ids), and scopes ids per chunker (self.name) so different
        # chunking strategies never collide on ids for the same text span
        # - each gets its own isolated set of chunk_ids even over
        # identical source text.
        normalized_content = content.strip()
        identity = (
                f"{source}|"
                f"{self.name}|"
                f"{normalized_content}"
                  )
        return hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

    @abstractmethod
    def chunk(self, documents:List[Document]) -> List[Document]:
        """Chunks the Document and returns the chunks"""
        pass

from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class BaseChunker(ABC):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def chunk(self, documents:List[Document]) -> List[Document]:
        """Chunks the Document and returns the chunks"""
        pass
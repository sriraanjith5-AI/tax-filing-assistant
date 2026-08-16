from abc import ABC, abstractmethod
from typing import List
from vectorstore.vectorstore_dataclass import SearchResult

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self,query_vector,top_k:int) -> List[SearchResult]:
        pass
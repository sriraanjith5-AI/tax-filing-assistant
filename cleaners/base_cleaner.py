from abc import ABC, abstractmethod
from langchain_core.documents import Document


class BaseCleaner(ABC):

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def clean(self, document: Document) -> None:
        """
        Cleans the Document in place.
        """
        pass
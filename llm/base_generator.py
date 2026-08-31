from abc import ABC, abstractmethod
from typing import List, Optional

from llm.generation_dataclass import GenerationResult


class BaseGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        query: str,
        contexts: List[str],
        sources: Optional[List[str]] = None,
        citation_labels: Optional[List[str]] = None,
    ) -> GenerationResult:
        """
        Synthesizes an answer to `query` using only `contexts` (retrieved
        chunk texts, in rank order - typically page_content from the
        retriever's SearchResults).

        sources, when given, is a same-length/order list of human-readable
        labels (one per context) that implementations may surface to the
        model so it can cite which excerpt a claim came from.

        citation_labels, when given, is a same-length/order list of the
        exact bracket-marker text to use per excerpt (e.g. "p.7") instead
        of a plain positional index - lets the cited marker itself name
        the source page directly, rather than an index a caller has to
        resolve against a separate source list.
        """
        pass

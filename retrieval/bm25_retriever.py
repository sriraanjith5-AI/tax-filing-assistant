import re
from typing import List

from rank_bm25 import BM25Okapi

from retrieval.base_retriever import BaseRetriever
from vectorstore.vectorstore_dataclass import SearchResult


class BM25Retriever(BaseRetriever):
    """
    Lexical (keyword) retriever using BM25.

    Embedding similarity is good at topical relevance but blurs
    fine-grained lexical distinctions - e.g. "Form W-4" vs.
    "Form W-4P", or "Step 2" vs. "Step 3" - because those differences
    barely move a dense vector. BM25 scores exact/near-exact term
    overlap directly, so it catches exactly the failures embeddings
    miss. Used as one leg of HybridRetriever, not as a standalone
    replacement (BM25 alone is blind to paraphrasing/synonyms).
    """

    def __init__(self, documents):
        """
        documents : list of langchain_core.documents.Document
            The full chunk corpus to index (same chunks stored in
            the vector store).
        """

        self.documents = documents

        corpus_tokens = [
            self._tokenize(document.page_content)
            for document in documents
        ]

        self.bm25 = BM25Okapi(corpus_tokens)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def retrieve(
        self,
        query_vector,
        top_k: int,
        query_text: str = None,
    ) -> List[SearchResult]:

        if query_text is None:
            raise ValueError(
                "BM25Retriever.retrieve() requires query_text for "
                "lexical scoring."
            )

        if not self.documents:
            return []

        query_tokens = self._tokenize(query_text)
        scores = self.bm25.get_scores(query_tokens)

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        return [
            SearchResult(
                document=self.documents[index],
                score=float(scores[index]),
            )
            for index in ranked_indexes
        ]

from typing import List

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from chunking.base_chunker import BaseChunker
from config import EMBEDDING_MODEL
from utils.logger import logging

logger = logging.getLogger(__name__)


class SemanticChunker(BaseChunker):
    """
    Splits text at embedding-similarity breakpoints instead of a fixed
    character count (RecursiveChunker) - consecutive sentences are
    grouped together until the semantic similarity to the next sentence
    drops below a percentile-based threshold, so chunk boundaries land
    on topic shifts rather than arbitrary character offsets.

    Uses langchain_huggingface.HuggingFaceEmbeddings (LangChain's
    Embeddings interface) purely to score sentence-boundary similarity
    - this is intentionally decoupled from the project's own
    HuggingFaceEmbedder (embedding/huggingface_embedder.py), which is
    still what actually embeds chunks for storage.

    Has no chunk_size/chunk_overlap knobs (unlike RecursiveChunker) -
    boundaries are determined by the similarity threshold, not a fixed
    length, so those UI fields don't apply to this chunker.
    """

    def __init__(self, embeddings=None):
        self._splitter = LCSemanticChunker(
            embeddings or HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        if len(documents) == 0:
            logger.info("Length of received Document is Empty")
            return []

        chunks = []

        for doc in documents:
            if len(doc.page_content) == 0:
                continue

            chunks_intermediate = self._splitter.split_documents([doc])
            if len(chunks_intermediate) == 0:
                continue

            for index, chunk in enumerate(chunks_intermediate):
                source = doc.metadata['source']
                page = doc.metadata['page']
                chunk_number = index + 1

                chunk.metadata = {}

                chunk.metadata['source'] = source
                chunk.metadata['page'] = page
                chunk.metadata['chunk_number'] = chunk_number
                chunk.metadata['total_chunks'] = len(chunks_intermediate)
                chunk.metadata['chunk_id'] = self.generate_chunk_id(
                    source,
                    chunk.page_content)
                chunks.append(chunk)

        return chunks

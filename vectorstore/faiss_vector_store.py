import pickle
from pathlib import Path
from typing import List

import faiss
import numpy as np
from langchain_core.documents import Document

from embedding.embedding_dataclass import EmbeddingResult
from vectorstore.base_vector_store import BaseVectorStore
from vectorstore.vectorstore_dataclass import VectorStoreResponse, SearchResult


class FaissVectorStore(BaseVectorStore):
    """
    FAISS-backed vector store, parallel to ChromaVectorStore - same
    add()/search()/get_all_documents() contract, plus get_ids_by_source()/
    delete() for reconciliation (ingestion/reconciler.py).

    FAISS itself only stores vectors - metadata/chunk_id lookups are
    kept in a sidecar dict, persisted alongside the index file so state
    survives across runs (mirrors ChromaVectorStore's persist_path).

    Vectors are L2-normalized and indexed with inner product
    (IndexFlatIP), which is equivalent to cosine similarity - matching
    ChromaVectorStore's "hnsw:space": "cosine" configuration so scores
    from the two stores are comparable.
    """

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "index.faiss"
        self.meta_path = self.index_dir / "meta.pkl"

        self.index = None  # lazily created on first add(), once we know the vector dim
        self.chunk_id_to_int_id = {}
        self.int_id_to_document = {}
        self._next_int_id = 0

        if self.index_path.exists() and self.meta_path.exists():
            self._load()

    def _load(self) -> None:
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "rb") as f:
            meta = pickle.load(f)
        self.chunk_id_to_int_id = meta["chunk_id_to_int_id"]
        self.int_id_to_document = meta["int_id_to_document"]
        self._next_int_id = meta["next_int_id"]

    def _save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(
                {
                    "chunk_id_to_int_id": self.chunk_id_to_int_id,
                    "int_id_to_document": self.int_id_to_document,
                    "next_int_id": self._next_int_id,
                },
                f,
            )

    @staticmethod
    def _normalize(vector) -> np.ndarray:
        array = np.asarray(vector, dtype="float32")
        norm = np.linalg.norm(array)
        if norm == 0:
            return array
        return array / norm

    def add(self, embeddings: List[EmbeddingResult]) -> VectorStoreResponse:
        try:
            total_received_chunks = len(embeddings)
            total_stored_chunks = 0
            total_skipped_chunks = 0
            total_failed_chunks = 0

            to_add_vectors = []
            to_add_int_ids = []

            for e in embeddings:
                chunk_id = e.document.metadata.get("chunk_id")
                if chunk_id is None:
                    total_failed_chunks += 1
                    continue
                if chunk_id in self.chunk_id_to_int_id:
                    total_skipped_chunks += 1
                    continue

                if self.index is None:
                    dim = len(e.vector)
                    self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

                int_id = self._next_int_id
                self._next_int_id += 1

                self.chunk_id_to_int_id[chunk_id] = int_id
                self.int_id_to_document[int_id] = e.document

                to_add_vectors.append(self._normalize(e.vector))
                to_add_int_ids.append(int_id)
                total_stored_chunks += 1

            if to_add_vectors:
                self.index.add_with_ids(
                    np.vstack(to_add_vectors),
                    np.asarray(to_add_int_ids, dtype="int64"),
                )
                self._save()

            return VectorStoreResponse(
                total_received_chunks, total_stored_chunks,
                total_skipped_chunks, total_failed_chunks,
            )
        except Exception as e:
            print(f"Error occurred while adding embeddings: {e}")
            return VectorStoreResponse(
                len(embeddings), 0, 0, len(embeddings),
            )

    def search(self, query_vector: List[float], top_k: int) -> List[SearchResult]:
        if top_k <= 0:
            return []
        if query_vector is None or len(query_vector) == 0:
            return []
        if self.index is None or self.index.ntotal == 0:
            return []

        query = self._normalize(query_vector).reshape(1, -1)
        scores, int_ids = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for score, int_id in zip(scores[0], int_ids[0]):
            if int_id == -1:
                continue
            document = self.int_id_to_document.get(int(int_id))
            if document is None:
                continue
            results.append(SearchResult(document=document, score=float(score)))
        return results

    def get_all_documents(self) -> List[Document]:
        return list(self.int_id_to_document.values())

    def get_ids_by_source(self, source: str) -> List[str]:
        return [
            chunk_id
            for chunk_id, int_id in self.chunk_id_to_int_id.items()
            if self.int_id_to_document[int_id].metadata.get("source") == source
        ]

    def delete(self, ids: List[str]) -> None:
        if not ids:
            return
        int_ids = [
            self.chunk_id_to_int_id.pop(chunk_id)
            for chunk_id in ids
            if chunk_id in self.chunk_id_to_int_id
        ]
        if not int_ids:
            return
        for int_id in int_ids:
            self.int_id_to_document.pop(int_id, None)
        self.index.remove_ids(np.asarray(int_ids, dtype="int64"))
        self._save()

    def clear(self) -> None:
        self.index = None
        self.chunk_id_to_int_id = {}
        self.int_id_to_document = {}
        self._next_int_id = 0
        for path in (self.index_path, self.meta_path):
            if path.exists():
                path.unlink()

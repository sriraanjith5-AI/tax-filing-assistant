from typing import List, Optional
import chromadb
from langchain_core.documents import Document
from config import VECTOR_DB_DIR
from embedding.embedding_dataclass import EmbeddingResult
from vectorstore.base_vector_store import BaseVectorStore
from vectorstore.vectorstore_dataclass import VectorStoreResponse, SearchResult

DEFAULT_COLLECTION_NAME = "IT_Filling_Assistant_Collection"

class ChromaVectorStore(BaseVectorStore):
    def __init__(self, collection_name: Optional[str] = None, persist_path: Optional[str] = None):
        #Create or Generate a ChromaDB Collection
        # collection_name/persist_path let callers point at an isolated
        # collection - e.g. a per-chunker collection for A/B comparing
        # RecursiveChunker vs a future semantic chunker
        # ("{base}__{chunker_name}"), or a throwaway test path - without
        # touching the production collection/data directory.
        path = persist_path or str(VECTOR_DB_DIR / "chroma_data")
        client = chromadb.PersistentClient(path=path)
        if client.heartbeat():
            # Cosine distance matches how the project's embedding
            # model (BAAI/bge-small-en-v1.5) is meant to be compared -
            # Chroma's HNSW index defaults to l2 otherwise.
            self.collection = client.get_or_create_collection(
                name=collection_name or DEFAULT_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            print("ChromaDB collection created or retrieved successfully.")
        else:
            raise Exception("Failed to connect to ChromaDB. Please check the connection.")
        
    def check_existing_id(self, ids: List[str]) -> List[str]:
        # Chroma's get() rejects a query `ids` list that contains
        # duplicates outright (DuplicateIDError), regardless of whether
        # they're already stored - dedupe before querying. Duplicates
        # do occur legitimately: content-addressed chunk_id means two
        # chunks with identical (source, content) - e.g. repeated
        # boilerplate text a chunker splits out as separate units on
        # different pages - hash to the same id by design.
        unique_ids = list(set(ids))
        db_ids=self.collection.get(ids=unique_ids)
        db_ids_unique=set(db_ids["ids"])
        embedding_ids_unique=set(unique_ids)
        duplicate_ids=db_ids_unique.intersection(embedding_ids_unique)
        return duplicate_ids

    def add(self, embeddings: List[EmbeddingResult]) -> VectorStoreResponse:
        # Initialized before anything that can raise, so the except
        # block below always has values to report instead of hitting
        # UnboundLocalError on top of whatever the original error was.
        total_received_chunks = len(embeddings)
        total_stored_chunks=0
        total_skipped_chunks=0
        total_failed_chunks=0
        try:
            to_add = []
            embedding_chunk_ids = [e.document.metadata.get("chunk_id") for e in embeddings]
            existing_ids = self.check_existing_id(embedding_chunk_ids)
            seen_ids = set()
            for e in embeddings:
                id = e.document.metadata.get("chunk_id")
                if id is None:
                    total_failed_chunks+=1
                elif id in seen_ids:
                    # Same content-addressed id already queued in this
                    # same add() call (e.g. duplicate/boilerplate text
                    # split out twice by the chunker) - collection.add()
                    # would also reject duplicate ids in one call.
                    total_skipped_chunks+=1
                elif id not in existing_ids:
                    to_add.append(e)
                    seen_ids.add(id)
                    total_stored_chunks+=1
                else:
                    total_skipped_chunks+=1
            if len(to_add) > 0:
                self.collection.add(documents=[e.document.page_content for e in to_add],
                                metadatas=[e.document.metadata for e in to_add],
                                ids=[e.document.metadata.get("chunk_id") for e in to_add],
                                embeddings=[e.vector for e in to_add])
            return VectorStoreResponse(total_received_chunks, total_stored_chunks, total_skipped_chunks, total_failed_chunks)
        except Exception as e:
            print(f"Error occurred while adding embeddings: {e}")
            return VectorStoreResponse(total_received_chunks, total_stored_chunks, total_skipped_chunks, total_failed_chunks)
        
    def search(self, query_vector: List[float], top_k: int) -> List[SearchResult]:
        if top_k <= 0:
            return []
        if query_vector is None or len(query_vector) == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]

        search_results = []
        for page_content, metadata, distance in zip(
            documents[0], metadatas[0], distances[0]
        ):
            search_results.append(
                SearchResult(
                    document=Document(
                        page_content=page_content,
                        metadata=metadata or {},
                    ),
                    # Cosine distance -> similarity, so higher is
                    # better - consistent with FakeVectorStore's
                    # cosine-similarity score.
                    score=1.0 - distance,
                )
            )
        return search_results

    def get_all_documents(self) -> List[Document]:
        total = self.collection.count()
        if total == 0:
            return []

        results = self.collection.get(
            include=["documents", "metadatas"],
            limit=total,
        )

        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

        return [
            Document(page_content=page_content, metadata=metadata or {})
            for page_content, metadata in zip(documents, metadatas)
        ]

    def get_ids_by_source(self, source: str) -> List[str]:
        """All chunk_ids currently stored for a given source document -
        used to reconcile stale chunks after re-ingesting an edited
        source (see ingestion/reconciler.py)."""
        results = self.collection.get(where={"source": source}, include=[])
        return results.get("ids") or []

    def delete(self, ids: List[str]) -> None:
        if not ids:
            return
        self.collection.delete(ids=ids)

    def clear(self) -> None:
        total = self.collection.count()
        if total == 0:
            return
        # Chroma's delete() needs an explicit ids/where filter - there's
        # no single "delete everything" call - so fetch every id first.
        all_ids = self.collection.get(include=[])["ids"]
        self.collection.delete(ids=all_ids)

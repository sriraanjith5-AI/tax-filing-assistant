from typing import Iterable, Union

from vectorstore.chroma_vector_store import ChromaVectorStore
from vectorstore.faiss_vector_store import FaissVectorStore


def reconcile_source(
    store: Union[ChromaVectorStore, FaissVectorStore],
    source: str,
    new_chunk_ids: Iterable[str],
) -> int:
    """After store.add()-ing freshly chunked+embedded content for `source`,
    remove any chunk ids previously stored for that source that are no
    longer produced by the current chunking run.

    chunk_id is content-addressed (see RecursiveChunker.generate_chunk_id),
    so store.add() already skips re-embedding/re-upserting chunks whose
    content is unchanged (same id -> already present). This function
    handles the other half: deleting ids that belonged to text which was
    edited or removed, so it doesn't linger as an orphaned, unreachable-
    by-current-content row.

    Returns the number of stale ids deleted.
    """
    new_ids = set(new_chunk_ids)
    existing_ids = set(store.get_ids_by_source(source))
    stale_ids = existing_ids - new_ids
    if stale_ids:
        store.delete(list(stale_ids))
    return len(stale_ids)

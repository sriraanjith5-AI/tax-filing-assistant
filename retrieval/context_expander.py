from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from retrieval.base_retriever import BaseRetriever
from vectorstore.vectorstore_dataclass import SearchResult
from utils.trace import record_stage, page_display


class ContextExpandingRetriever(BaseRetriever):
    """
    Post-retrieval step: for each chunk that survives the wrapped
    retriever, pull in its adjacent chunks by document position (not
    similarity) and merge them into one expanded Document before the
    generator ever sees it.

    Chunk boundaries are an artifact of chunking, not of the source
    document - a chunk can score well on its own while missing a
    sentence/clause finished in its neighbor. Every stage before this
    one (embedding search, BM25, cross-encoder rerank) scores chunk
    content in isolation, so that kind of boundary loss is invisible
    to all of them - only a positional lookup catches it.

    Position is (source, page, chunk_number): chunk_number restarts
    at 1 per page (see RecursiveChunker/SemanticChunker), so neighbors
    are only ever looked up within the same source+page - this never
    stitches together chunks from different pages/documents.

    chunk_id itself is content-addressed (BaseChunker.generate_chunk_id),
    not positional, so it can't be incremented/decremented to find a
    neighbor - that's why this looks neighbors up via (source, page,
    chunk_number) instead.

    If two surviving chunks are themselves adjacent (e.g. chunk_number
    3 and 4 on the same page both make it past reranking), expanding
    each independently would give overlapping windows ([2,3,4] and
    [3,4,5]) - the shared chunks would then appear twice in the
    generator's context, under two different citations. retrieve()
    merges any such overlapping windows into one Document before
    returning, so each region of the source document is only ever
    passed to the generator once.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        vector_store,
        window: int = 1,
    ):
        """
        Parameters
        ----------
        base_retriever : BaseRetriever
            The upstream retriever (typically the fully assembled
            hybrid + cross-encoder-reranked retriever) whose results
            get expanded. Expansion runs AFTER this retriever's own
            top_k truncation, so it never changes which chunks were
            selected - only how much surrounding text each one carries.

        vector_store : BaseVectorStore
            Used once, lazily on first retrieve(), to build an
            in-memory (source, page, chunk_number) -> Document index
            via get_all_documents() - same pattern build_retriever
            already uses to build the BM25 leg over the same corpus
            (retrieval/default_retriever.py), so this needs no new
            vector-store query methods.

        window : int
            Number of neighboring chunks to pull in on each side of a
            surviving chunk. window=1 merges [n-1, n, n+1]; window=0
            disables expansion (results are returned unchanged).
        """
        self.base_retriever = base_retriever
        self.vector_store = vector_store
        self.window = window
        self._position_index: Optional[Dict[Tuple, Document]] = None

    def _build_position_index(self) -> Dict[Tuple, Document]:
        index = {}
        for document in self.vector_store.get_all_documents():
            source = document.metadata.get("source")
            chunk_number = document.metadata.get("chunk_number")
            if source is None or chunk_number is None:
                # No positional metadata to index this document by -
                # it simply can't be found as anyone's neighbor.
                continue
            index[(source, document.metadata.get("page"), chunk_number)] = document
        return index

    def retrieve(
        self,
        query_vector,
        top_k: int,
        query_text: str = None,
    ) -> List[SearchResult]:

        results = self.base_retriever.retrieve(
            query_vector,
            top_k=top_k,
            query_text=query_text,
        )

        if self.window <= 0 or not results:
            return results

        if self._position_index is None:
            self._position_index = self._build_position_index()

        return self._expand_and_merge(results)

    def _window_range(self, document: Document):
        """(source, page, lo, hi) this document's expansion would cover,
        or None if it has no positional metadata to expand against."""
        source = document.metadata.get("source")
        chunk_number = document.metadata.get("chunk_number")

        if source is None or chunk_number is None:
            return None

        page = document.metadata.get("page")
        total_chunks = document.metadata.get("total_chunks")

        lo = max(1, chunk_number - self.window)
        hi = chunk_number + self.window
        if total_chunks is not None:
            hi = min(hi, total_chunks)

        return (source, page, lo, hi)

    def _expand_and_merge(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Expands each surviving chunk's window, then merges any whose
        windows overlap on the same (source, page) into a single
        Document, instead of shipping the shared, duplicated text to
        the generator twice under two different citations.

        Two separately-ranked top-k hits are often adjacent chunks on
        the same page - e.g. chunk_number 3 and 4 both surviving the
        reranker independently. Expanding each in isolation (window=1)
        gives [2,3,4] and [3,4,5]: chunks 3 and 4 then appear, in full,
        inside *both* expanded Documents. Left alone, that means the
        generator sees the same paragraph twice under two different
        "[p.N]" citations, and a user sees two near-identical source
        cards for what is really one contiguous excerpt.
        """
        ranges = [self._window_range(result.document) for result in results]

        # Chunks with no positional metadata can't be merged with
        # anything - each is its own single-member group, handled via
        # the same single-chunk path as an unmerged one below.
        spans_by_page = {}
        for i, rng in enumerate(ranges):
            if rng is None:
                continue
            source, page, lo, hi = rng
            spans_by_page.setdefault((source, page), []).append((lo, hi, i))

        # index -> which merged group (list of member indices) it belongs to.
        group_of = {}
        groups = []
        for spans in spans_by_page.values():
            spans.sort(key=lambda span: span[0])
            group_lo, group_hi, members = spans[0]
            members = [members]
            for lo, hi, i in spans[1:]:
                if lo <= group_hi:  # window overlaps/touches the running group
                    group_hi = max(group_hi, hi)
                    members.append(i)
                else:
                    groups.append((group_lo, group_hi, members))
                    group_lo, group_hi, members = lo, hi, [i]
            groups.append((group_lo, group_hi, members))

        for group in groups:
            for i in group[2]:
                group_of[i] = group

        output = []
        seen_groups = set()
        for i, result in enumerate(results):
            group = group_of.get(i)

            if group is None:
                # No positional metadata - can't expand or merge.
                output.append((i, result))
                continue

            if len(group[2]) == 1:
                # Sole occupant of its merged range - same as before,
                # no other surviving chunk's window overlapped it.
                document = self._expand_single(result.document)
                output.append((i, SearchResult(document=document, score=result.score)))
                continue

            # Part of a multi-chunk merge - only emit it once, at the
            # position of its highest-ranked (best-scoring) member, so
            # the merged excerpt appears where its strongest evidence
            # would have.
            if id(group) in seen_groups:
                continue
            seen_groups.add(id(group))

            lo, hi, members = group
            source = ranges[members[0]][0]
            page = ranges[members[0]][1]
            best_member = max(members, key=lambda m: results[m].score)
            document = self._build_range_document(
                source, page, lo, hi, anchor=results[best_member].document,
            )
            score = max(results[m].score for m in members)
            output.append((best_member, SearchResult(document=document, score=score)))

        output.sort(key=lambda item: item[0])
        merged_group_count = sum(1 for group in groups if len(group[2]) > 1)
        record_stage(
            "context_expansion",
            window=self.window,
            input_count=len(results),
            output_count=len(output),
            overlapping_groups_merged=merged_group_count,
            output=[
                {
                    "chunk_id": r.document.metadata.get("chunk_id"),
                    "source": r.document.metadata.get("source"),
                    "page_display": page_display(r.document.metadata),
                    "score": round(float(r.score), 4) if r.score is not None else None,
                    "expanded_chunk_ids": r.document.metadata.get("expanded_chunk_ids"),
                }
                for _, r in output
            ],
        )
        return [item[1] for item in output]

    def _expand_single(self, document: Document) -> Document:
        """Expands one chunk's window in isolation - used when no other
        surviving chunk's window overlaps it, so there's nothing to merge."""
        source = document.metadata.get("source")
        chunk_number = document.metadata.get("chunk_number")

        if source is None or chunk_number is None:
            # No positional metadata (e.g. a store/fixture that doesn't
            # set it) - nothing to expand against, return as-is.
            return document

        page = document.metadata.get("page")
        total_chunks = document.metadata.get("total_chunks")

        lo = max(1, chunk_number - self.window)
        hi = chunk_number + self.window
        if total_chunks is not None:
            hi = min(hi, total_chunks)

        merged = self._collect_range(source, page, lo, hi, fallback={chunk_number: document})

        if len(merged) <= 1:
            # No neighbor actually found - return the original chunk
            # untouched rather than wrapping it in a new Document.
            return document

        metadata = dict(document.metadata)
        metadata["expanded_chunk_ids"] = [d.metadata.get("chunk_id") for d in merged]
        metadata["context_expanded"] = True

        return Document(
            page_content="\n\n".join(d.page_content for d in merged),
            metadata=metadata,
        )

    def _build_range_document(self, source, page, lo, hi, anchor: Document) -> Document:
        """Builds one merged Document spanning chunk_numbers [lo, hi],
        used when >=2 surviving chunks' windows overlapped into one
        range. `anchor` (the best-scoring member) supplies the base
        metadata (page, total_chunks, etc.) - its own chunk_id/
        chunk_number no longer identify a single chunk once several
        have been folded together, so they're dropped in favor of the
        explicit range."""
        merged = self._collect_range(
            source, page, lo, hi,
            fallback={anchor.metadata.get("chunk_number"): anchor},
        )

        metadata = dict(anchor.metadata)
        metadata["chunk_number"] = None
        metadata["expanded_chunk_ids"] = [d.metadata.get("chunk_id") for d in merged]
        metadata["context_expanded"] = True
        metadata["expanded_range"] = (lo, hi)

        return Document(
            page_content="\n\n".join(d.page_content for d in merged),
            metadata=metadata,
        )

    def _collect_range(self, source, page, lo, hi, fallback: dict) -> List[Document]:
        """Looks up chunk_numbers [lo, hi] in the position index, in
        order. `fallback` supplies documents for chunk_numbers that are
        the caller's own surviving chunk(s) - those are guaranteed
        present even if, for some reason, they aren't (yet) in the
        index built from the vector store."""
        collected = []
        for number in range(lo, hi + 1):
            neighbor = fallback.get(number) or self._position_index.get((source, page, number))
            if neighbor is None:
                # Edge of document, or the neighbor isn't in the index
                # (e.g. it was filtered out upstream) - just skip it,
                # merge whatever neighbors were actually found.
                continue
            collected.append(neighbor)
        return collected

import html
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from config import TOP_K, CHUNK_SIZE, CHUNK_OVERLAP
from ingestion.build_index import build_index
from ingestion.registry import build_retriever
from utils.trace import start_trace, clear_trace, record_stage, page_display

# Matches the "[p.N]" citation markers OpenAIGenerator is instructed to
# emit (see GENERATION_CITATION_INSTRUCTION / ask()'s citation_labels,
# which label each excerpt "[p.N]" instead of a plain positional index) -
# used to turn them into links back to the matching source card in
# _linkify_citations() below. The marker names the actual PDF page
# directly, so it's traceable without indirecting through a chunk number.
_CITATION_PATTERN = re.compile(r"\[p\.(\d+)\]")


@dataclass
class RetrievedChunk:
    source: str
    page: Optional[int]
    score: float
    text: str
    expanded: bool = False

    @property
    def page_display(self) -> Optional[int]:
        """1-indexed page number for display/citation - metadata['page']
        is 0-indexed (pypdf/loader convention); this is what a reader
        would actually see printed on the physical PDF page."""
        return self.page + 1 if self.page is not None else None


@dataclass
class AskResult:
    query: str
    answer: str
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    # HTML-escaped `answer` with every "[p.N]" citation turned into a link
    # to the matching source card (#source-pN below) - built once here so
    # the template can render it directly with `| safe` instead of doing
    # citation parsing in Jinja. `answer` itself stays plain text for any
    # caller that just wants the raw generated string (e.g. eval/logging).
    answer_html: str = ""
    # Correlates this answer with its full per-stage breakdown in
    # logs/retrieval_traces.jsonl (see utils/trace.py) - grep the file
    # for this id to see exactly what each retrieval stage did for this
    # specific question (BM25/embedding candidates, RRF fusion, every
    # cross-encoder score including dropped ones, expansion merges).
    request_id: str = ""


def _linkify_citations(answer: str, valid_pages) -> str:
    escaped = html.escape(answer)
    valid_pages = set(valid_pages)

    def _replace(match: "re.Match") -> str:
        page = int(match.group(1))
        if page in valid_pages:
            return f'<a href="#source-p{page}" class="cite">[p.{page}]</a>'
        # Page not among this answer's sources (model hallucinated a page
        # reference, or this text legitimately contains "[p.N]" for some
        # other reason) - leave as plain text rather than link to a
        # source card that doesn't exist.
        return match.group(0)

    return _CITATION_PATTERN.sub(_replace, escaped)


class QueryPipeline:
    """
    Wires the project's proven-best retrieval config together with
    OpenAIGenerator into one ask(query) call, for live end-user queries -
    as opposed to evaluation/run_comparison.py, which rebuilds a fresh
    index per comparison run against the golden dataset.

    Config choice (RecursiveChunker, chunk_size=700/overlap=120, ChromaDB,
    hybrid BM25+embedding fused, cross-encoder reranked, then
    context-expanded) is the best-measured combination so far - see
    evaluation/RETRIEVAL_EVALUATION_JOURNAL.md and
    evaluation/results/context_expansion_comparison_*.csv /
    evaluation/results/ragas_comparison_hybrid_reranked_expanded_*.csv.

    Built once and cached (see get_query_pipeline()) - re-ingesting the
    corpus per request would re-embed every chunk on every question;
    build_index()'s idempotent add() only saves the *storage* write, not
    the embedding compute.

    Rebuilt with fresh=True on every app startup (once per process, via
    __init__ here) - wipes the collection before re-ingesting, so a
    restart always starts from exactly one copy of each chunk instead of
    accumulating duplicates across restarts. This is what actually
    caught and fixed a real duplication bug: differently-cased source
    paths across process runs were silently producing a second chunk_id
    for identical content each time (see
    BaseDocumentLoader.normalize_source), which had cluttered generation
    context with duplicated/near-duplicate chunks.
    """

    CHUNKER = "recursive"
    VECTOR_STORE = "chroma"
    RETRIEVER_KIND = "hybrid_reranked_expanded"

    def __init__(self):
        self.store, self.embedder, self.add_response = build_index(
            self.CHUNKER, self.VECTOR_STORE,
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            fresh=True,
        )
        self.retriever = build_retriever(self.RETRIEVER_KIND, self.store)
        # Lazy: OpenAIGenerator() requires OPENAI_API_KEY at construction
        # time - deferring construction to first ask() means the pipeline
        # (and retrieval alone) can still be built/inspected without a key,
        # and a missing key surfaces as a friendly per-request error
        # instead of failing app startup outright.
        self._generator = None

    @property
    def generator(self):
        if self._generator is None:
            from llm.generator import OpenAIGenerator
            self._generator = OpenAIGenerator()
        return self._generator

    def ask(self, query: str, top_k: int = TOP_K) -> AskResult:
        request_id = uuid.uuid4().hex[:12]
        trace = start_trace(query, request_id)
        try:
            return self._ask(query, top_k, request_id)
        finally:
            trace.write()
            clear_trace()

    def _ask(self, query: str, top_k: int, request_id: str) -> AskResult:
        start = time.perf_counter()
        query_vector = self.embedder.embed_query(query)
        record_stage("query_embedding", vector_dims=len(query_vector))
        retrieved_results = [
            r
            for r in self.retriever.retrieve(
                query_vector, top_k=top_k, query_text=query,
            )
            if r.document is not None
        ]
        retrieval_latency_ms = (time.perf_counter() - start) * 1000
        record_stage(
            "final_retrieval_result",
            latency_ms=round(retrieval_latency_ms, 1),
            count=len(retrieved_results),
            chunks=[
                {
                    "chunk_id": r.document.metadata.get("chunk_id"),
                    "source": r.document.metadata.get("source"),
                    "page_display": page_display(r.document.metadata),
                    "score": round(float(r.score), 4) if r.score is not None else None,
                    "expanded": bool(r.document.metadata.get("context_expanded", False)),
                }
                for r in retrieved_results
            ],
        )

        # Built once, same order, and shared by both the generator (so it
        # can cite "[p.N]") and the source cards below (so "[p.N]"
        # resolves to the right one) - contexts[i]/sources[i]/
        # citation_labels[i]/chunks[i] all describe the same retrieved
        # chunk.
        contexts = [r.document.page_content for r in retrieved_results]
        page_displays = [page_display(r.document.metadata) for r in retrieved_results]
        sources = [
            f"{Path(r.document.metadata.get('source', 'unknown')).name}"
            f"{' p.' + str(page) if page is not None else ''}"
            for r, page in zip(retrieved_results, page_displays)
        ]
        # The bracket marker itself names the page directly (e.g. "p.7")
        # instead of a plain positional index - so a citation in the
        # answer is traceable straight to the source PDF page without
        # indirecting through a chunk number. Falls back to a non-page
        # marker for the rare chunk missing page metadata (never matches
        # _CITATION_PATTERN, so it just can't be cited/linked).
        citation_labels = [
            f"p.{page}" if page is not None else f"source-{i}"
            for i, page in enumerate(page_displays, start=1)
        ]

        generation_result = self.generator.generate(
            query, contexts, sources=sources, citation_labels=citation_labels,
        )
        record_stage(
            "generation",
            context_chunk_count=len(contexts),
            latency_ms=round(generation_result.latency_ms, 1),
            answer=generation_result.answer,
            # The exact excerpts sent to the LLM, verbatim, paired with
            # the citation label each one was tagged with in the prompt
            # (see llm/generator.py:build_messages) - this is what makes
            # "did the model cite the chunk that actually contains the
            # answer?" checkable straight from the trace, without
            # needing the UI open side by side to compare source text.
            context_sent_to_llm=[
                {"citation_label": label, "source": source, "text": text}
                for label, source, text in zip(citation_labels, sources, contexts)
            ],
        )

        chunks = [
            RetrievedChunk(
                source=Path(r.document.metadata.get("source", "unknown")).name,
                page=r.document.metadata.get("page"),
                score=r.score,
                text=r.document.page_content,
                expanded=bool(r.document.metadata.get("context_expanded", False)),
            )
            for r in retrieved_results
        ]
        valid_pages = {c.page_display for c in chunks if c.page_display is not None}

        return AskResult(
            query=query,
            answer=generation_result.answer,
            answer_html=_linkify_citations(generation_result.answer, valid_pages),
            retrieved_chunks=chunks,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_result.latency_ms,
            request_id=request_id,
        )


_pipeline: Optional[QueryPipeline] = None


def get_query_pipeline() -> QueryPipeline:
    """Builds the QueryPipeline on first call and reuses it for every
    subsequent /ask request in this process."""
    global _pipeline
    if _pipeline is None:
        _pipeline = QueryPipeline()
    return _pipeline

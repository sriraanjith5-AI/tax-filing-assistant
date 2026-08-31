import contextvars
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

TRACE_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "retrieval_traces.jsonl"

# Holds the trace for whichever ask() call is currently in flight on
# this async/thread context - not a constructor/retrieve() argument on
# BaseRetriever, since that would mean threading a `trace=None` param
# through every retriever implementation and its ~20 call sites (tests,
# evaluation/run_comparison.py, experiments/), almost none of which have
# a trace to record to. contextvars scope this per request instead, so
# record_stage() below is a silent no-op everywhere except inside one
# QueryPipeline.ask() call.
_current_trace: contextvars.ContextVar = contextvars.ContextVar(
    "current_retrieval_trace", default=None
)


@dataclass
class RetrievalTrace:
    """
    One structured record per pipeline stage of a single ask() call -
    query embedding, each retriever leg (BM25/embedding), RRF fusion,
    cross-encoder rerank (including candidates the threshold dropped,
    not just survivors), context-window expansion/merging, and
    generation - written as one JSON line to
    logs/retrieval_traces.jsonl once the request finishes.

    This is what makes "why did the model cite p.7 over the
    higher-scoring p.2 chunk?" or "why were there duplicate source
    cards?" answerable by reading a log line instead of re-deriving it
    from code inspection each time.
    """

    query: str
    request_id: str
    started_at: float = field(default_factory=time.perf_counter)
    stages: List[Dict[str, Any]] = field(default_factory=list)

    def add_stage(self, name: str, **fields) -> None:
        self.stages.append({
            "stage": name,
            "elapsed_ms": round((time.perf_counter() - self.started_at) * 1000, 1),
            **fields,
        })

    def write(self) -> None:
        TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "request_id": self.request_id,
            "query": self.query,
            "timestamp": time.time(),
            "stages": self.stages,
        }
        with TRACE_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


def start_trace(query: str, request_id: str) -> RetrievalTrace:
    """Activates a new trace for the current context. Must be paired
    with clear_trace() (see QueryPipeline.ask()'s try/finally) so a
    trace never leaks into an unrelated later call on the same thread."""
    trace = RetrievalTrace(query=query, request_id=request_id)
    _current_trace.set(trace)
    return trace


def current_trace() -> Optional[RetrievalTrace]:
    return _current_trace.get()


def clear_trace() -> None:
    _current_trace.set(None)


def record_stage(name: str, **fields) -> None:
    """Appends a stage record to the active trace, if any. No-op when
    nothing has called start_trace() in this context - e.g. the
    evaluation harness and every test, which import and exercise these
    retrievers directly without ever starting a trace."""
    trace = _current_trace.get()
    if trace is not None:
        trace.add_stage(name, **fields)


def page_display(metadata: Dict[str, Any]) -> Optional[int]:
    """1-indexed page number for logging/citation - metadata['page'] is
    0-indexed (pypdf/loader convention). Every trace stage reports THIS
    value, under the key 'page_display', instead of the raw metadata -
    so a page number in the trace always matches the same chunk's
    "[p.N]" citation in the generated answer and its "p.N" source card
    in the UI. Mirrors RetrievedChunk.page_display in query_pipeline.py."""
    page = metadata.get("page")
    return page + 1 if page is not None else None


def summarize_results(results, limit: int = None) -> List[Dict[str, Any]]:
    """Common (chunk_id, source, page_display, chunk_number, score)
    summary used by every retriever's trace record - keeps trace lines
    compact (no full chunk text) while still being enough to answer
    "was this chunk seen here, and at what score/rank?"."""
    items = results if limit is None else results[:limit]
    return [
        {
            "chunk_id": r.document.metadata.get("chunk_id"),
            "source": r.document.metadata.get("source"),
            "page_display": page_display(r.document.metadata),
            "chunk_number": r.document.metadata.get("chunk_number"),
            "score": round(float(r.score), 4) if r.score is not None else None,
        }
        for r in items
    ]

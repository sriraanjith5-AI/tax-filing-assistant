import uuid

from fastapi import FastAPI, BackgroundTasks, Form, Request
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import CHUNK_SIZE, CHUNK_OVERLAP
from ingestion.registry import CHUNKERS, VECTOR_STORES, RETRIEVERS, LOADERS
from evaluation.run_comparison import RunConfig, RunResult, execute_run, load_run_history
from retrieval.query_pipeline import get_query_pipeline

app = FastAPI(title="Tax Filing Assistant")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.on_event("startup")
def _warm_query_pipeline():
    # Builds the /ask index (ingest -> embed -> store) and retriever at
    # server boot instead of on the first user request - otherwise
    # whoever asks the first question eats the ~1-2 min ingestion cost
    # (see retrieval/query_pipeline.py). Only builds the retrieval side;
    # OpenAIGenerator stays lazy (constructed on first ask()) so a
    # missing OPENAI_API_KEY doesn't block startup - it's not needed
    # until someone actually asks a question.
    get_query_pipeline()

# In-memory run registry (run_id -> RunResult). Cleared on server
# restart; full run history is persisted separately to
# evaluation/results/comparison_runs.csv (see load_run_history()).
RUNS: dict[str, RunResult] = {}


def _best_run(history) -> dict | None:
    if history is None or history.empty or "ndcg_at_10" not in history.columns:
        return None
    done = history[history["status"] == "done"]
    if done.empty:
        return None
    return done.loc[done["ndcg_at_10"].idxmax()].to_dict()


@app.get("/")
def home(request: Request):
    history = load_run_history()
    history_rows = history.to_dict(orient="records")[::-1] if not history.empty else []
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "loaders": LOADERS,
            "chunkers": list(CHUNKERS.keys()),
            "vector_stores": VECTOR_STORES,
            "retrievers": RETRIEVERS,
            "default_chunk_size": CHUNK_SIZE,
            "default_chunk_overlap": CHUNK_OVERLAP,
            "best_run": _best_run(history),
            "history_rows": history_rows,
            "history_columns": list(history.columns) if not history.empty else [],
        },
    )


@app.post("/runs")
def create_run(
    background_tasks: BackgroundTasks,
    chunker: str = Form(...),
    vector_store: str = Form(...),
    retriever: str = Form(...),
    chunk_size: int = Form(CHUNK_SIZE),
    chunk_overlap: int = Form(CHUNK_OVERLAP),
    loader: str = Form("pypdf"),
    include_ragas: bool = Form(False),
    include_generation: bool = Form(False),
):
    run_id = uuid.uuid4().hex[:8]
    config = RunConfig(
        chunker=chunker,
        vector_store=vector_store,
        retriever=retriever,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        loader=loader,
        include_ragas=include_ragas,
        include_generation=include_generation,
    )
    RUNS[run_id] = RunResult(run_id=run_id, config=config)
    background_tasks.add_task(execute_run, run_id, config, RUNS)
    return RedirectResponse(url=f"/runs/{run_id}", status_code=303)


@app.get("/runs/{run_id}")
def run_page(request: Request, run_id: str):
    result = RUNS.get(run_id)
    return templates.TemplateResponse(
        request,
        "run.html",
        {"run_id": run_id, "run": result},
    )


@app.get("/runs/{run_id}/status")
def run_status(run_id: str):
    result = RUNS.get(run_id)
    if result is None:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return JSONResponse(result.to_dict())


@app.get("/ask")
def ask_form(request: Request):
    return templates.TemplateResponse(request, "ask.html", {})


@app.post("/ask")
def ask(request: Request, query: str = Form(...)):
    """
    Live single-turn Q&A: retrieves + generates an answer for one
    question using the project's best-measured retriever config
    (see retrieval/query_pipeline.py), as opposed to /runs which
    evaluates a config against the golden dataset in bulk.

    The pipeline (index + retriever) is built once and cached across
    requests; only OPENAI_API_KEY / a bad query can fail per-request.
    """
    context = {"query": query}
    try:
        pipeline = get_query_pipeline()
        context["result"] = pipeline.ask(query)
    except Exception as exc:
        context["error"] = str(exc)
    return templates.TemplateResponse(request, "ask.html", context)

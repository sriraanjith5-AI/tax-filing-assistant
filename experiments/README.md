# Experiments

Superseded, abandoned, or never-finished code, kept for reference rather
than deleted outright. Nothing here is imported by the current pipeline
(`app.py`, `ingestion/`, `retrieval/`, `llm/`, or `evaluation/`'s live
components — `retrieval_metrics.py`, `ragas_retrieval_evaluator.py`,
`generation_metrics.py`, `run_comparison.py`).

See [`evaluation/RETRIEVAL_EVALUATION_JOURNAL.md`](../evaluation/RETRIEVAL_EVALUATION_JOURNAL.md) for the full story of how retrieval evaluation evolved and why each of these was tried and set aside.

## `experiments/evaluation/`

| File | Why it's here |
|---|---|
| `rag_evaluator.py` | An earlier, complete evaluation harness (hit@k/recall@k/reciprocal_rank via stable evidence-text matching, no LLM judge) - superseded by the newer, currently-used combination of `evaluation/retrieval_metrics.py` (classical) + `evaluation/ragas_retrieval_evaluator.py` + `evaluation/generation_metrics.py` (LLM-judged) that `run_comparison.py` and the live `/runs` dashboard actually use. Not imported by anything in the current pipeline, but still exercised by `tests/test_evaluator*.py` (see "Still referencing these" below). |
| `rag_evaluator_v0.py`, `rag_evaluator_v1.py`, `rag_evaluator_v2.py` | Earlier drafts of `rag_evaluator.py` above. Not imported by anything, including their own like-numbered test files - `tests/test_evaluator_v0.py`/`_v1.py`/`_v2.py` all import the final `rag_evaluator.py`, not these drafts. Kept purely as a historical record of how that evaluator evolved. |
| `evaluation_dataclass.py` | The `EvaluationResult` dataclass `rag_evaluator.py` (and its drafts above) return results as. Only ever used by those files, so it moved with them. |
| `golden_dataset.json`, `golden_dataset_gt_v0.json` | Earlier golden datasets, superseded by `evaluation/golden_dataset_with_retrieval_ground_truth.json`. |
| `golden_case_dataclass.py` | Unused dataclass, never wired into the current pipeline. |
| `golden_dataset_loader.py` | Empty stub, never implemented. |

## `experiments/retrieval/`

| File | Why it's here |
|---|---|
| `query_expander.py`, `query_expansion_retriever.py` | LLM-based query expansion — tried as a precision improvement (Step 6–7 of the journal) and dropped after it regressed both precision and recall in testing. Kept in case it's worth revisiting with a tighter rewrite prompt. |

## `experiments/cleaners/`

| File | Why it's here |
|---|---|
| `dummy_cleaner.py` | A trivial demo `BaseCleaner` (just appends a `[CLEANED]` marker) written to exercise the cleaner pipeline during development. Only ever referenced by a commented-out import in `tests/test_pdfloader.py` - never actually used. |
| `ocr_cleaner.py` | Empty stub, never implemented - presumably intended for cleaning up OCR artifacts if a scanned (rather than digitally-native) PDF were ever ingested. |

## `experiments/ingestion/`

| File | Why it's here |
|---|---|
| `embedder.py` | Empty stub, never implemented - superseded by `embedding/huggingface_embedder.py`, which is what the current pipeline actually uses. |

## Still referencing these

`tests/test_evaluator_v0.py` and `tests/test_evaluator_v1.py` are themselves
superseded by `tests/test_evaluator.py` / `tests/test_evaluator_v2.py`, but
still point at `experiments/evaluation/golden_dataset.json` so they remain
runnable rather than silently broken. All four of `tests/test_evaluator*.py`
import `RAGEvaluator` from `experiments/evaluation/rag_evaluator.py` - moving
that file here didn't break them, since their `sys.path.insert(PROJECT_ROOT)`
makes `experiments` importable as a namespace package, same as any other
top-level package in this project.

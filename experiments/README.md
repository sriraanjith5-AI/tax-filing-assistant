# Experiments

Superseded and abandoned code, kept for reference rather than deleted outright. Nothing here is imported by the current pipeline (`evaluation/`, `retrieval/`, or `llm/`).

See [`evaluation/RETRIEVAL_EVALUATION_JOURNAL.md`](../evaluation/RETRIEVAL_EVALUATION_JOURNAL.md) for the full story of how retrieval evaluation evolved and why each of these was tried and set aside.

## `experiments/evaluation/`

| File | Why it's here |
|---|---|
| `rag_evaluator_v0.py`, `rag_evaluator_v1.py`, `rag_evaluator_v2.py` | Earlier drafts of `evaluation/rag_evaluator.py`, superseded by the current version. |
| `golden_dataset.json`, `golden_dataset_gt_v0.json` | Earlier golden datasets, superseded by `evaluation/golden_dataset_with_retrieval_ground_truth.json`. |
| `golden_case_dataclass.py` | Unused dataclass, never wired into the current pipeline. |
| `golden_dataset_loader.py` | Empty stub, never implemented. |

## `experiments/retrieval/`

| File | Why it's here |
|---|---|
| `query_expander.py`, `query_expansion_retriever.py` | LLM-based query expansion — tried as a precision improvement (Step 6–7 of the journal) and dropped after it regressed both precision and recall in testing. Kept in case it's worth revisiting with a tighter rewrite prompt. |

## Still referencing these

`tests/test_evaluator_v0.py` and `tests/test_evaluator_v1.py` are themselves superseded by `tests/test_evaluator.py` / `tests/test_evaluator_v2.py`, but still point at `experiments/evaluation/golden_dataset.json` so they remain runnable rather than silently broken.

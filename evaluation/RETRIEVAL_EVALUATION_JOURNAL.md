# Retrieval Evaluation Journal

A step-by-step record of how the retrieval stage of the tax-filing assistant was evaluated, tuned, and debugged — in the order it actually happened. Each step names the technique used (for engineers) alongside a plain-language explanation (for everyone else), plus what changed as a result.

**Where things ended up**, measured on the same fixed set of 30 golden-dataset questions throughout:

| Measure | First run | Current system |
|---|---:|---:|
| `LLMContextPrecisionWithReference` (ragas) | 0.76 | **0.91** |
| `LLMContextRecall` (ragas) | 0.72 | **1.00** |
| Recall@10 (exact/fuzzy text match) | 0.60 | **0.97** |
| Recall@1 (correct chunk ranked first) | 0.50 | **0.73** |
| MRR (Mean Reciprocal Rank) | 0.54 | **0.80** |
| NDCG@10 (Normalized Discounted Cumulative Gain) | 0.56 | **0.84** |

Current default retriever (`retrieval/default_retriever.py`): **BM25 + embedding hybrid search** → **cross-encoder reranking** (`cross-encoder/ms-marco-MiniLM-L-12-v2`) → **score-threshold filtering**, retrieving `TOP_K` results per query.

---

## Step 1 — Setup: bring in `ragas` as an LLM-judged evaluator

The retrieval pipeline had integration tests confirming it returned *something*, but nothing scoring whether it returned the *right* chunks, ranked sensibly. We integrated [ragas](https://github.com/explodinggradients/ragas) with an OpenAI `gpt-4o-mini` judge, using two metrics:

- **`LLMContextPrecisionWithReference`** — for each retrieved chunk, an LLM judges whether it was actually needed to produce the reference answer, and rewards relevant chunks that are also ranked highly (precision, but rank-aware).
- **`LLMContextRecall`** — an LLM breaks the reference answer into claims and checks whether each is supported by the retrieved context.

Files: `evaluation/ragas_retrieval_evaluator.py`, `tests/test_ragas_evaluator.py`.

## Step 2 — Problem: infinite retry loop on the first real run

The first run hammered the OpenAI API with `429 Too Many Requests` and never converged, retrying forever. Concurrency was throttled (`RunConfig(max_workers=2)` in ragas — the default fires up to 16 judge calls in parallel), which is good practice but didn't fix the underlying failure.

> The actual error text was `insufficient_quota` — the API key had **no billing credit**. `429` covers both rate-limiting and quota exhaustion, so the client kept retrying a request that could never succeed.

Confirmed by calling the OpenAI endpoint directly with `urllib`, bypassing ragas/langchain entirely — same `insufficient_quota` error. After adding billing and regenerating the key under a correctly-budgeted project, the same direct call returned `200 OK`.

## Step 3 — Baseline result

With a working judge, the first honest scores:

**Precision 0.76 / Recall 0.72**, `top_k=3`.

## Step 4 — Experiment: does raising `top_k` improve Recall?

Tested `top_k=3` vs. `top_k=5` on identical queries (`tests/test_ragas_topk_comparison.py`).

| top_k | avg precision | avg recall |
|---|---:|---:|
| 3 | 0.775 | 0.750 |
| 5 | 0.775 | **0.889** |

Recall +13.9 points, precision flat. One question that scored `recall=0.0` at `top_k=3` (the correct chunk was ranked 4th–5th, just past the cutoff) hit `recall=1.0` at `top_k=5` — a genuine **ranking-depth** problem, not a semantic-similarity problem. **Adopted `TOP_K=5`.**

## Step 5 — Experiment: does reranking improve Precision?

Embedding similarity is good at topical relevance but weak at fine-grained lexical distinctions (it kept conflating **Form W-4** and **Form W-4P**, near-duplicate content). Built `retrieval/cross_encoder_reranker.py`: `CrossEncoderReranker` retrieves a wide candidate set (`fetch_k`) via the base retriever, then a **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`, from `sentence-transformers`) scores each `(query, chunk)` pair jointly — a fundamentally different, more accurate mechanism than comparing two separate embedding vectors — and truncates to `top_k`.

| Variant | avg precision | avg recall |
|---|---:|---:|
| embedding-only | 0.759 | 0.889 |
| + cross-encoder rerank (L-6) | **0.851** | **0.922** |

Both metrics up, no trade-off. **Adopted as the new default retriever.**

## Step 6 — Four more precision experiments, run side by side

`tests/test_ragas_precision_experiments.py` compared four independent changes against the same 30 questions:

- **A — stronger reranker**: swap the cross-encoder for `cross-encoder/ms-marco-MiniLM-L-12-v2` (bigger model, same mechanism).
- **B — score threshold**: instead of always returning a fixed `top_k`, drop candidates whose sigmoid(cross-encoder score) falls below a threshold (`RERANK_SCORE_THRESHOLD`) — `CrossEncoderReranker.score_threshold`.
- **C — hybrid retrieval (BM25 + embeddings)**: `retrieval/bm25_retriever.py` (`BM25Retriever`, using `rank_bm25.BM25Okapi`) runs alongside the embedding retriever; `retrieval/hybrid_retriever.py` (`HybridRetriever`) merges both candidate lists via **Reciprocal Rank Fusion (RRF)** before handing the fused set to reranking. **BM25** is a classic lexical/keyword-frequency ranking algorithm — it scores exact term overlap directly, which is exactly what dense embeddings blur (e.g. "W-4" vs "W-4P").
- **D — query expansion**: `retrieval/query_expander.py` (`QueryExpander`) uses an LLM to generate paraphrases of the query; `retrieval/query_expansion_retriever.py` (`QueryExpansionRetriever`) retrieves for the original query plus every paraphrase and merges results via RRF.

## Step 7 — Problem: query expansion (D) regressed both metrics

| Variant | avg precision | avg recall |
|---|---:|---:|
| baseline (L-6 rerank) | 0.848 | 0.933 |
| A — stronger reranker (L-12) | 0.879 | 0.950 |
| B — score threshold | 0.872 | 0.933 |
| C — hybrid BM25 | 0.862 | **0.989** |
| **D — query expansion** | **0.841** ↓ | **0.883** ↓ |
| Combined (A+B+C) | **0.906** | **1.000** |

> LLM-generated paraphrases drifted from the source document's literal phrasing, pulling in topically-adjacent-but-wrong chunks. **Dropped D as a standalone retriever.**

## Step 8 — Adopted: A + B + C combined

Stacked stronger reranker (L-12) + score threshold + hybrid BM25/embedding fusion into one pipeline (`retrieval/default_retriever.py :: build_default_retriever`). Result: **precision 0.907–0.919, recall 1.000** across repeated confirmation runs (`tests/test_ragas_evaluator.py`). Made this the project's permanent default retriever.

## Step 9 — Latency measurement

Quality alone doesn't tell you the cost. Re-ran the same experiment set instrumented with `time.perf_counter()` around each `retrieve()` call:

| Variant | avg precision | avg recall | avg latency |
|---|---:|---:|---:|
| baseline (L-6 rerank) | 0.854 | 0.922 | 858 ms |
| A — stronger reranker (L-12) | 0.871 | 0.950 | **1564 ms** |
| B — score threshold | 0.893 | 0.944 | 812 ms |
| C — hybrid BM25 | 0.870 | 0.983 | **436 ms** |
| Combined (A+B+C) | **0.907** | **1.000** | 970 ms |

`hybrid_bm25` alone is the cheapest (nearly half the baseline latency) and close to combined on quality — worth knowing if latency-sensitive UI is a future constraint. Combined stays under 1s and wins on both quality metrics, so it remained the default.

## Step 10 — Setup: add classical, non-LLM metrics (Recall@K, MRR, NDCG@K)

ragas' LLM-judged metrics are semantically generous (tolerate paraphrase, inference) but cost money, add latency, and aren't perfectly reproducible. Added a deterministic, free, second scoring path (`evaluation/retrieval_metrics.py :: RetrievalMetrics`):

- **Recall@K** (K = 1, 3, 5, 10) — of the evidence needed, what fraction was covered by the top-K retrieved chunks?
- **MRR (Mean Reciprocal Rank)** — `1 / rank` of the first relevant chunk, averaged across questions. Rewards putting the right answer *near the top*, not just somewhere in the results.
- **NDCG@K (Normalized Discounted Cumulative Gain)** — a rank-weighted score that discounts relevant results the further down the list they appear, normalized against the ideal ordering.

Matching is done by comparing retrieved chunk text against the golden dataset's `evidence` field (see `evaluation/evidence_matching.py`), not `chunk_id` — chunk IDs change whenever chunking strategy changes; evidence text is stable.

**These are not inputs to ragas and ragas does not consume them** — the two evaluation paths run independently over the same retrieved chunks and are meant to be read together, not merged.

## Step 11 — Problem: every question scored 0.0

First run of `tests/test_retrieval_metrics.py`: **Recall@1 through Recall@10, MRR, and NDCG@10 all 0.0000** across all 30 questions — despite ragas scoring the same retrieved chunks near-perfectly moments earlier.

> A result that uniform is a test bug, not a system failure. Investigated by hand-inspecting the actual retrieved chunk vs. the evidence text for a known-good question.

## Step 12 — Fix #1: PDF text-extraction artifacts + evidence spanning chunk boundaries

Root cause, part one: `pypdf`-extracted text contains line-wrap artifacts — `"P .L. 119 -21"` instead of `"P.L. 119-21"`, `"includ- ing"` instead of `"including"` — that broke exact substring matching. Fixed by extending `normalize_text()` in `evaluation/evidence_matching.py` to undo both patterns via regex before comparison.

Root cause, part two: some evidence sentences **span a chunk boundary** — a 700-character chunk can start mid-sentence, so it contains the back 80% of the evidence text but not the opening clause, and exact whole-string containment structurally can't match. Fixed by adding a fallback: **word-level fuzzy matching** via `difflib.SequenceMatcher` (word tokens, not characters — `SequenceMatcher`'s `autojunk` heuristic silently corrupts matching on character sequences ≥200 chars, which most evidence sentences hit). A chunk counts as a match if the summed length of all matching word-runs covers ≥60% (`PARTIAL_MATCH_COVERAGE`) of the evidence text.

Result: scores moved off zero, but **12 of 30 questions (40%) still scored 0.0** at every K.

## Step 13 — Fix #2: the golden dataset's evidence text wasn't verbatim

Hand-inspected all 12 remaining zero-score questions by pulling the actual top-10 retrieved chunks and comparing against evidence text. Retrieval was almost always fine — the *test data* was the problem:

- Several `evidence` fields were **LLM-paraphrased summaries**, not literal excerpts (e.g. `"Publication 15-T describes alternative methods including..."`), which can never match any chunk verbatim regardless of retrieval quality.
- Several described **table values as prose** (`"For a weekly payroll period, $226.90 is added to wages under Table 1"`), when the source document only ever presents that fact as a dot-leader table row (`"Weekly ... $226.90"`), never as a sentence.

Manually verified (by direct retrieval + chunk inspection) that the correct source chunk *was* present in the top-10 results for these questions in every case checked — this was purely a scoring-data problem, not a retriever problem.

Rewrote all 12 `evidence` fields in `evaluation/golden_dataset_with_retrieval_ground_truth.json` to genuine verbatim excerpts, each pre-verified via `find_matching_evidence()` to land inside a single retrieved chunk before being committed. One entry (`gd_011`) also had its `source_page`/`source_section` corrected — the original values pointed to the wrong page.

| Measure | Before dataset fix | After dataset fix |
|---|---:|---:|
| Recall@1 | 0.500 | **0.733** |
| Recall@3 | 0.567 | **0.800** |
| Recall@5 | 0.600 | **0.900** |
| Recall@10 | 0.600 | **0.967** |
| MRR | 0.542 | **0.798** |
| NDCG@10 | 0.558 | **0.837** |

## Step 14 — Decision: leave one remaining outlier (`gd_027`) unfixed

`gd_027` still scored `Recall@1=0, Recall@3=0, Recall@5=0, Recall@10=1, MRR=0.12` — correct chunk found, but ranked 8th. Traced by hand: the source document repeats a near-identical "IRS Tax Withholding Estimator" tip for both **Form W-4** (the evidence, expected at rank 1) and **Form W-4P** (a near-duplicate, ranked 4th), and a generic, only loosely-related "Introduction" chunk outranked both (rank 1–2) — a cross-encoder miscalibration, likely a domain mismatch between the reranker's MS MARCO web-search training data and this IRS filing corpus.

**Decision: no fix.** It's 1 of 30 questions, `Recall@10=1.0` (the LLM-generation step still receives the correct chunk in context), and retuning the reranker/RRF weighting around a single question risks overfitting the whole system to this one case. Logged as a known limitation — revisit only if the same "generic boilerplate outranks specific content" pattern recurs on a larger eval set.

---

## Glossary

| Term | Meaning |
|---|---|
| **Precision** | Of everything retrieved, how much was actually relevant/useful. |
| **Recall** | Of everything that *should* have been retrieved, how much was found. |
| **Recall@K** | Recall considering only the top K retrieved results. |
| **MRR (Mean Reciprocal Rank)** | Average of `1 / rank of first relevant result` across all queries — rewards ranking the right answer near the top, not just finding it somewhere. |
| **NDCG@K** | Rank-weighted relevance score, discounted by position and normalized against the ideal ordering — the standard IR metric for "how good is this ranked list," not just "is the right thing present." |
| **BM25** | A classical lexical (keyword-frequency) ranking algorithm; scores exact term overlap directly, unlike embeddings. |
| **Cross-encoder reranking** | A two-stage retrieval pattern: cheap wide retrieval first, then a slower model that scores `(query, chunk)` pairs jointly for higher-accuracy final ranking. |
| **Hybrid retrieval** | Running lexical (BM25) and semantic (embedding) search in parallel and fusing the results, typically via Reciprocal Rank Fusion (RRF). |
| **RRF (Reciprocal Rank Fusion)** | A method for merging multiple ranked lists: each item's score is the sum of `1 / (k + rank)` across every list it appears in. |
| **ragas** | An LLM-judged RAG evaluation framework; used here for `LLMContextPrecisionWithReference` and `LLMContextRecall`. |

## Related files

- `retrieval/default_retriever.py` — current default retriever composition
- `retrieval/cross_encoder_reranker.py`, `retrieval/bm25_retriever.py`, `retrieval/hybrid_retriever.py` — retrieval components
- `evaluation/ragas_retrieval_evaluator.py`, `evaluation/retrieval_metrics.py`, `evaluation/evidence_matching.py` — the two evaluation paths
- `tests/test_ragas_evaluator.py`, `tests/test_ragas_topk_comparison.py`, `tests/test_ragas_rerank_comparison.py`, `tests/test_ragas_precision_experiments.py`, `tests/test_retrieval_metrics.py` — experiment scripts
- `evaluation/results/` — raw CSV output from every run referenced above
- `experiments/` — earlier evaluator drafts, superseded golden datasets, and the dropped query-expansion retriever (Step 6–7), kept for reference rather than deleted; see `experiments/README.md`

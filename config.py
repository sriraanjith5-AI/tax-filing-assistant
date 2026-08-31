from pathlib import Path

# ----------------------------
# Project Paths
# ----------------------------

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"

PDF_DIR = DATA_DIR / "pdf"

CSV_DIR = DATA_DIR / "csv"

EXCEL_DIR = DATA_DIR / "excel"

VECTOR_DB_DIR = BASE_DIR / "vectorstore"

EVAL_DIR = BASE_DIR / "evaluation"


# ----------------------------
# Chunking
# ----------------------------

CHUNK_SIZE = 700

CHUNK_OVERLAP = 120

CHUNK_MIN_SIZE = 200

# ----------------------------
# Embedding Model
# ----------------------------

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 2


# ----------------------------
# Retrieval
# ----------------------------

TOP_K = 5


# ----------------------------
# Reranking
# ----------------------------
# CrossEncoderReranker is the default retriever: VectorRetriever
# fetches RERANK_FETCH_K wide candidates by embedding similarity,
# then the cross-encoder reranks and truncates to TOP_K.
# Confirmed via evaluation/results/ragas_rerank_comparison_*.csv:
# avg precision 0.7587 -> 0.8508, avg recall 0.8889 -> 0.9222.

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_FETCH_K = 15

# Stronger (larger) cross-encoder, tried as a precision experiment
# against RERANK_MODEL. See evaluation/results/ragas_precision_
# experiments_*.csv for the comparison.
RERANK_MODEL_STRONG = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Minimum sigmoid(cross-encoder score) a candidate must clear to be
# kept, when using CrossEncoderReranker's score_threshold option
# instead of a fixed top_k cutoff.
RERANK_SCORE_THRESHOLD = 0.5

# ----------------------------
# Context Expansion
# ----------------------------
# ContextExpandingRetriever (retrieval/context_expander.py) is a
# post-retrieval step: for each chunk that survives reranking, its
# +/- CONTEXT_EXPANSION_WINDOW neighbors (by document position) are
# merged in before the generator sees it, so a chunk-size cutoff
# landing mid-thought doesn't cut off context the generator needs.
# window=1 merges [n-1, n, n+1]; set to 0 to disable expansion.
CONTEXT_EXPANSION_WINDOW = 1

# Number of LLM-generated paraphrases QueryExpander produces per
# query for QueryExpansionRetriever.
QUERY_EXPANSION_VARIANTS = 2


# ----------------------------
# Ragas Evaluation (LLM judge)
# ----------------------------

# Model used as the ragas judge for LLM-based context metrics
# (LLMContextPrecisionWithReference, LLMContextRecall).
# Requires OPENAI_API_KEY to be set (e.g. in a .env file).
RAGAS_JUDGE_MODEL = "gpt-4o-mini"


# ----------------------------
# Generation (LLM answer synthesis)
# ----------------------------

# Model used to generate an answer from retrieved context (llm/generator.py).
# Same tier as RAGAS_JUDGE_MODEL, so generation and judging cost are
# comparable. Requires OPENAI_API_KEY to be set (e.g. in a .env file).
GENERATION_MODEL = "gpt-4o-mini"

# Deterministic, so repeated runs of the same config are reproducible/
# comparable - same reasoning as using a fixed judge temperature for ragas.
GENERATION_TEMPERATURE = 0

# Instructs the model to answer strictly from the retrieved context and say
# so when the context doesn't contain the answer, rather than falling back
# on its own knowledge. This is also what makes the Faithfulness score
# meaningful - an ungrounded system prompt would let the model answer
# correctly from prior knowledge while still scoring low on faithfulness
# (or vice versa), muddying what the metric is actually measuring.
#
# The length/scope instruction below was added after diagnosing why
# AnswerCorrectness scored noticeably lower than Faithfulness/AnswerRelevancy
# despite answers being fully grounded and on-topic: AnswerCorrectness is
# scored by statement-overlap against the golden dataset's terse reference
# answers, so a longer/more-detailed-but-correct answer loses points for
# "extra" content the reference didn't mention. Matching the reference's
# answer length directly targets that, on top of being better UX anyway.
GENERATION_SYSTEM_PROMPT = (
    "You are a tax filing assistant. Answer the user's question using ONLY "
    "the provided context excerpts below. If the context does not contain "
    "enough information to answer the question, say so explicitly instead "
    "of guessing or using outside knowledge. Answer in 1-2 sentences, "
    "stating only the specific fact(s) the question asks for - do not add "
    "related details, background, or caveats the question didn't ask about. "
    "Cite specific figures/rules from the context where relevant."
)

# Appended to GENERATION_SYSTEM_PROMPT only when build_messages() is given
# `sources` (i.e. live /ask answers - see retrieval/query_pipeline.py's
# _linkify_citations, which turns these bracket numbers into links back to
# the matching source card). NOT used for evaluation/run_comparison.py's
# generation runs (it never passes `sources`), so the golden-dataset
# faithfulness/answer_relevancy/answer_correctness baselines in
# evaluation/results/ stay comparable to before citations were added -
# bracket markers in the answer text would otherwise skew those scores
# against reference answers that were never written to contain them.
GENERATION_CITATION_INSTRUCTION = (
    " Each context excerpt is labeled with a bracketed page reference, e.g. "
    "\"[p.7] ...\" - after every claim you make, add the exact bracketed "
    "page reference(s) it came from, copied verbatim (e.g. \"the standard "
    "deduction is $14,600 [p.7].\"), so the claim can be checked against "
    "that page of the source document. Never invent a page reference that "
    "wasn't shown in the context."
)
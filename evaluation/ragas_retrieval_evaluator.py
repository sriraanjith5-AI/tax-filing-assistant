import os

from dotenv import load_dotenv

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI

from config import RAGAS_JUDGE_MODEL


class RagasRetrievalEvaluator:
    """
    Evaluates retrieval quality using ragas' LLM-judged context metrics.

    This complements RAGEvaluator (evaluation/rag_evaluator.py), which
    scores Hit@K / Recall@K / MRR via exact substring matching against
    stable evidence text. That approach is precise but blind to chunks
    that are semantically relevant without literally containing the
    evidence string.

    Metrics used here:

    - LLMContextPrecisionWithReference: for each retrieved chunk, an
      LLM judges whether it was necessary to arrive at the reference
      answer, and rewards chunks that are both relevant AND ranked
      highly (precision@k, ranking-aware).

    - LLMContextRecall: an LLM breaks the reference answer into
      claims/sentences and checks whether each one is supported by the
      retrieved context, i.e. did retrieval bring back everything
      needed.

    Both metrics need:
      - user_input          : the query
      - retrieved_contexts   : chunk text actually returned by the
                                retriever, in rank order
      - reference             : the expected/reference answer

    Requires OPENAI_API_KEY (see .env.example / config.RAGAS_JUDGE_MODEL).
    """

    def __init__(self, judge_model: str = None, max_workers: int = 2):

        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to a .env file at the "
                "project root before running ragas evaluation."
            )

        judge_model = judge_model or RAGAS_JUDGE_MODEL

        chat_model = ChatOpenAI(
            model=judge_model,
            temperature=0,
        )

        self.judge_llm = LangchainLLMWrapper(chat_model)

        self.metrics = [
            LLMContextPrecisionWithReference(llm=self.judge_llm),
            LLMContextRecall(llm=self.judge_llm),
        ]

        # ------------------------------------------------------
        # Rate-limit safety.
        #
        # ragas' default RunConfig fires up to 16 judge calls
        # concurrently. Most OpenAI accounts (especially free/low
        # tier) can't sustain that and end up in a 429 retry storm
        # that looks like a hang. max_workers throttles concurrency
        # so requests are spaced out instead of all firing at once.
        # Raise it if your account's rate limit comfortably allows
        # more parallel calls.
        # ------------------------------------------------------

        self.run_config = RunConfig(max_workers=max_workers)

    # ============================================================
    # SAMPLE CONSTRUCTION
    # ============================================================

    def build_sample(
        self,
        query,
        retrieved_results,
        expected_answer,
    ):
        """
        Convert one retriever call + golden-dataset entry into a
        ragas sample dict.
        """

        retrieved_contexts = [
            result.document.page_content
            for result in retrieved_results
            if result.document is not None
        ]

        return {
            "user_input": query,
            "retrieved_contexts": retrieved_contexts,
            "reference": expected_answer,
        }

    # ============================================================
    # EVALUATION
    # ============================================================

    def evaluate(self, samples):
        """
        Run ragas evaluate() over a list of sample dicts built with
        build_sample().

        Returns a ragas EvaluationResult (supports .to_pandas()).
        """

        dataset = EvaluationDataset.from_list(samples)

        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.judge_llm,
            run_config=self.run_config,
        )

        return result

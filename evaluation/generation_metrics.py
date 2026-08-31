import os

from dotenv import load_dotenv

from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from config import RAGAS_JUDGE_MODEL, EMBEDDING_MODEL


class RagasGenerationEvaluator:
    """
    Scores LLM-generated answers (llm/generator.py) using ragas' generation-
    quality metrics - complements RagasRetrievalEvaluator
    (evaluation/ragas_retrieval_evaluator.py), which scores the retrieved
    context itself rather than an answer synthesized from it.

    Metrics used here:

    - Faithfulness: breaks the answer into individual claims and checks each
      is actually supported by the retrieved context (catches hallucination/
      claims not grounded in what was retrieved).

    - AnswerRelevancy: checks the answer actually addresses the question
      asked (an answer can be fully faithful to context yet not answer the
      question).

    - AnswerCorrectness: compares the answer against the golden dataset's
      expected_answer (factuality + semantic similarity), i.e. is the
      answer actually right, not just grounded/on-topic.

    Faithfulness needs only an LLM judge; AnswerRelevancy/AnswerCorrectness
    also need an embeddings model. Reuses the project's own BGE embeddings
    (langchain_huggingface.HuggingFaceEmbeddings, same as
    chunking/semantic_chunker.py) rather than requiring a second paid API -
    only the judge LLM needs OPENAI_API_KEY.

    Requires OPENAI_API_KEY (see .env.example / config.RAGAS_JUDGE_MODEL).
    """

    def __init__(self, judge_model: str = None, max_workers: int = 2):

        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to a .env file at the "
                "project root before running generation evaluation."
            )

        judge_model = judge_model or RAGAS_JUDGE_MODEL

        chat_model = ChatOpenAI(
            model=judge_model,
            temperature=0,
        )

        self.judge_llm = LangchainLLMWrapper(chat_model)
        self.embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        )

        self.metrics = [
            Faithfulness(llm=self.judge_llm),
            AnswerRelevancy(llm=self.judge_llm, embeddings=self.embeddings),
            AnswerCorrectness(llm=self.judge_llm, embeddings=self.embeddings),
        ]

        # Same rate-limit safety as RagasRetrievalEvaluator - throttle
        # concurrent judge calls instead of firing ragas' default (up to 16
        # at once), which most OpenAI accounts can't sustain without
        # hitting a 429 retry storm.
        self.run_config = RunConfig(max_workers=max_workers)

    # ============================================================
    # SAMPLE CONSTRUCTION
    # ============================================================

    def build_sample(
        self,
        query,
        retrieved_results,
        answer,
        expected_answer,
    ):
        """
        Convert one generate() call + its retrieved context + golden-
        dataset entry into a ragas sample dict.
        """

        retrieved_contexts = [
            result.document.page_content
            for result in retrieved_results
            if result.document is not None
        ]

        return {
            "user_input": query,
            "response": answer,
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

        Returns a ragas EvaluationResult (supports .to_pandas()), with
        columns "faithfulness", "answer_relevancy", "answer_correctness".
        """

        dataset = EvaluationDataset.from_list(samples)

        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.judge_llm,
            run_config=self.run_config,
        )

        return result

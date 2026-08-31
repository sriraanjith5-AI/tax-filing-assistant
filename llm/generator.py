import os
import time
from typing import List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from config import (
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    GENERATION_SYSTEM_PROMPT,
    GENERATION_CITATION_INSTRUCTION,
)
from llm.base_generator import BaseGenerator
from llm.generation_dataclass import GenerationResult
from utils.logger import logging

logger = logging.getLogger(__name__)


class OpenAIGenerator(BaseGenerator):
    """
    Generates an answer from retrieved context via ChatOpenAI - same
    instantiation pattern as the ragas judge LLM
    (evaluation/ragas_retrieval_evaluator.py), just used for direct answer
    synthesis instead of as a ragas judge.

    Requires OPENAI_API_KEY (see .env.example / config.GENERATION_MODEL).
    """

    def __init__(self, model: str = None, temperature: float = None):
        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to a .env file at the "
                "project root before running answer generation."
            )

        self.model = ChatOpenAI(
            model=model or GENERATION_MODEL,
            temperature=temperature if temperature is not None else GENERATION_TEMPERATURE,
        )

    @staticmethod
    def build_messages(
        query: str,
        contexts: List[str],
        sources: Optional[List[str]] = None,
        citation_labels: Optional[List[str]] = None,
    ) -> list:
        """
        Parameters
        ----------
        sources : list[str], optional
            One human-readable label per context (e.g. "IRS_Publication15T.pdf
            p.7"), same length/order as `contexts`. When given, each excerpt
            is tagged with its source AND the system prompt gains an
            instruction to cite the bracketed marker, so a caller can
            resolve it back to a specific document/page (see
            retrieval/query_pipeline.py, which does exactly that for the
            /ask endpoint). Omit for callers that only have bare chunk text
            (e.g. the evaluation harness) - excerpts are still numbered, but
            neither the source tag nor the citation instruction is added, so
            answers scored against evaluation/golden_dataset's reference
            answers aren't penalized for bracket markers those references
            never contained.

        citation_labels : list[str], optional
            The exact text to put inside each excerpt's bracket marker
            (e.g. "p.7"), same length/order as `contexts`. Without this,
            the bracket is just the excerpt's position (`[1]`, `[2]`, ...)
            - a caller has to look the number up against a separate source
            list to know what it refers to. With it, the citation is
            self-describing and traceable straight to the source page
            without that indirection (e.g. `[p.7]` directly names the PDF
            page, not an arbitrary retrieval-order index). Only meaningful
            together with `sources`.
        """
        if sources is not None and len(sources) != len(contexts):
            raise ValueError(
                "sources, when provided, must be the same length as contexts."
            )
        if citation_labels is not None and len(citation_labels) != len(contexts):
            raise ValueError(
                "citation_labels, when provided, must be the same length as contexts."
            )

        def label(i: int, context: str) -> str:
            marker = citation_labels[i - 1] if citation_labels else str(i)
            tag = f" ({sources[i - 1]})" if sources else ""
            return f"[{marker}]{tag} {context}"

        context_block = "\n\n".join(
            label(i, context) for i, context in enumerate(contexts, start=1)
        ) or "(no context retrieved)"

        system_prompt = GENERATION_SYSTEM_PROMPT
        if sources is not None:
            system_prompt += GENERATION_CITATION_INSTRUCTION

        return [
            ("system", system_prompt),
            ("human", f"Context:\n{context_block}\n\nQuestion: {query}"),
        ]

    def generate(
        self,
        query: str,
        contexts: List[str],
        sources: Optional[List[str]] = None,
        citation_labels: Optional[List[str]] = None,
    ) -> GenerationResult:
        messages = self.build_messages(query, contexts, sources=sources, citation_labels=citation_labels)

        start = time.perf_counter()
        response = self.model.invoke(messages)
        latency_ms = (time.perf_counter() - start) * 1000

        answer = response.content if hasattr(response, "content") else str(response)

        return GenerationResult(answer=answer, latency_ms=latency_ms)

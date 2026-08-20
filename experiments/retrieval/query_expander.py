import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from config import RAGAS_JUDGE_MODEL


class QueryExpander:
    """
    Generates alternate phrasings of a query using an LLM.

    Golden-dataset queries are often short and colloquial (e.g.
    "What is Publication 15-T used for?") while the source document
    uses formal/legal phrasing. When the literal query wording embeds
    poorly against that phrasing, retrieval misses chunks that a
    differently-worded version of the same question would have
    found. QueryExpansionRetriever runs retrieval for each variant
    and merges the results.
    """

    def __init__(self, model: str = None, num_variants: int = 2):

        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to a .env file at "
                "the project root before using QueryExpander."
            )

        model = model or RAGAS_JUDGE_MODEL

        self.llm = ChatOpenAI(model=model, temperature=0.3)
        self.num_variants = num_variants

    def expand(self, query: str):
        """
        Returns up to num_variants alternate phrasings of query
        (does NOT include the original query itself).
        """

        prompt = (
            f"Rewrite the following question into {self.num_variants} "
            "alternative phrasings that preserve its exact meaning "
            "but use different wording (e.g. more formal/legal "
            "phrasing, synonyms, or a differently structured "
            "question). Return ONLY the alternative questions, one "
            "per line, with no numbering and no extra commentary.\n\n"
            f"Question: {query}"
        )

        response = self.llm.invoke(prompt)

        lines = [
            line.strip(" -•\t")
            for line in response.content.strip().split("\n")
            if line.strip()
        ]

        return lines[: self.num_variants]

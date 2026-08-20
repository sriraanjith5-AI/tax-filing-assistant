from evaluation.evaluation_dataclass import EvaluationResult


def _normalize(text: str) -> str:
    """Lowercase and fold typographic punctuation (curly quotes/apostrophes,
    en/em dashes) to their plain-ASCII equivalents so keyword matching isn't
    thrown off by PDF-extraction artifacts that don't reflect a real
    retrieval miss."""
    replacements = {
        "’": "'", "‘": "'",
        "“": '"', "”": '"',
        "–": "-", "—": "-",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.lower()


class RAGEvaluator:

    # How many pages away from the golden `source_page` still counts as a
    # hit. PDF page numbers and chunk `page` metadata can be off-by-one
    # depending on cover/TOC pages, and a fact can legitimately span onto
    # the next page, so an exact-only match is too strict.
    PAGE_TOLERANCE = 1

    # Minimum fraction of `answer_must_contain` keywords that must show up
    # in the retrieved text for retrieval to count as accurate.
    KEYWORD_COVERAGE_THRESHOLD = 0.5

    def evaluate_retrieval(
        self,
        question_id,
        query,
        expected_answer,
        retrieved_results,
        source_page=None,
        required_keywords=None,
    ):

        required_keywords = required_keywords or []
        retrieved_count = len(retrieved_results)

        if retrieved_count == 0:
            return EvaluationResult(
                question_id=question_id,
                query=query,
                expected_answer=expected_answer,
                actual_answer="",
                retrieved_count=0,
                top_score=0.0,
                retrieval_success=False,
                answer_score=0.0,
                evaluation_status="RETRIEVAL_FAILED",
                page_hit=False,
                matched_keywords=[],
                missing_keywords=list(required_keywords),
                keyword_coverage=0.0,
                accuracy_status="RETRIEVAL_FAILED",
            )

        top_score = retrieved_results[0].score

        # --------------------------------------------------------------
        # Page-hit: does any retrieved chunk come from (near) the page
        # the golden answer was sourced from?
        # --------------------------------------------------------------
        page_hit = False
        if source_page is not None:
            for result in retrieved_results:
                page = result.document.metadata.get("page")
                if page is None:
                    continue
                if abs(int(page) - int(source_page)) <= self.PAGE_TOLERANCE:
                    page_hit = True
                    break

        # --------------------------------------------------------------
        # Keyword coverage: what fraction of the golden
        # `answer_must_contain` terms actually appear in the retrieved
        # text? This is what catches "retrieved something, but not the
        # right thing" cases the count-only check used to miss.
        # --------------------------------------------------------------
        combined_text = _normalize(
            " ".join(
                result.document.page_content for result in retrieved_results
            )
        )

        matched_keywords = []
        missing_keywords = []

        for keyword in required_keywords:
            if _normalize(keyword) in combined_text:
                matched_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)

        keyword_coverage = (
            len(matched_keywords) / len(required_keywords)
            if required_keywords
            else 1.0
        )

        # --------------------------------------------------------------
        # Overall accuracy verdict. `page_hit` is reported for visibility
        # but does NOT gate the verdict: this document's golden `source_page`
        # values (printed page numbers) drift against PyPDFLoader's 0-indexed
        # physical page by a variable 0-2 page offset, so it's too noisy a
        # signal to fail a retrieval on by itself. Keyword coverage against
        # the retrieved text is the reliable signal.
        # --------------------------------------------------------------
        is_accurate = keyword_coverage >= self.KEYWORD_COVERAGE_THRESHOLD

        accuracy_status = "ACCURATE" if is_accurate else "INACCURATE"

        return EvaluationResult(
            question_id=question_id,
            query=query,
            expected_answer=expected_answer,
            actual_answer="",
            retrieved_count=retrieved_count,
            top_score=top_score,
            retrieval_success=True,
            answer_score=keyword_coverage,
            evaluation_status="RETRIEVAL_SUCCESS",
            page_hit=page_hit,
            matched_keywords=matched_keywords,
            missing_keywords=missing_keywords,
            keyword_coverage=keyword_coverage,
            accuracy_status=accuracy_status,
        )

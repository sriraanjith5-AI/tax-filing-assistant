import re

from langchain_core.documents import Document

from cleaners.base_cleaner import BaseCleaner


class WhitespaceCleaner(BaseCleaner):

    def clean(self, document: Document) -> None:

        text = document.page_content

        text = re.sub(r"\s+", " ", text).strip()

        document.page_content = text
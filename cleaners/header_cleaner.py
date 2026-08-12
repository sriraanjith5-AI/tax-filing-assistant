import re

from langchain_core.documents import Document

from cleaners.base_cleaner import BaseCleaner


class HeaderCleaner(BaseCleaner):

    def clean(self, document: Document) -> None:

        text = document.page_content

        # Remove "Page X of Y"
        text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text)

        # Remove Publication 15-T, Publication 17 etc.
        text = re.sub(r"Publication\s+\d+[A-Z-]*", "", text)

        document.page_content = text
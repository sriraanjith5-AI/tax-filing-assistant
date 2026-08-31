from langchain_core.documents import Document

from cleaners.base_cleaner import BaseCleaner


class DummyCleaner(BaseCleaner):

    def clean(self, document: Document) -> None:

        document.page_content += "\n[CLEANED]"
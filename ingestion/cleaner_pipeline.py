from typing import List

from langchain_core.documents import Document

from cleaners.base_cleaner import BaseCleaner

import logging

logger = logging.getLogger(__name__)


class CleanerPipeline:

    def __init__(self, cleaners: List[BaseCleaner]):

        self.cleaners = cleaners

        self.failed_documents = []

    def clean(self, documents: List[Document]) -> List[Document]:

        if documents is None:
            raise ValueError("Documents cannot be None")

        if len(documents) == 0:
            logger.warning("No documents received.")
            return []

        logger.info(f"Received {len(documents)} documents.")

        cleaned_documents = []

        for index, document in enumerate(documents):

            try:

                logger.info(f"Processing document {index+1}")

                for cleaner in self.cleaners:

                    logger.info(
                        f"Executing {cleaner.__class__.__name__}"
                    )

                    cleaner.clean(document)

                cleaned_documents.append(document)

            except Exception as ex:

                logger.exception(ex)

                self.failed_documents.append(document)

        logger.info(
            f"Successfully cleaned {len(cleaned_documents)} documents."
        )

        logger.info(
            f"Failed documents : {len(self.failed_documents)}"
        )

        return cleaned_documents
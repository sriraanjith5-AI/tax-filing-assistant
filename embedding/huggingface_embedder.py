from embedding.base_embedder import BaseEmbedder
from typing import List
from langchain_core.documents import Document
from embedding.embedding_dataclass import EmbeddingResult,EmbeddingResponse
from sentence_transformers import SentenceTransformer

import logging

logger = logging.getLogger(__name__)

class HuggingFaceEmbedder(BaseEmbedder):
    def __init__(self,model_configuration,model=None):
        self.model_name=model_configuration.model_name
        self.batch_size=model_configuration.batch_size
        if model is None:
            self.model = SentenceTransformer(self.model_name)
        else:
            self.model = model        

    def embed(self,documents:List[Document]) -> EmbeddingResponse:
        if len(documents) == 0:
            logger.info("No documents to embed.")
            return EmbeddingResponse(
                embed_status="EMPTY_INPUT",
                total_no_documents=0,
                successful_documents=[],
                failed_documents=[]
            )

        start = 0
        end = self.batch_size
        successful_results=[]
        failed_results=[]
        embed_status = ""

        while (start < len(documents)):
            batch_documents = documents[start:end]
            texts = [doc.page_content for doc in batch_documents]

            try:
                vectors = self.model.encode(texts)
                if len(vectors) != len(batch_documents):
                    raise ValueError(
                        f"Mismatch between number of vectors ({len(vectors)}) "
                        f"and documents ({len(batch_documents)})."
                    )
            except Exception as exc:
                # The batch call is a single vectorized operation, so a
                # failure here doesn't tell us which individual document(s)
                # caused it. Fall back to embedding one document at a time
                # so we can attribute success/failure per document instead
                # of failing the whole batch.
                logger.error(f"Batch embedding failed for range {start} to {end}: {exc}")
                logger.info("Falling back to per-document embedding for this batch.")

                for document in batch_documents:
                    try:
                        vector = self.model.encode([document.page_content])[0]
                        if vector is None or len(vector) == 0:
                            logger.error(f"Failed to embed document during individual embedding {document.metadata.get('id', '<no id>')} and vector is empty.")
                            failed_results.append(document)
                        else:
                            successful_results.append(EmbeddingResult(document=document, vector=vector))
                    except Exception as doc_exc:
                        logger.error(f"Failed to embed document {document.metadata.get('id', '<no id>')}: {doc_exc}")
                        failed_results.append(document)
            else:
                for document,vector in zip(batch_documents,vectors):
                    if vector is None or len(vector) == 0:
                        logger.error(f"Failed to embed document during batch document embedding {document.metadata.get('id', '<no id>')} and vector is empty or None.")
                        failed_results.append(document)
                    else:
                        result = EmbeddingResult(
                            document=document,
                            vector=vector
                        )
                        successful_results.append(result)
            start += self.batch_size
            end += self.batch_size
        logger.info(f"Total documents embedded: {len(successful_results)}, failed to embed: {len(failed_results)}")
        if len(failed_results) == 0 and len(successful_results) > 0:
            embed_status="SUCCESS"
        elif len(successful_results) == 0 and len(failed_results) > 0:
            embed_status="FAILED"
        else:
            embed_status="PARTIAL_SUCCESS"
        return EmbeddingResponse(
                        embed_status=embed_status,
                        total_no_documents=len(documents),
                        successful_documents=successful_results,
                        failed_documents=failed_results
                    )


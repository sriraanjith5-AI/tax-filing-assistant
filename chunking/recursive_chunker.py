from chunking.base_chunker import BaseChunker
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_MIN_SIZE,CHUNK_SIZE,CHUNK_OVERLAP
from utils.logger import logging
import hashlib

logger = logging.getLogger(__name__)

class RecursiveChunker(BaseChunker):
    def generate_chunk_id(self, content: str) -> str:
        normalized_content=content.strip()
        return hashlib.sha256(
            normalized_content.encode("utf-8")
        ).hexdigest()

    def chunk(self,documents:List[Document]) -> List[Document]:
        if len(documents) == 0:
            logger.info("Length of received Document is Empty")
            return []
        chunks=[]

        splitter=RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
        )
        
        for doc in documents:
            if len(doc.page_content) == 0:
                continue
            else:
                chunks_intermediate=splitter.split_documents([doc])
                if len(chunks_intermediate) == 0:
                    continue
                else:
                    if len(chunks_intermediate) > 1:
                    #Optimize the chunks by merging small chunks with the previous one
                        if len(chunks_intermediate[-1].page_content) < CHUNK_MIN_SIZE:
                            chunks_intermediate[-2].page_content = chunks_intermediate[-2].page_content + chunks_intermediate[-1].page_content
                            chunks_intermediate.pop()
                #Add metadata
                for index,chunk in enumerate(chunks_intermediate):
                    chunk.metadata['source'] = doc.metadata['source']
                    chunk.metadata['chunk_number'] = index + 1
                    chunk.metadata['total_chunks'] = len(chunks_intermediate)
                    chunk.metadata['chunk_id'] = self.generate_chunk_id(chunk.page_content)
                    chunks.append(chunk)
                    #Log the summary
        return chunks
            


        



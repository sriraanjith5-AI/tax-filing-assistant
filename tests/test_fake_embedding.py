import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from embedding.embedding_dataclass import EmbeddingConfig
from embedding.huggingface_embedder import HuggingFaceEmbedder


class FakeEmbeddingModel:

    def encode(self, texts):
        raise RuntimeError("Simulated embedding failure")
        """ 
        if len(texts) > 1:
            raise RuntimeError("Simulated batch embedding failure")
        if "Employers must withhold" in texts[0]:
            raise RuntimeError("Simulated individual document failure")

        return [[0.1, 0.2, 0.3]]
 """
documents = [
    Document(
        page_content="The standard deduction is available to eligible taxpayers.",
        metadata={"id": "doc-1"}
    ),
    Document(
        page_content="Employers must withhold federal income tax from employee wages.",
        metadata={"id": "doc-2"}
    ),
    Document(
        page_content="Form W-4 is used by employees to provide withholding information.",
        metadata={"id": "doc-3"}
    )
]

embedding_config = EmbeddingConfig()

fake_model = FakeEmbeddingModel()
embedder = HuggingFaceEmbedder(embedding_config,model=fake_model)

response = embedder.embed(documents)

print(f"Status: {response.embed_status}")
print(f"Total: {response.total_no_documents}")
print(f"Successful: {len(response.successful_documents)}")
print(f"Failed: {len(response.failed_documents)}")

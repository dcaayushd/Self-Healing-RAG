import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from self_healing_rag.config import Settings
from self_healing_rag.vector_store import VectorStoreManager


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        return [float(len(text)), float(text.count("rag")), 1.0]


def test_collection_embedding_model_mismatch_fails(tmp_path):
    settings = Settings(chroma_path=tmp_path / "chroma", checkpoint_db=tmp_path / "checkpoints.sqlite")
    manager = VectorStoreManager(settings, embedding=FakeEmbeddings())
    doc = Document(
        page_content="rag content",
        metadata={
            "chunk_id": "chunk-1",
            "source": "unit",
            "source_type": "text",
            "page": -1,
            "chunk_index": 0,
            "content_hash": "abc",
            "embedding_model": settings.embedding_model,
            "citation_label": "unit",
        },
    )
    manager.add_documents([doc], collection="default")

    other_settings = Settings(
        chroma_path=tmp_path / "chroma",
        checkpoint_db=tmp_path / "checkpoints.sqlite",
        embedding_model="different-model",
    )
    other_manager = VectorStoreManager(other_settings, embedding=FakeEmbeddings())

    with pytest.raises(ValueError, match="different-model"):
        other_manager.search("rag", collection="default", top_k=1, fetch_k=1)


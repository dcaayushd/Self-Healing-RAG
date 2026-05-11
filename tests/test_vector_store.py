import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from self_healing_rag.config import Settings
from self_healing_rag.vector_store import VectorStoreManager, _select_diverse_results, _select_ranked_results


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


def test_overview_returns_representative_chunks_and_stats(tmp_path):
    settings = Settings(chroma_path=tmp_path / "chroma", checkpoint_db=tmp_path / "checkpoints.sqlite")
    manager = VectorStoreManager(settings, embedding=FakeEmbeddings())
    docs = [
        Document(
            page_content="first source introduction",
            metadata={
                "chunk_id": "a-0",
                "source": "a.md",
                "source_type": "markdown",
                "page": -1,
                "chunk_index": 0,
                "content_hash": "a0",
                "embedding_model": settings.embedding_model,
                "citation_label": "a.md",
            },
        ),
        Document(
            page_content="second source introduction",
            metadata={
                "chunk_id": "b-0",
                "source": "b.md",
                "source_type": "markdown",
                "page": -1,
                "chunk_index": 0,
                "content_hash": "b0",
                "embedding_model": settings.embedding_model,
                "citation_label": "b.md",
            },
        ),
    ]
    manager.add_documents(docs, collection="default")

    overview = manager.overview(collection="default", limit=2)
    stats = manager.collection_stats("default")

    assert [chunk.citation_id for chunk in overview] == ["C1", "C2"]
    assert {chunk.source for chunk in overview} == {"a.md", "b.md"}
    assert stats.chunk_count == 2
    assert stats.source_count == 2


def test_search_result_selection_diversifies_sources():
    results = [
        (
            Document(page_content=f"same source {idx}", metadata={"chunk_id": f"a-{idx}", "source": "a.md"}),
            float(idx),
        )
        for idx in range(5)
    ] + [
        (Document(page_content="second source", metadata={"chunk_id": "b-0", "source": "b.md"}), 5.0),
        (Document(page_content="third source", metadata={"chunk_id": "c-0", "source": "c.md"}), 6.0),
    ]

    selected = _select_diverse_results(results, top_k=4)

    assert len(selected) == 4
    assert {doc.metadata["source"] for doc, _ in selected} >= {"a.md", "b.md"}


def test_hybrid_ranker_promotes_lexically_relevant_result():
    results = [
        (Document(page_content="generic background material", metadata={"chunk_id": "a", "source": "a.md"}), 0.1),
        (
            Document(
                page_content="hantavirus transmission happens through infected rodent excreta",
                metadata={"chunk_id": "b", "source": "b.md"},
            ),
            0.8,
        ),
    ]

    selected = _select_ranked_results("hantavirus transmission", results, top_k=1)

    assert selected[0].document.metadata["chunk_id"] == "b"
    assert selected[0].relevance > 0


def test_reingesting_same_chunk_id_replaces_existing_document(tmp_path):
    settings = Settings(chroma_path=tmp_path / "chroma", checkpoint_db=tmp_path / "checkpoints.sqlite")
    manager = VectorStoreManager(settings, embedding=FakeEmbeddings())
    metadata = {
        "chunk_id": "same-id",
        "source": "unit",
        "source_type": "text",
        "page": -1,
        "chunk_index": 0,
        "content_hash": "abc",
        "embedding_model": settings.embedding_model,
        "citation_label": "unit",
    }

    manager.add_documents([Document(page_content="old rag content", metadata=metadata)], collection="default")
    manager.add_documents([Document(page_content="new rag content", metadata=metadata)], collection="default")

    stats = manager.collection_stats("default")
    chunks = manager.search("new", collection="default", top_k=3, fetch_k=3)

    assert stats.chunk_count == 1
    assert chunks[0].content == "new rag content"
    assert chunks[0].relevance is not None

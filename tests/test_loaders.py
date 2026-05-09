import socket

import pytest

from self_healing_rag.config import Settings
from self_healing_rag.loaders import load_and_chunk_sources
from self_healing_rag.security import UnsafeUrlError, validate_public_url


def test_text_loader_adds_deterministic_chunk_metadata(tmp_path):
    doc = tmp_path / "note.md"
    doc.write_text("Self-healing RAG retrieves evidence and critiques answers.\n" * 5)
    settings = Settings(chroma_path=tmp_path / "chroma", checkpoint_db=tmp_path / "checkpoints.sqlite")

    first = load_and_chunk_sources([str(doc)], settings, collection="default")
    second = load_and_chunk_sources([str(doc)], settings, collection="default")

    assert first
    assert [chunk.metadata["chunk_id"] for chunk in first] == [chunk.metadata["chunk_id"] for chunk in second]
    metadata = first[0].metadata
    assert metadata["source_type"] == "markdown"
    assert metadata["embedding_model"] == "nomic-embed-text"
    assert metadata["citation_label"] == "note.md"


def test_private_url_is_blocked(monkeypatch):
    def fake_getaddrinfo(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(UnsafeUrlError):
        validate_public_url("https://example.test/page")


def test_localhost_url_is_blocked_without_dns():
    with pytest.raises(UnsafeUrlError):
        validate_public_url("http://localhost:8000")


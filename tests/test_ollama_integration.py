import os

import pytest

from self_healing_rag.config import Settings
from self_healing_rag.service import RagService


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_TESTS") != "1",
    reason="Set RUN_OLLAMA_TESTS=1 with Ollama running and default models pulled.",
)


def test_local_ollama_pipeline_executes_end_to_end(tmp_path):
    source = tmp_path / "doc.md"
    source.write_text(
        "The self-healing RAG pipeline retries with a reformulated query "
        "when the critic rejects an ungrounded answer."
    )
    settings = Settings(
        chroma_path=tmp_path / "chroma",
        checkpoint_db=tmp_path / "checkpoints.sqlite",
        chunk_size=500,
        chunk_overlap=50,
    )
    service = RagService(settings)

    ingest = service.ingest_sources([str(source)], collection="default")
    response = service.ask("What happens when the critic rejects an answer?", collection="default", max_attempts=2)

    assert ingest.chunks_added > 0
    assert response.attempts
    assert response.status in {"answered", "insufficient_info"}


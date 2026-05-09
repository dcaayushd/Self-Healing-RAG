from fastapi.testclient import TestClient

from self_healing_rag.api import app, get_rag_service
from self_healing_rag.constants import FALLBACK_ANSWER
from self_healing_rag.schemas import AskResponse, HealthResponse, IngestResponse


class FakeService:
    def health(self):
        return HealthResponse(
            ok=True,
            chroma_path="tmp/chroma",
            checkpoint_db="tmp/checkpoints.sqlite",
            default_collection="default",
            chat_model="llama3.1:8b",
            embedding_model="nomic-embed-text",
        )

    def ingest_sources(self, sources, *, collection=None):
        return IngestResponse(collection=collection or "default", chunks_added=1, sources=sources, ids=["1"])

    def ingest_upload(self, filename, content, *, collection=None):
        return IngestResponse(collection=collection or "default", chunks_added=1, sources=[f"upload:{filename}"], ids=["1"])

    def ask(self, question, *, collection=None, max_attempts=None, thread_id=None):
        return AskResponse(status="insufficient_info", answer=FALLBACK_ANSWER, citations=[], attempts=[], thread_id=thread_id or "t1")


def client():
    app.dependency_overrides[get_rag_service] = lambda: FakeService()
    return TestClient(app)


def test_health_endpoint():
    response = client().get("/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_ingest_endpoint():
    response = client().post("/ingest", json={"sources": ["./data/docs"], "collection": "docs"})
    assert response.status_code == 200
    assert response.json()["chunks_added"] == 1


def test_upload_endpoint():
    response = client().post(
        "/ingest/upload",
        data={"collection": "docs"},
        files={"file": ("note.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["sources"] == ["upload:note.txt"]


def test_ask_endpoint():
    response = client().post("/ask", json={"question": "What?", "collection": "docs", "thread_id": "thread"})
    assert response.status_code == 200
    assert response.json()["thread_id"] == "thread"


from collections import OrderedDict

from self_healing_rag.config import Settings
from self_healing_rag.schemas import AskResponse, IndexStats
from self_healing_rag.service import RagService


class FakeEngine:
    def __init__(self) -> None:
        self.calls = 0

    def ask(self, question, *, collection=None, max_attempts=None, thread_id=None, focus_sources=None):
        self.calls += 1
        return AskResponse(
            status="answered",
            answer=f"answer {self.calls}",
            citations=[],
            attempts=[],
            thread_id=thread_id or f"thread-{self.calls}",
            focus_sources=focus_sources or [],
            total_ms=10.0,
        )


class FakeVectorStore:
    def __init__(self) -> None:
        self.deleted_collections: list[str] = []

    def collection_stats(self, collection):
        return IndexStats(
            collection=collection,
            chunk_count=3,
            source_count=1,
            sources=["doc.md"],
            source_types=["markdown"],
            embedding_model="fake-embed",
            is_empty=False,
        )

    def delete_collection(self, collection):
        self.deleted_collections.append(collection)

    def delete_sources(self, collection, sources):
        return len(sources)


def make_service(tmp_path, *, answer_cache_size=64):
    service = RagService.__new__(RagService)
    service.settings = Settings(
        chroma_path=tmp_path / "chroma",
        checkpoint_db=tmp_path / "checkpoints.sqlite",
        answer_cache_size=answer_cache_size,
    )
    service.vector_store = FakeVectorStore()
    service.engine = FakeEngine()
    service._answer_cache = OrderedDict()
    return service


def test_service_answer_cache_reuses_same_thread_answer_and_returns_copies(tmp_path):
    service = make_service(tmp_path)

    first = service.ask(
        " What is this document about? ",
        collection="docs",
        max_attempts=3,
        thread_id="thread-1",
        focus_sources=["doc.md"],
    )
    first.answer = "caller mutation"
    second = service.ask(
        "what is this document about?",
        collection="docs",
        max_attempts=3,
        thread_id="thread-1",
        focus_sources=["doc.md"],
    )

    assert service.engine.calls == 1
    assert second.answer == "answer 1"
    assert second.thread_id == "thread-1"


def test_service_answer_cache_is_not_used_without_thread_id(tmp_path):
    service = make_service(tmp_path)

    service.ask("What is this document about?", collection="docs")
    service.ask("What is this document about?", collection="docs")

    assert service.engine.calls == 2


def test_service_answer_cache_invalidates_when_collection_is_reset(tmp_path):
    service = make_service(tmp_path)

    service.ask("What is this document about?", collection="docs", thread_id="thread-1")
    service.reset_collection("docs")
    service.ask("What is this document about?", collection="docs", thread_id="thread-1")

    assert service.engine.calls == 2
    assert service.vector_store.deleted_collections == ["docs"]

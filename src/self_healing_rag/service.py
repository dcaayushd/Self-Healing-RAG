from __future__ import annotations

from functools import lru_cache

from self_healing_rag.config import Settings, get_settings
from self_healing_rag.graph import OllamaRagComponents, SelfHealingRag
from self_healing_rag.loaders import load_and_chunk_sources, load_and_chunk_upload
from self_healing_rag.schemas import AskResponse, HealthResponse, IngestResponse
from self_healing_rag.vector_store import VectorStoreManager


class RagService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.vector_store = VectorStoreManager(self.settings)
        self.engine = SelfHealingRag(
            self.settings,
            components=OllamaRagComponents(self.settings, vector_store=self.vector_store),
            use_checkpointer=True,
        )

    def ingest_sources(self, sources: list[str], *, collection: str | None = None) -> IngestResponse:
        target_collection = collection or self.settings.default_collection
        docs = load_and_chunk_sources(sources, self.settings, collection=target_collection)
        ids = self.vector_store.add_documents(docs, collection=target_collection)
        return IngestResponse(collection=target_collection, chunks_added=len(ids), sources=sources, ids=ids)

    def ingest_upload(self, filename: str, content: bytes, *, collection: str | None = None) -> IngestResponse:
        target_collection = collection or self.settings.default_collection
        docs = load_and_chunk_upload(filename, content, self.settings, collection=target_collection)
        ids = self.vector_store.add_documents(docs, collection=target_collection)
        return IngestResponse(collection=target_collection, chunks_added=len(ids), sources=[f"upload:{filename}"], ids=ids)

    def ask(
        self,
        question: str,
        *,
        collection: str | None = None,
        max_attempts: int | None = None,
        thread_id: str | None = None,
    ) -> AskResponse:
        return self.engine.ask(question, collection=collection, max_attempts=max_attempts, thread_id=thread_id)

    def reset_collection(self, collection: str | None = None) -> None:
        self.vector_store.delete_collection(collection or self.settings.default_collection)

    def health(self) -> HealthResponse:
        return HealthResponse(
            ok=True,
            chroma_path=str(self.settings.chroma_path),
            checkpoint_db=str(self.settings.checkpoint_db),
            default_collection=self.settings.default_collection,
            chat_model=self.settings.chat_model,
            embedding_model=self.settings.embedding_model,
        )


@lru_cache
def build_service() -> RagService:
    return RagService()

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import re

from self_healing_rag.config import Settings, get_settings
from self_healing_rag.graph import OllamaRagComponents, SelfHealingRag
from self_healing_rag.loaders import load_and_chunk_sources, load_and_chunk_upload
from self_healing_rag.schemas import AskResponse, HealthResponse, IndexStats, IngestResponse
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
        self._answer_cache: OrderedDict[tuple, AskResponse] = OrderedDict()

    def ingest_sources(self, sources: list[str], *, collection: str | None = None) -> IngestResponse:
        target_collection = collection or self.settings.default_collection
        docs = load_and_chunk_sources(sources, self.settings, collection=target_collection)
        ids = self.vector_store.add_documents(docs, collection=target_collection)
        self._clear_answer_cache(target_collection)
        actual_sources = sorted({str(doc.metadata.get("source", "unknown")) for doc in docs})
        return IngestResponse(collection=target_collection, chunks_added=len(ids), sources=actual_sources, ids=ids)

    def ingest_upload(self, filename: str, content: bytes, *, collection: str | None = None) -> IngestResponse:
        target_collection = collection or self.settings.default_collection
        docs = load_and_chunk_upload(filename, content, self.settings, collection=target_collection)
        ids = self.vector_store.add_documents(docs, collection=target_collection)
        self._clear_answer_cache(target_collection)
        return IngestResponse(collection=target_collection, chunks_added=len(ids), sources=[f"upload:{filename}"], ids=ids)

    def ask(
        self,
        question: str,
        *,
        collection: str | None = None,
        max_attempts: int | None = None,
        thread_id: str | None = None,
        focus_sources: list[str] | None = None,
    ) -> AskResponse:
        target_collection = collection or self.settings.default_collection
        target_attempts = max_attempts or self.settings.max_attempts
        cache_key = self._answer_cache_key(
            question,
            collection=target_collection,
            max_attempts=target_attempts,
            thread_id=thread_id,
            focus_sources=focus_sources,
        )
        cached = self._cached_answer(cache_key)
        if cached is not None:
            return cached

        response = self.engine.ask(
            question,
            collection=target_collection,
            max_attempts=target_attempts,
            thread_id=thread_id,
            focus_sources=focus_sources,
        )
        self._store_answer_cache(cache_key, response)
        return response

    def reset_collection(self, collection: str | None = None) -> None:
        target_collection = collection or self.settings.default_collection
        self.vector_store.delete_collection(target_collection)
        self._clear_answer_cache(target_collection)

    def delete_sources(self, sources: list[str], *, collection: str | None = None) -> int:
        target_collection = collection or self.settings.default_collection
        deleted = self.vector_store.delete_sources(target_collection, sources)
        if deleted:
            self._clear_answer_cache(target_collection)
        return deleted

    def collection_stats(self, collection: str | None = None) -> IndexStats:
        return self.vector_store.collection_stats(collection or self.settings.default_collection)

    def health(self) -> HealthResponse:
        return HealthResponse(
            ok=True,
            chroma_path=str(self.settings.chroma_path),
            checkpoint_db=str(self.settings.checkpoint_db),
            default_collection=self.settings.default_collection,
            chat_model=self.settings.chat_model,
            embedding_model=self.settings.embedding_model,
        )

    def _answer_cache_key(
        self,
        question: str,
        *,
        collection: str,
        max_attempts: int,
        thread_id: str | None,
        focus_sources: list[str] | None,
    ) -> tuple | None:
        if not self.settings.answer_cache_size or not thread_id:
            return None
        stats = self.vector_store.collection_stats(collection)
        normalized_question = re.sub(r"\s+", " ", question.strip().lower())
        normalized_sources = tuple(sorted(source for source in focus_sources or [] if source))
        return (
            normalized_question,
            collection,
            max_attempts,
            thread_id,
            normalized_sources,
            stats.chunk_count,
            stats.source_count,
            stats.embedding_model,
            self.settings.critic_mode,
            self.settings.use_llm_rewrite,
        )

    def _cached_answer(self, cache_key: tuple | None) -> AskResponse | None:
        if cache_key is None:
            return None
        cached = self._answer_cache.get(cache_key)
        if cached is None:
            return None
        self._answer_cache.move_to_end(cache_key)
        return cached.model_copy(deep=True)

    def _store_answer_cache(self, cache_key: tuple | None, response: AskResponse) -> None:
        if cache_key is None or not self.settings.answer_cache_size:
            return
        self._answer_cache[cache_key] = response.model_copy(deep=True)
        self._answer_cache.move_to_end(cache_key)
        while len(self._answer_cache) > self.settings.answer_cache_size:
            self._answer_cache.popitem(last=False)

    def _clear_answer_cache(self, collection: str | None = None) -> None:
        if collection is None:
            self._answer_cache.clear()
            return
        for key in list(self._answer_cache):
            if len(key) > 1 and key[1] == collection:
                self._answer_cache.pop(key, None)


@lru_cache
def build_service() -> RagService:
    return RagService()

from __future__ import annotations

from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from self_healing_rag.config import Settings
from self_healing_rag.schemas import RetrievedChunk


class VectorStoreManager:
    def __init__(self, settings: Settings, embedding: Embeddings | None = None) -> None:
        self.settings = settings
        self.settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
        self.embedding = embedding or OllamaEmbeddings(
            model=self.settings.embedding_model,
            base_url=self.settings.ollama_base_url,
        )

    def add_documents(self, documents: list[Document], *, collection: str) -> list[str]:
        if not documents:
            return []
        self._ensure_collection(collection)
        ids = [str(doc.metadata["chunk_id"]) for doc in documents]
        vector_store = self._vector_store(collection)
        vector_store.add_documents(documents, ids=ids)
        return ids

    def search(self, query: str, *, collection: str, top_k: int, fetch_k: int) -> list[RetrievedChunk]:
        self._ensure_collection(collection)
        vector_store = self._vector_store(collection)
        limit = max(top_k, fetch_k)
        results = vector_store.similarity_search_with_score(query, k=limit)
        chunks: list[RetrievedChunk] = []
        for idx, (doc, score) in enumerate(results[:top_k], start=1):
            metadata = doc.metadata
            page = int(metadata.get("page", -1))
            chunks.append(
                RetrievedChunk(
                    id=str(metadata.get("chunk_id", "")),
                    content=doc.page_content,
                    source=str(metadata.get("source", "unknown")),
                    source_type=str(metadata.get("source_type", "text")),
                    citation_label=str(metadata.get("citation_label", metadata.get("source", "unknown"))),
                    page=page if page >= 0 else None,
                    chunk_index=int(metadata.get("chunk_index", idx - 1)),
                    score=float(score),
                    citation_id=f"C{idx}",
                )
            )
        return chunks

    def delete_collection(self, collection: str) -> None:
        try:
            self.client.delete_collection(collection)
        except Exception as exc:
            if "does not exist" not in str(exc).lower():
                raise

    def _vector_store(self, collection: str) -> Chroma:
        return Chroma(
            client=self.client,
            collection_name=collection,
            embedding_function=self.embedding,
            collection_metadata={"embedding_model": self.settings.embedding_model},
        )

    def _ensure_collection(self, collection: str) -> None:
        metadata = {"embedding_model": self.settings.embedding_model}
        chroma_collection = self.client.get_or_create_collection(collection, metadata=metadata)
        existing_model = (chroma_collection.metadata or {}).get("embedding_model")
        if existing_model and existing_model != self.settings.embedding_model:
            raise ValueError(
                f"Collection '{collection}' was built with embedding model '{existing_model}', "
                f"but current model is '{self.settings.embedding_model}'."
            )
        if not existing_model:
            chroma_collection.modify(metadata=metadata)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


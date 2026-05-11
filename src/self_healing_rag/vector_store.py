from __future__ import annotations

from collections import defaultdict
from math import ceil
from pathlib import Path
import re

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from self_healing_rag.config import Settings
from self_healing_rag.schemas import IndexStats, RetrievedChunk
from self_healing_rag.source_policy import is_system_sample_source, preferred_default_sources, source_filter_where


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
        self._delete_existing_ids(collection, ids)
        vector_store = self._vector_store(collection)
        vector_store.add_documents(documents, ids=ids)
        return ids

    def search(
        self,
        query: str,
        *,
        collection: str,
        top_k: int,
        fetch_k: int,
        focus_sources: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        self._ensure_collection(collection)
        vector_store = self._vector_store(collection)
        limit = max(top_k, fetch_k)
        where = source_filter_where(focus_sources)
        kwargs = {"filter": where} if where else {}
        results = vector_store.similarity_search_with_score(query, k=limit, **kwargs)
        chunks: list[RetrievedChunk] = []
        for idx, ranked in enumerate(_select_ranked_results(query, results, top_k=top_k), start=1):
            chunks.append(
                _chunk_from_document(
                    ranked.document,
                    citation_id=f"C{idx}",
                    score=float(ranked.distance),
                    relevance=ranked.relevance,
                    retrieval_rank=idx,
                    fallback_index=idx - 1,
                )
            )
        return chunks

    def overview(self, *, collection: str, limit: int, focus_sources: list[str] | None = None) -> list[RetrievedChunk]:
        self._ensure_collection(collection)
        chroma_collection = self.client.get_collection(collection)
        count = chroma_collection.count()
        if count == 0:
            return []

        fetch_limit = min(max(limit * 8, limit), count)
        where = source_filter_where(focus_sources or preferred_default_sources(self.collection_stats(collection).sources))
        kwargs = {"where": where} if where else {}
        result = chroma_collection.get(include=["documents", "metadatas"], limit=fetch_limit, **kwargs)
        rows = []
        for fallback_index, (doc_id, content, metadata) in enumerate(
            zip(result.get("ids", []), result.get("documents", []), result.get("metadatas", []), strict=False)
        ):
            metadata = dict(metadata or {})
            metadata.setdefault("chunk_id", doc_id)
            rows.append((content or "", metadata, fallback_index))

        rows.sort(key=lambda row: _overview_sort_key(row[1], row[2]))
        selected = _round_robin_sources(rows, limit=limit)
        return [
            _chunk_from_document(
                Document(page_content=content, metadata=metadata),
                citation_id=f"C{idx}",
                score=None,
                relevance=1.0,
                retrieval_rank=idx,
                fallback_index=fallback_index,
            )
            for idx, (content, metadata, fallback_index) in enumerate(selected, start=1)
        ]

    def collection_stats(self, collection: str) -> IndexStats:
        try:
            chroma_collection = self.client.get_collection(collection)
        except Exception:
            return IndexStats(collection=collection, chunk_count=0, source_count=0, embedding_model=self.settings.embedding_model)

        count = chroma_collection.count()
        metadata = chroma_collection.metadata or {}
        if count == 0:
            return IndexStats(
                collection=collection,
                chunk_count=0,
                source_count=0,
                embedding_model=str(metadata.get("embedding_model", self.settings.embedding_model)),
            )

        result = chroma_collection.get(include=["metadatas"], limit=min(count, 5000))
        metadatas = [dict(item or {}) for item in result.get("metadatas", [])]
        sources = sorted({str(item.get("source", "unknown")) for item in metadatas})
        source_types = sorted({str(item.get("source_type", "unknown")) for item in metadatas})
        return IndexStats(
            collection=collection,
            chunk_count=count,
            source_count=len(sources),
            sources=sources[:200],
            source_types=source_types,
            embedding_model=str(metadata.get("embedding_model", self.settings.embedding_model)),
            is_empty=False,
        )

    def delete_sources(self, collection: str, sources: list[str]) -> int:
        if not sources:
            return 0
        try:
            chroma_collection = self.client.get_collection(collection)
        except Exception:
            return 0
        result = chroma_collection.get(where=source_filter_where(sources), include=[])
        ids = result.get("ids", [])
        if ids:
            chroma_collection.delete(ids=ids)
        return len(ids)

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

    def _delete_existing_ids(self, collection: str, ids: list[str]) -> None:
        if not ids:
            return
        chroma_collection = self.client.get_collection(collection)
        existing = chroma_collection.get(ids=ids, include=[]).get("ids", [])
        if existing:
            chroma_collection.delete(ids=existing)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _chunk_from_document(
    doc: Document,
    *,
    citation_id: str,
    score: float | None,
    relevance: float | None = None,
    retrieval_rank: int | None = None,
    fallback_index: int,
) -> RetrievedChunk:
    metadata = doc.metadata
    page = int(metadata.get("page", -1))
    return RetrievedChunk(
        id=str(metadata.get("chunk_id", "")),
        content=doc.page_content,
        source=str(metadata.get("source", "unknown")),
        source_type=str(metadata.get("source_type", "text")),
        citation_label=str(metadata.get("citation_label", metadata.get("source", "unknown"))),
        page=page if page >= 0 else None,
        chunk_index=int(metadata.get("chunk_index", fallback_index)),
        score=score,
        relevance=relevance,
        retrieval_rank=retrieval_rank,
        citation_id=citation_id,
    )


def _overview_sort_key(metadata: dict, fallback_index: int) -> tuple[str, str, int, int, int]:
    page = int(metadata.get("page", -1))
    page_sort = page if page >= 0 else 10**9
    return (
        "1" if is_system_sample_source(str(metadata.get("source", ""))) else "0",
        str(metadata.get("source", "")),
        page_sort,
        int(metadata.get("chunk_index", fallback_index)),
        fallback_index,
    )


def _round_robin_sources(rows: list[tuple[str, dict, int]], *, limit: int) -> list[tuple[str, dict, int]]:
    by_source: dict[str, list[tuple[str, dict, int]]] = {}
    for row in rows:
        by_source.setdefault(str(row[1].get("source", "unknown")), []).append(row)

    selected: list[tuple[str, dict, int]] = []
    while len(selected) < limit and any(by_source.values()):
        for source in sorted(by_source):
            if by_source[source]:
                selected.append(by_source[source].pop(0))
                if len(selected) == limit:
                    return selected
    return selected


def _select_diverse_results(results: list[tuple[Document, float]], *, top_k: int) -> list[tuple[Document, float]]:
    unique: list[tuple[Document, float]] = []
    seen_ids: set[str] = set()
    for doc, score in results:
        chunk_id = str(doc.metadata.get("chunk_id") or id(doc))
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        unique.append((doc, score))

    if len(unique) <= top_k:
        return unique

    sources = {str(doc.metadata.get("source", "unknown")) for doc, _ in unique}
    if len(sources) <= 1:
        return unique[:top_k]

    max_per_source = max(2, ceil(top_k / min(len(sources), top_k)))
    selected: list[tuple[Document, float]] = []
    per_source: defaultdict[str, int] = defaultdict(int)
    deferred: list[tuple[Document, float]] = []

    for doc, score in unique:
        source = str(doc.metadata.get("source", "unknown"))
        if per_source[source] < max_per_source:
            selected.append((doc, score))
            per_source[source] += 1
        else:
            deferred.append((doc, score))
        if len(selected) == top_k:
            return selected

    for item in deferred:
        selected.append(item)
        if len(selected) == top_k:
            return selected
    return selected[:top_k]


class RankedResult:
    def __init__(self, document: Document, distance: float, relevance: float) -> None:
        self.document = document
        self.distance = distance
        self.relevance = relevance


def _select_ranked_results(query: str, results: list[tuple[Document, float]], *, top_k: int) -> list[RankedResult]:
    unique = _dedupe_results(results)
    if not unique:
        return []

    scored = [
        (
            doc,
            distance,
            _hybrid_relevance(query, doc, distance=distance, vector_rank=rank),
        )
        for rank, (doc, distance) in enumerate(unique, start=1)
    ]
    scored.sort(key=lambda item: item[2], reverse=True)
    selected = _mmr_select(scored, top_k=top_k)
    return [RankedResult(doc, distance, relevance) for doc, distance, relevance in selected]


def _dedupe_results(results: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
    unique: list[tuple[Document, float]] = []
    seen_ids: set[str] = set()
    for doc, score in results:
        chunk_id = str(doc.metadata.get("chunk_id") or id(doc))
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        unique.append((doc, score))
    return unique


def _hybrid_relevance(query: str, doc: Document, *, distance: float, vector_rank: int) -> float:
    vector_score = 1.0 / vector_rank
    semantic_score = 1.0 / (1.0 + max(float(distance), 0.0))
    lexical_score = _lexical_overlap(query, doc.page_content)
    title_score = _lexical_overlap(query, f"{doc.metadata.get('source', '')} {doc.metadata.get('citation_label', '')}")
    score = (0.32 * vector_score) + (0.18 * semantic_score) + (0.42 * lexical_score) + (0.08 * title_score)
    return max(0.0, min(1.0, score))


def _lexical_overlap(query: str, text: str) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    text_tokens = _tokens(text)
    if not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    return overlap / len(query_tokens)


def _mmr_select(
    scored: list[tuple[Document, float, float]],
    *,
    top_k: int,
    relevance_weight: float = 0.74,
) -> list[tuple[Document, float, float]]:
    selected: list[tuple[Document, float, float]] = []
    remaining = list(scored)
    per_source: defaultdict[str, int] = defaultdict(int)

    while remaining and len(selected) < top_k:
        best_idx = 0
        best_score = float("-inf")
        for idx, item in enumerate(remaining):
            doc, _, relevance = item
            source = str(doc.metadata.get("source", "unknown"))
            redundancy = _max_similarity(doc, [selected_item[0] for selected_item in selected])
            source_penalty = 0.08 * per_source[source]
            mmr_score = (relevance_weight * relevance) - ((1 - relevance_weight) * redundancy) - source_penalty
            if mmr_score > best_score:
                best_idx = idx
                best_score = mmr_score
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        per_source[str(chosen[0].metadata.get("source", "unknown"))] += 1
    return selected


def _max_similarity(doc: Document, selected_docs: list[Document]) -> float:
    if not selected_docs:
        return 0.0
    doc_tokens = _tokens(doc.page_content)
    if not doc_tokens:
        return 0.0
    return max(_jaccard(doc_tokens, _tokens(selected.page_content)) for selected in selected_docs)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]{3,}", text.lower()) if token not in _STOPWORDS}


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "about",
    "what",
    "which",
    "when",
    "where",
    "does",
    "into",
    "your",
    "you",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "can",
    "could",
    "would",
    "should",
}

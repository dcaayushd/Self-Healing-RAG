from __future__ import annotations

import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

os.environ.setdefault("USER_AGENT", "self-healing-rag/0.1")

from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from self_healing_rag.config import Settings
from self_healing_rag.security import is_url, validate_public_url


SUPPORTED_FILE_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".html", ".htm"}


def load_and_chunk_sources(sources: Iterable[str], settings: Settings, *, collection: str) -> list[Document]:
    documents: list[Document] = []
    for source in sources:
        documents.extend(_load_one_source(source, settings))
    return split_and_prepare_documents(documents, settings, collection=collection)


def load_and_chunk_upload(filename: str, content: bytes, settings: Settings, *, collection: str) -> list[Document]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_FILE_EXTENSIONS:
        raise ValueError(f"Unsupported upload extension: {suffix or '<none>'}")

    with NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(content)
        tmp.flush()
        docs = _load_local_file(Path(tmp.name), display_source=f"upload:{filename}")
    return split_and_prepare_documents(docs, settings, collection=collection)


def split_and_prepare_documents(documents: list[Document], settings: Settings, *, collection: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    split_docs = splitter.split_documents(documents)

    prepared: list[Document] = []
    counters: dict[str, int] = {}
    for doc in split_docs:
        source = str(doc.metadata.get("source", "unknown"))
        page = _normalize_page(doc.metadata.get("page"))
        key = f"{source}:{page}"
        chunk_index = counters.get(key, 0)
        counters[key] = chunk_index + 1

        content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
        chunk_id = _chunk_id(
            collection=collection,
            source=source,
            page=page,
            chunk_index=chunk_index,
            content_hash=content_hash,
            embedding_model=settings.embedding_model,
        )
        metadata = {
            "chunk_id": chunk_id,
            "source": source,
            "source_type": str(doc.metadata.get("source_type", "text")),
            "page": page if page is not None else -1,
            "chunk_index": chunk_index,
            "content_hash": content_hash,
            "embedding_model": settings.embedding_model,
            "citation_label": _citation_label(source, page),
            "start_index": int(doc.metadata.get("start_index", -1)),
        }
        prepared.append(Document(page_content=doc.page_content, metadata=metadata))
    return prepared


def _load_one_source(source: str, settings: Settings) -> list[Document]:
    if is_url(source):
        validate_public_url(source, allow_private_urls=settings.allow_private_urls)
        loader = WebBaseLoader([source], requests_kwargs={"timeout": settings.url_timeout_seconds})
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({"source": source, "source_type": "url"})
        return docs

    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    if path.is_dir():
        docs: list[Document] = []
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in SUPPORTED_FILE_EXTENSIONS:
                docs.extend(_load_local_file(child))
        if not docs:
            raise ValueError(f"No supported documents found in directory: {source}")
        return docs
    return _load_local_file(path)


def _load_local_file(path: Path, *, display_source: str | None = None) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FILE_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {suffix or '<none>'}")

    source = display_source or str(path.resolve())
    if suffix == ".pdf":
        docs = PyPDFLoader(str(path)).load()
        for doc in docs:
            doc.metadata.update({"source": source, "source_type": "pdf"})
        return docs

    raw_text = path.read_text(encoding="utf-8", errors="replace")
    source_type = _source_type_for_suffix(suffix)
    if suffix in {".html", ".htm"}:
        raw_text = BeautifulSoup(raw_text, "html.parser").get_text("\n", strip=True)

    return [Document(page_content=raw_text, metadata={"source": source, "source_type": source_type})]


def _source_type_for_suffix(suffix: str) -> str:
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix in {".html", ".htm"}:
        return "html"
    return "text"


def _normalize_page(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _citation_label(source: str, page: int | None) -> str:
    label = source
    if source.startswith("upload:"):
        label = source.removeprefix("upload:")
    elif not source.startswith(("http://", "https://")):
        label = Path(source).name or source
    if page is not None and page >= 0:
        return f"{label} p.{page + 1}"
    return label


def _chunk_id(
    *,
    collection: str,
    source: str,
    page: int | None,
    chunk_index: int,
    content_hash: str,
    embedding_model: str,
) -> str:
    raw = "|".join([collection, source, str(page), str(chunk_index), content_hash, embedding_model])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

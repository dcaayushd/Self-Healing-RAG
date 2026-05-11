from __future__ import annotations

from pathlib import Path


SYSTEM_SAMPLE_FILENAMES = {"self_healing_rag.md"}


def is_system_sample_source(source: str) -> bool:
    cleaned = source.removeprefix("upload:")
    return Path(cleaned).name in SYSTEM_SAMPLE_FILENAMES


def preferred_default_sources(sources: list[str]) -> list[str]:
    if not sources:
        return []
    non_sample = [source for source in sources if not is_system_sample_source(source)]
    return non_sample or sources


def display_source_name(source: str) -> str:
    cleaned = source.removeprefix("upload:")
    name = Path(cleaned).name
    return name or cleaned


def source_filter_where(sources: list[str] | None) -> dict | None:
    filtered = [source for source in sources or [] if source]
    if not filtered:
        return None
    if len(filtered) == 1:
        return {"source": filtered[0]}
    return {"source": {"$in": filtered}}

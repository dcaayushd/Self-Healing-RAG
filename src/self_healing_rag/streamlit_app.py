from __future__ import annotations

from typing import Any
from uuid import uuid4

import streamlit as st

from self_healing_rag.config import get_settings
from self_healing_rag.constants import FALLBACK_ANSWER
from self_healing_rag.schemas import AskResponse, IngestResponse
from self_healing_rag.service import RagService


SUPPORTED_UPLOAD_TYPES = ["pdf", "txt", "md", "markdown", "html", "htm"]


@st.cache_resource(show_spinner=False)
def get_service() -> RagService:
    return RagService()


def main() -> None:
    st.set_page_config(page_title="Self-Healing RAG", page_icon="SH", layout="wide")
    _init_state()
    settings = get_settings()
    service = get_service()

    st.title("Self-Healing RAG")
    st.caption("Retrieve, answer, critique grounding, and retry before returning a safe fallback.")

    with st.sidebar:
        st.header("Knowledge Base")
        collection = st.text_input("Collection", value=settings.default_collection)
        max_attempts = st.slider("Max attempts", min_value=1, max_value=6, value=settings.max_attempts)

        health = service.health()
        st.caption(f"Chat: `{health.chat_model}`")
        st.caption(f"Embeddings: `{health.embedding_model}`")

        st.divider()
        st.subheader("Ingest")
        _render_upload_ingest(service, collection)
        _render_source_ingest(service, collection)

        st.divider()
        st.subheader("Session")
        if st.button("New chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid4())
            st.rerun()
        st.caption(f"Thread: `{st.session_state.thread_id}`")

        confirm_reset = st.checkbox("Enable collection reset")
        if st.button("Reset collection", disabled=not confirm_reset, use_container_width=True):
            _run_action("Resetting collection...", service.reset_collection, collection)
            st.session_state.messages = []
            st.success(f"Reset `{collection}`.")

    _render_chat_history()

    if prompt := st.chat_input("Ask a question about the indexed documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving and checking grounding..."):
                    response = service.ask(
                        prompt,
                        collection=collection,
                        max_attempts=max_attempts,
                        thread_id=st.session_state.thread_id,
                    )
                _render_answer(response.model_dump())
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.answer, "response": response.model_dump()}
                )
            except Exception as exc:
                st.error(str(exc))
                st.session_state.messages.append({"role": "assistant", "content": FALLBACK_ANSWER})


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("thread_id", str(uuid4()))


def _render_upload_ingest(service: RagService, collection: str) -> None:
    uploaded_files = st.file_uploader(
        "Upload PDFs or docs",
        type=SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files=True,
    )
    if st.button("Ingest uploads", disabled=not uploaded_files, use_container_width=True):
        responses: list[IngestResponse] = []
        try:
            with st.spinner("Embedding uploaded files..."):
                for uploaded in uploaded_files or []:
                    responses.append(service.ingest_upload(uploaded.name, uploaded.getvalue(), collection=collection))
            chunks = sum(response.chunks_added for response in responses)
            st.success(f"Ingested {chunks} chunks from {len(responses)} uploaded file(s).")
        except Exception as exc:
            st.error(str(exc))


def _render_source_ingest(service: RagService, collection: str) -> None:
    raw_sources = st.text_area(
        "Local paths or exact URLs",
        placeholder="./data/docs\nhttps://example.com/single-page",
        height=96,
    )
    sources = _source_lines(raw_sources)
    if st.button("Ingest paths / URLs", disabled=not sources, use_container_width=True):
        try:
            response = _run_action("Embedding sources...", service.ingest_sources, sources, collection=collection)
            st.success(f"Ingested {response.chunks_added} chunks from {len(response.sources)} source(s).")
        except Exception as exc:
            st.error(str(exc))


def _render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("response"):
                _render_answer(message["response"])
            else:
                st.markdown(message["content"])


def _render_answer(response: dict[str, Any]) -> None:
    status = response.get("status", "insufficient_info")
    answer = response.get("answer", FALLBACK_ANSWER)
    if status == "answered":
        st.markdown(answer)
    else:
        st.warning(answer)

    citations = response.get("citations") or []
    if citations:
        with st.expander("Citations", expanded=True):
            st.dataframe(
                [
                    {
                        "id": item["id"],
                        "source": item["citation_label"],
                        "page": _page_label(item.get("page")),
                        "score": item.get("score"),
                        "excerpt": item["excerpt"],
                    }
                    for item in citations
                ],
                hide_index=True,
                use_container_width=True,
            )

    attempts = response.get("attempts") or []
    if attempts:
        with st.expander("Critic attempts"):
            for attempt in attempts:
                st.markdown(_attempt_summary(attempt))
                if attempt.get("missing_claims"):
                    st.caption("Missing claims: " + "; ".join(attempt["missing_claims"]))
                if attempt.get("invalid_citations"):
                    st.caption("Invalid citations: " + ", ".join(attempt["invalid_citations"]))


def _run_action(label: str, func, *args, **kwargs):
    with st.spinner(label):
        return func(*args, **kwargs)


def _source_lines(raw_sources: str) -> list[str]:
    return [line.strip() for line in raw_sources.splitlines() if line.strip()]


def _page_label(page: int | None) -> str:
    return "" if page is None else str(page + 1)


def _attempt_summary(attempt: dict[str, Any]) -> str:
    verdict = "accepted" if attempt.get("critic_accepted") else "rejected"
    return (
        f"**Attempt {attempt.get('attempt')}**: {verdict} · "
        f"{attempt.get('retrieved_count', 0)} chunk(s) · query `{attempt.get('query', '')}`\n\n"
        f"{attempt.get('critic_reason', '')}"
    )


if __name__ == "__main__":
    main()

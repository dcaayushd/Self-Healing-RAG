from __future__ import annotations

from html import escape
from typing import Any
from uuid import uuid4

import streamlit as st

from self_healing_rag.config import Settings, get_settings
from self_healing_rag.constants import FALLBACK_ANSWER
from self_healing_rag.diagnostics import run_diagnostics
from self_healing_rag.schemas import IndexStats, IngestResponse
from self_healing_rag.service import RagService
from self_healing_rag.source_policy import display_source_name, preferred_default_sources


SUPPORTED_UPLOAD_TYPES = ["pdf", "txt", "md", "markdown", "html", "htm"]
SAMPLE_QUESTIONS = [
    "What is this document about?",
    "What are the key facts?",
    "What should I know first?",
]


@st.cache_resource(show_spinner=False)
def get_service() -> RagService:
    return RagService()


def main() -> None:
    st.set_page_config(page_title="Self-Healing RAG", page_icon="SH", layout="wide")
    _init_state()
    _apply_theme(st.session_state.theme_mode)

    settings = get_settings()
    service = get_service()
    collection, max_attempts, stats, focus_sources = _render_sidebar(service, settings)
    _render_chat_page(
        service,
        collection=collection,
        max_attempts=max_attempts,
        stats=stats,
        focus_sources=focus_sources,
    )


def _render_sidebar(service: RagService, settings: Settings) -> tuple[str, int, IndexStats, list[str]]:
    with st.sidebar:
        _render_sidebar_brand()
        _render_chat_logs()
        st.divider()

        st.markdown("### Knowledge")
        collection = st.text_input("Collection", value=settings.default_collection)
        stats = _safe_stats(service, collection)

        with st.expander("Sources to search", expanded=not stats.is_empty):
            focus_sources = _render_source_scope(stats)

        with st.expander("Add documents", expanded=stats.is_empty):
            _render_upload_ingest(service, collection, key_prefix="sidebar_upload", compact=True)
            _render_source_ingest(service, collection, compact=True)

        st.divider()
        st.markdown("### Chat")
        st.selectbox("Theme", ["Auto", "Light", "Dark"], key="theme_mode")
        max_attempts = st.slider("Healing attempts", min_value=1, max_value=6, value=settings.max_attempts)
        if st.button("New chat", width="stretch"):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid4())
            st.rerun()
        st.caption(f"Thread `{st.session_state.thread_id}`")

        with st.expander("Advanced"):
            _render_runtime_details(service, stats, focus_sources)
            st.divider()
            _render_system_health(settings)
            st.divider()
            confirm_reset = st.checkbox("Enable collection reset")
            if st.button("Reset collection", disabled=not confirm_reset, width="stretch"):
                try:
                    _run_action("Resetting collection...", service.reset_collection, collection)
                    st.session_state.messages = []
                    st.session_state.ingest_events = []
                    st.session_state.focus_sources = []
                    st.session_state.upload_key += 1
                    st.success(f"Reset `{collection}`.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    return collection, max_attempts, stats, focus_sources


def _render_chat_page(
    service: RagService,
    *,
    collection: str,
    max_attempts: int,
    stats: IndexStats,
    focus_sources: list[str],
) -> None:
    st.markdown(
        """
        <div class="chat-header">
            <div>
                <h1>Self-Healing RAG</h1>
                <p>Ask your indexed documents. Every answer is retrieved, cited, checked, and retried before fallback.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if stats.is_empty:
        st.info("Add documents from the left sidebar to start a grounded chat.")
    elif not st.session_state.messages:
        _render_empty_chat_suggestions(stats)

    _render_chat_history()

    pending_prompt = st.session_state.pop("pending_prompt", None)
    if pending_prompt:
        _submit_prompt(
            service,
            pending_prompt,
            collection=collection,
            max_attempts=max_attempts,
            focus_sources=focus_sources,
        )
        return

    prompt = _render_prompt_input(disabled=stats.is_empty)
    if prompt:
        st.session_state.pending_prompt = prompt
        st.rerun()


def _render_empty_chat_suggestions(stats: IndexStats) -> None:
    _, center, _ = st.columns([0.12, 0.76, 0.12])
    with center:
        st.markdown(
            """
            <div class='empty-chat-copy'>
                <h3>Start with a grounded question</h3>
                <p>Ask a broad question, request key facts, or narrow sources in the sidebar when you mean one specific file.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cols = st.columns(len(SAMPLE_QUESTIONS))
        for idx, question in enumerate(SAMPLE_QUESTIONS):
            if cols[idx].button(question, disabled=stats.is_empty, width="stretch"):
                st.session_state.pending_prompt = question
                st.rerun()


def _render_prompt_input(*, disabled: bool) -> str | None:
    prompt = st.chat_input("Start typing...", disabled=disabled)
    if prompt and prompt.strip():
        return prompt.strip()
    return None


def _submit_prompt(
    service: RagService,
    prompt: str,
    *,
    collection: str,
    max_attempts: int,
    focus_sources: list[str],
) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    _record_chat_log(prompt)
    _render_user_bubble(prompt)
    thinking = st.empty()
    thinking.markdown(
        (
            "<div class='message-row assistant-row thinking-row'>"
            "<div class='assistant-avatar'>AI</div>"
            "<div class='assistant-bubble thinking-bubble'>Retrieving evidence, generating, and checking grounding...</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    try:
        response = service.ask(
            prompt,
            collection=collection,
            max_attempts=max_attempts,
            thread_id=st.session_state.thread_id,
            focus_sources=focus_sources,
        )
        response_payload = response.model_dump()
        st.session_state.messages.append({"role": "assistant", "content": response.answer, "response": response_payload})
    except Exception as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": FALLBACK_ANSWER,
                "response": {
                    "status": "insufficient_info",
                    "answer": FALLBACK_ANSWER,
                    "citations": [],
                    "attempts": [{"attempt": 1, "critic_reason": str(exc), "critic_accepted": False}],
                },
            }
        )
    thinking.empty()
    st.rerun()


def _render_upload_ingest(
    service: RagService,
    collection: str,
    *,
    key_prefix: str = "upload",
    compact: bool = False,
) -> None:
    uploaded_files = st.file_uploader(
        "Upload PDFs or docs" if not compact else "Pick files",
        type=SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files=True,
        key=f"{key_prefix}_{st.session_state.upload_key}",
        label_visibility="visible",
    )
    if compact:
        st.caption("PDF, TXT, Markdown, and HTML files. The picker resets after indexing.")
    if st.button(
        "Ingest uploads" if not compact else "Index selected files",
        disabled=not uploaded_files,
        width="stretch",
        key=f"{key_prefix}_button_{st.session_state.upload_key}",
    ):
        responses: list[IngestResponse] = []
        try:
            with st.spinner("Embedding uploaded files..."):
                for uploaded in uploaded_files or []:
                    responses.append(service.ingest_upload(uploaded.name, uploaded.getvalue(), collection=collection))
            _record_ingest(responses)
            chunks = sum(response.chunks_added for response in responses)
            _set_focus_sources([source for response in responses for source in response.sources])
            st.session_state.upload_key += 1
            st.success(f"Ingested {chunks} chunks from {len(responses)} uploaded file(s).")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def _render_source_ingest(service: RagService, collection: str, *, compact: bool = False) -> None:
    raw_sources = st.text_area(
        "Local paths or exact URLs",
        placeholder="./data/docs\nhttps://example.com/single-page",
        height=92 if compact else 104,
        key=f"{'sidebar' if compact else 'main'}_source_ingest",
    )
    sources = _source_lines(raw_sources)
    if st.button(
        "Index paths / URLs" if compact else "Ingest paths / URLs",
        disabled=not sources,
        width="stretch",
        key=f"{'sidebar' if compact else 'main'}_source_ingest_button",
    ):
        try:
            response = _run_action("Embedding sources...", service.ingest_sources, sources, collection=collection)
            _record_ingest([response])
            _set_focus_sources(response.sources)
            st.success(f"Ingested {response.chunks_added} chunks from {len(response.sources)} source(s).")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def _render_runtime_details(service: RagService, stats: IndexStats, focus_sources: list[str]) -> None:
    health = service.health()
    details = [
        ("Collection", stats.collection),
        ("Chunks", str(stats.chunk_count)),
        ("Sources", str(stats.source_count)),
        ("Retrieval scope", _scope_label(stats, focus_sources)),
        ("Chat model", health.chat_model),
        ("Embedding model", health.embedding_model),
    ]
    st.markdown("#### Runtime details")
    for label, value in details:
        st.caption(f"{label}: `{value}`")


def _render_system_health(settings: Settings) -> None:
    diagnostics = run_diagnostics(settings)
    for name, ok, detail in diagnostics.checks:
        status = "ok" if ok else "fix"
        st.caption(f"{status} · {name}: {detail}")


def _render_chat_history() -> None:
    for message in st.session_state.messages:
        if message["role"] == "assistant" and message.get("response"):
            _render_answer(message["response"])
        elif message["role"] == "assistant":
            _render_assistant_bubble(message["content"], status="insufficient_info")
        else:
            _render_user_bubble(message["content"])


def _render_answer(response: dict[str, Any]) -> None:
    status = response.get("status", "insufficient_info")
    answer = response.get("answer", FALLBACK_ANSWER)

    _render_assistant_bubble(answer, status=status)
    _render_answer_meta(response)
    _render_citations(response.get("citations") or [])
    _render_attempts(response.get("attempts") or [])


def _render_answer_meta(response: dict[str, Any]) -> None:
    attempts = response.get("attempts") or []
    if not attempts:
        return
    final_attempt = attempts[-1]
    status = "grounded" if response.get("status") == "answered" else "not enough evidence"
    confidence = _percent_label(final_attempt.get("retrieval_confidence"))
    strategy = final_attempt.get("retrieval_strategy", "hybrid")
    citations = len(response.get("citations") or [])
    st.markdown(
        (
            "<div class='answer-meta'>"
            f"<span>{escape(status)}</span>"
            f"<span>{escape(strategy)}</span>"
            f"<span>{escape(confidence)}</span>"
            f"<span>{citations} citation(s)</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_user_bubble(content: str) -> None:
    st.markdown(
        f"<div class='message-row user-row'><div class='user-bubble'>{_html_text(content)}</div></div>",
        unsafe_allow_html=True,
    )


def _render_assistant_bubble(content: str, *, status: str) -> None:
    tone = "answered" if status == "answered" else "insufficient"
    st.markdown(
        (
            "<div class='message-row assistant-row'>"
            "<div class='assistant-avatar'>AI</div>"
            f"<div class='assistant-bubble {tone}'>{_html_text(content)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        return
    with st.expander("Evidence citations", expanded=True):
        st.markdown("<div class='citation-list'>", unsafe_allow_html=True)
        for item in citations:
            page = _page_label(item.get("page")) or "n/a"
            relevance = _score_label(item.get("relevance")) or "n/a"
            st.markdown(
                (
                    "<div class='citation-card'>"
                    f"<div class='citation-title'>{escape(item['id'])} · {escape(item['citation_label'])}</div>"
                    f"<div class='citation-meta'>page {escape(page)} · relevance {escape(relevance)}</div>"
                    f"<p>{escape(item['excerpt'])}</p>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


def _render_attempts(attempts: list[dict[str, Any]]) -> None:
    if not attempts:
        return
    with st.expander("Critic attempt trace", expanded=False):
        for item in attempts:
            verdict = "accepted" if item.get("critic_accepted") else "rejected"
            confidence = _percent_label(item.get("retrieval_confidence"))
            st.markdown(
                (
                    "<div class='attempt-card'>"
                    f"<strong>Attempt {escape(str(item.get('attempt', '')))}</strong>"
                    f"<span>{escape(verdict)}</span>"
                    f"<span>{escape(str(item.get('retrieval_strategy', 'hybrid')))}</span>"
                    f"<span>{escape(confidence)}</span>"
                    f"<span>{escape(str(item.get('retrieved_count', 0)))} chunks</span>"
                    f"<code>{escape(item.get('query', ''))}</code>"
                    f"<p>{escape(item.get('critic_reason', ''))}</p>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def _render_chat_logs() -> None:
    logs = st.session_state.get("chat_logs", [])
    st.markdown(
        (
            "<div class='logs-header'>"
            "<div><strong>Chat logs</strong></div>"
            f"<span>{len(logs)}/50</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    if not logs:
        st.caption("Recent questions will appear here.")
        return

    for idx, prompt in enumerate(reversed(logs[-12:])):
        label = _short_label(prompt)
        if st.button(label, key=f"chat_log_{idx}_{hash(prompt)}", width="stretch"):
            st.session_state.pending_prompt = prompt
            st.rerun()

    if st.button("Clear history", key="clear_chat_logs", width="stretch"):
        st.session_state.chat_logs = []
        st.rerun()


def _render_sidebar_brand() -> None:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-mark">SH</div>
            <div>
                <strong>Self-Healing RAG</strong>
                <span>Grounded document chat</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _run_action(label: str, func, *args, **kwargs):
    with st.spinner(label):
        return func(*args, **kwargs)


def _record_ingest(responses: list[IngestResponse]) -> None:
    for response in responses:
        st.session_state.ingest_events.append(
            {
                "collection": response.collection,
                "chunks": response.chunks_added,
                "sources": len(response.sources),
            }
        )


def _set_focus_sources(sources: list[str]) -> None:
    filtered = [source for source in sources if source]
    if filtered:
        st.session_state.focus_sources = filtered


def _record_chat_log(prompt: str) -> None:
    normalized = prompt.strip()
    if not normalized:
        return
    logs = [item for item in st.session_state.get("chat_logs", []) if item != normalized]
    logs.append(normalized)
    st.session_state.chat_logs = logs[-50:]


def _render_source_scope(stats: IndexStats) -> list[str]:
    if stats.is_empty:
        st.caption("Index a document before choosing sources.")
        return []

    available = stats.sources
    current = [source for source in st.session_state.get("focus_sources", []) if source in available]
    mode = st.radio(
        "Retrieval scope",
        ["All indexed sources", "Choose sources"],
        horizontal=False,
        key="source_scope_mode",
        help="Use Choose sources when a question says 'this document' and you mean one specific file.",
    )
    if mode == "All indexed sources":
        st.session_state.focus_sources = []
        st.caption("The retriever searches the collection and diversifies evidence across sources.")
        return []

    default = current or preferred_default_sources(available)[: min(6, len(available))]
    selected = st.multiselect(
        "Selected sources",
        options=available,
        default=default,
        format_func=display_source_name,
        help="Limits retrieval to the selected source IDs.",
    )
    st.caption("Selected-source mode keeps vague questions grounded in the files you mean.")
    st.session_state.focus_sources = selected
    return selected


def _scope_label(stats: IndexStats, focus_sources: list[str]) -> str:
    if stats.is_empty:
        return "no evidence"
    if not focus_sources:
        return "searching all sources"
    if len(focus_sources) == 1:
        return f"searching {display_source_name(focus_sources[0])}"
    return f"searching {len(focus_sources)} selected sources"


def _safe_stats(service: RagService, collection: str) -> IndexStats:
    try:
        return service.collection_stats(collection)
    except Exception:
        return IndexStats(collection=collection, chunk_count=0, source_count=0)


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("thread_id", str(uuid4()))
    st.session_state.setdefault("ingest_events", [])
    st.session_state.setdefault("focus_sources", [])
    st.session_state.setdefault("upload_key", 0)
    st.session_state.setdefault("chat_logs", [])
    st.session_state.setdefault("theme_mode", "Auto")
    st.session_state.setdefault("source_scope_mode", "All indexed sources")


def _apply_theme(mode: str = "Auto") -> None:
    if mode == "Dark":
        tokens = _theme_tokens("dark")
        media_override = ""
    elif mode == "Light":
        tokens = _theme_tokens("light")
        media_override = ""
    else:
        tokens = _theme_tokens("light")
        dark = _theme_tokens("dark")
        media_override = f"""
        @media (prefers-color-scheme: dark) {{
            :root {{
                --rag-bg: {dark['bg']};
                --rag-sidebar: {dark['sidebar']};
                --rag-panel: {dark['panel']};
                --rag-card: {dark['card']};
                --rag-text: {dark['text']};
                --rag-muted: {dark['muted']};
                --rag-border: {dark['border']};
                --rag-soft-border: {dark['soft_border']};
                --rag-accent: {dark['accent']};
                --rag-accent-text: {dark['accent_text']};
                --rag-input: {dark['input']};
                --rag-input-text: {dark['input_text']};
                --rag-shadow: {dark['shadow']};
            }}
        }}
        """

    css = """
        <style>
        :root {
            --rag-bg: {tokens['bg']};
            --rag-sidebar: {tokens['sidebar']};
            --rag-panel: {tokens['panel']};
            --rag-card: {tokens['card']};
            --rag-text: {tokens['text']};
            --rag-muted: {tokens['muted']};
            --rag-border: {tokens['border']};
            --rag-soft-border: {tokens['soft_border']};
            --rag-accent: {tokens['accent']};
            --rag-accent-text: {tokens['accent_text']};
            --rag-input: {tokens['input']};
            --rag-input-text: {tokens['input_text']};
            --rag-shadow: {tokens['shadow']};
        }
        {media_override}
        .stApp {
            background: var(--rag-bg);
            color: var(--rag-text);
        }
        .block-container {
            max-width: none;
            min-height: 100vh;
            margin: 0;
            padding: 1.9rem 1.65rem 8.5rem 1.65rem;
            background: var(--rag-bg);
            border: 0;
            border-radius: 0;
            box-shadow: none;
        }
        section[data-testid="stSidebar"] {
            width: 320px !important;
            min-width: 320px !important;
            background: var(--rag-sidebar);
            border-right: 1px solid var(--rag-border);
            color: var(--rag-text);
        }
        section[data-testid="stSidebar"] > div:first-child {
            width: 320px !important;
            padding: 1rem 1.15rem 7rem 1.15rem;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            padding-top: 1rem !important;
        }
        section[data-testid="stSidebar"] * {
            color: var(--rag-text);
        }
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            border: 1px solid var(--rag-border);
            border-radius: 18px;
            background: linear-gradient(135deg, var(--rag-card), var(--rag-panel));
            padding: 0.85rem;
            margin: 0 0 1.05rem 0;
            box-shadow: var(--rag-shadow);
        }
        .brand-mark {
            width: 2.35rem;
            height: 2.35rem;
            border-radius: 12px;
            display: grid;
            place-items: center;
            background: var(--rag-accent);
            color: var(--rag-accent-text) !important;
            font-weight: 900;
            letter-spacing: 0;
        }
        .sidebar-brand strong {
            display: block;
            line-height: 1.1;
            color: var(--rag-text);
        }
        .sidebar-brand span {
            display: block;
            margin-top: 0.18rem;
            color: var(--rag-muted);
            font-size: 0.84rem;
            line-height: 1.25;
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] {
            background: var(--rag-card);
            border: 1px solid var(--rag-border);
            box-shadow: none;
        }
        div[data-testid="stExpander"] summary,
        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary {
            background: var(--rag-card) !important;
            color: var(--rag-text) !important;
            border-radius: 14px !important;
        }
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea {
            background: var(--rag-panel);
            color: var(--rag-text);
            border: 1px solid var(--rag-border);
        }
        h1 { letter-spacing: 0; margin-bottom: 0.15rem; }
        .chat-header {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1rem;
            position: sticky;
            top: 0;
            z-index: 900;
            padding: 0.75rem 0 0.75rem 0;
            background: linear-gradient(180deg, var(--rag-bg) 78%, color-mix(in srgb, var(--rag-bg) 0%, transparent));
        }
        .chat-header h1 {
            color: var(--rag-text);
            font-size: clamp(1.6rem, 3vw, 2.4rem);
            line-height: 1.02;
        }
        .chat-header p {
            color: var(--rag-muted);
            margin: 0.35rem 0 0 0;
            max-width: 46rem;
        }
        .eyebrow {
            color: var(--rag-accent);
            font-weight: 800;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .empty-chat-copy {
            margin: 2rem auto 1.2rem auto;
            text-align: center;
            max-width: 48rem;
        }
        .empty-chat-copy h3 {
            margin: 0 0 0.45rem 0;
            color: var(--rag-text);
            font-size: 1.35rem;
        }
        .empty-chat-copy p {
            margin: 0;
            color: var(--rag-muted);
        }
        div[data-testid="stExpander"] {
            border-radius: 14px;
            border-color: var(--rag-border);
            background: var(--rag-card);
        }
        .message-row {
            display: flex;
            gap: 0.65rem;
            margin: 0.85rem 0;
        }
        .user-row {
            justify-content: flex-end;
        }
        .assistant-row {
            justify-content: flex-start;
            align-items: flex-start;
        }
        .user-bubble {
            max-width: min(620px, 84%);
            background: var(--rag-accent);
            color: var(--rag-accent-text);
            border-radius: 18px 18px 4px 18px;
            padding: 0.78rem 0.95rem;
            font-weight: 650;
            box-shadow: var(--rag-shadow);
            line-height: 1.45;
        }
        .assistant-avatar {
            flex: 0 0 auto;
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            background: var(--rag-input);
            color: var(--rag-accent);
            display: grid;
            place-items: center;
            font-size: 0.72rem;
            font-weight: 800;
            margin-top: 0.2rem;
        }
        .assistant-bubble {
            max-width: min(680px, 88%);
            background: var(--rag-card);
            color: var(--rag-text);
            border: 1px solid var(--rag-border);
            border-radius: 18px 18px 18px 4px;
            padding: 0.92rem 1rem;
            box-shadow: var(--rag-shadow);
            line-height: 1.52;
        }
        .assistant-bubble.insufficient {
            border-color: color-mix(in srgb, var(--rag-accent) 44%, var(--rag-border));
            background: var(--rag-panel);
        }
        .answer-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: -0.35rem 0 0.85rem 2.65rem;
        }
        .answer-meta span {
            border: 1px solid var(--rag-border);
            border-radius: 999px;
            background: var(--rag-panel);
            color: var(--rag-muted);
            font-size: 0.78rem;
            padding: 0.16rem 0.52rem;
        }
        .thinking-bubble {
            color: var(--rag-muted);
            font-weight: 650;
        }
        .thinking-bubble::before {
            content: "";
            display: inline-block;
            width: 0.58rem;
            height: 0.58rem;
            margin-right: 0.55rem;
            border-radius: 999px;
            background: var(--rag-accent);
            animation: ragPulse 1.1s ease-in-out infinite;
        }
        @keyframes ragPulse {
            0%, 100% { transform: scale(0.7); opacity: 0.45; }
            50% { transform: scale(1); opacity: 1; }
        }
        div[data-testid="stChatInput"] {
            position: fixed;
            left: 344px !important;
            right: 1.5rem !important;
            bottom: 1rem !important;
            width: auto !important;
            max-width: none !important;
            z-index: 1000;
            padding: 0.75rem 0;
            background: linear-gradient(180deg, color-mix(in srgb, var(--rag-bg) 8%, transparent), var(--rag-bg) 46%);
        }
        div[data-testid="stChatInput"] > div {
            border: 1px solid var(--rag-border);
            border-radius: 16px;
            background: var(--rag-panel);
            padding: 0.35rem;
            box-shadow: var(--rag-shadow);
        }
        div[data-testid="stChatInput"] textarea {
            border-radius: 999px;
            background: var(--rag-input);
            color: var(--rag-input-text);
            border: 0;
        }
        div[data-testid="stChatInput"] textarea::placeholder {
            color: color-mix(in srgb, var(--rag-input-text) 62%, transparent);
        }
        .logs-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.55rem 0 0.9rem 0;
            color: var(--rag-text);
        }
        .logs-header span {
            color: var(--rag-muted);
            font-size: 0.82rem;
        }
        .citation-list {
            display: grid;
            gap: 0.6rem;
        }
        .citation-card,
        .attempt-card {
            border: 1px solid var(--rag-border);
            border-radius: 12px;
            background: var(--rag-card);
            padding: 0.85rem 0.95rem;
        }
        .citation-title {
            font-weight: 700;
            color: var(--rag-text);
            margin-bottom: 0.15rem;
        }
        .citation-meta {
            color: var(--rag-muted);
            font-size: 0.82rem;
            margin-bottom: 0.55rem;
        }
        .citation-card p,
        .attempt-card p {
            margin: 0.2rem 0 0 0;
            color: var(--rag-muted);
            line-height: 1.45;
        }
        .attempt-card {
            display: grid;
            grid-template-columns: auto auto auto auto auto 1fr;
            gap: 0.5rem;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        .attempt-card span {
            border: 1px solid var(--rag-border);
            border-radius: 999px;
            padding: 0.12rem 0.5rem;
            color: var(--rag-muted);
            font-size: 0.82rem;
        }
        .attempt-card code {
            white-space: normal;
            color: var(--rag-muted);
        }
        .attempt-card p {
            grid-column: 1 / -1;
        }
        .stButton > button {
            border-radius: 999px;
            border: 1px solid var(--rag-border);
            background: var(--rag-card);
            color: var(--rag-text);
            box-shadow: none;
        }
        .stButton > button:hover {
            border-color: var(--rag-accent);
            color: var(--rag-text);
            background: var(--rag-panel);
        }
        [data-baseweb="tag"] {
            background: var(--rag-accent) !important;
            color: var(--rag-accent-text) !important;
            border-radius: 999px !important;
        }
        [data-baseweb="tag"] span {
            color: var(--rag-accent-text) !important;
        }
        [data-baseweb="select"] > div {
            background: var(--rag-panel) !important;
            border-color: var(--rag-border) !important;
            color: var(--rag-text) !important;
        }
        @media (max-width: 900px) {
            section[data-testid="stSidebar"],
            section[data-testid="stSidebar"] > div:first-child {
                width: min(88vw, 320px) !important;
                min-width: min(88vw, 320px) !important;
            }
            .block-container {
                border-radius: 0;
                min-height: 100vh;
                padding-top: 3.5rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
            div[data-testid="stChatInput"] {
                left: 1rem;
                right: 1rem;
                bottom: 0.75rem;
            }
            .attempt-card { grid-template-columns: 1fr; }
        }
        </style>
        """
    replacements = {
        "{tokens['bg']}": tokens["bg"],
        "{tokens['sidebar']}": tokens["sidebar"],
        "{tokens['panel']}": tokens["panel"],
        "{tokens['card']}": tokens["card"],
        "{tokens['text']}": tokens["text"],
        "{tokens['muted']}": tokens["muted"],
        "{tokens['border']}": tokens["border"],
        "{tokens['soft_border']}": tokens["soft_border"],
        "{tokens['accent']}": tokens["accent"],
        "{tokens['accent_text']}": tokens["accent_text"],
        "{tokens['input']}": tokens["input"],
        "{tokens['input_text']}": tokens["input_text"],
        "{tokens['shadow']}": tokens["shadow"],
        "{media_override}": media_override,
    }
    for placeholder, value in replacements.items():
        css = css.replace(placeholder, value)
    st.markdown(css, unsafe_allow_html=True)


def _theme_tokens(mode: str) -> dict[str, str]:
    if mode == "dark":
        return {
            "bg": "#101318",
            "sidebar": "#171b22",
            "panel": "#1d222b",
            "card": "#222832",
            "text": "#f5f1e8",
            "muted": "rgba(245, 241, 232, 0.66)",
            "border": "rgba(245, 241, 232, 0.12)",
            "soft_border": "rgba(245, 241, 232, 0.08)",
            "accent": "#f0b429",
            "accent_text": "#171204",
            "input": "#080a0d",
            "input_text": "#fffaf0",
            "shadow": "0 16px 44px rgba(0, 0, 0, 0.22)",
        }
    return {
        "bg": "#f4efe4",
        "sidebar": "#fbf7ef",
        "panel": "#f7f1e6",
        "card": "#fffdf8",
        "text": "#1f211d",
        "muted": "rgba(31, 33, 29, 0.62)",
        "border": "rgba(31, 33, 29, 0.12)",
        "soft_border": "rgba(31, 33, 29, 0.08)",
        "accent": "#f0b429",
        "accent_text": "#171204",
        "input": "#17191f",
        "input_text": "#fffdf8",
        "shadow": "0 16px 44px rgba(65, 50, 20, 0.10)",
    }


def _source_lines(raw_sources: str) -> list[str]:
    return [line.strip() for line in raw_sources.splitlines() if line.strip()]


def _page_label(page: int | None) -> str:
    return "" if page is None else str(page + 1)


def _score_label(score: float | None) -> str:
    return "" if score is None else f"{score:.4f}"


def _percent_label(score: float | None) -> str:
    if score is None:
        return "confidence n/a"
    return f"{round(score * 100)}% confidence"


def _short_label(text: str, *, limit: int = 38) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "..."


def _html_text(text: str) -> str:
    return "<br>".join(escape(text).splitlines())


def _attempt_summary(attempt: dict[str, Any]) -> str:
    verdict = "accepted" if attempt.get("critic_accepted") else "rejected"
    return (
        f"**Attempt {attempt.get('attempt')}**: {verdict} · "
        f"{attempt.get('retrieved_count', 0)} chunk(s) · query `{attempt.get('query', '')}`\n\n"
        f"{attempt.get('critic_reason', '')}"
    )


if __name__ == "__main__":
    main()

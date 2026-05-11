from __future__ import annotations

import sqlite3
import re
from typing import Literal, Protocol
from uuid import uuid4

import warnings
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from self_healing_rag.config import Settings
from self_healing_rag.constants import FALLBACK_ANSWER
from self_healing_rag.prompts import (
    ANSWER_SYSTEM,
    CRITIC_SYSTEM,
    REWRITE_SYSTEM,
    answer_user_prompt,
    cited_ids,
    critic_user_prompt,
    normalize_answer_citations,
    parse_critic_json,
    rewrite_user_prompt,
)
from self_healing_rag.schemas import AskResponse, AttemptTrace, Citation, CriticResult, RetrievedChunk
from self_healing_rag.vector_store import VectorStoreManager


class RagComponents(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        collection: str,
        top_k: int,
        fetch_k: int,
        focus_sources: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        ...

    def generate_answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        ...

    def critique(self, question: str, chunks: list[RetrievedChunk], answer: str) -> CriticResult:
        ...

    def reformulate(self, question: str, current_query: str, critic: CriticResult) -> str:
        ...


class RagState(TypedDict, total=False):
    question: str
    collection: str
    current_query: str
    retrieved_chunks: list[dict]
    answer: str
    critic: dict
    attempt_count: int
    max_attempts: int
    status: str
    attempts: list[dict]
    error_notes: list[str]
    runtime_error: str
    focus_sources: list[str]
    retrieval_strategy: str
    retrieval_confidence: float


class OllamaRagComponents:
    def __init__(self, settings: Settings, vector_store: VectorStoreManager | None = None) -> None:
        self.settings = settings
        self.vector_store = vector_store or VectorStoreManager(settings)
        self.llm = ChatOllama(
            model=settings.chat_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

    def retrieve(
        self,
        query: str,
        *,
        collection: str,
        top_k: int,
        fetch_k: int,
        focus_sources: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if is_overview_question(query):
            chunks = self.vector_store.overview(collection=collection, limit=top_k, focus_sources=focus_sources)
            if chunks:
                return chunks
        return self.vector_store.search(
            query,
            collection=collection,
            top_k=top_k,
            fetch_k=fetch_k,
            focus_sources=focus_sources,
        )

    def generate_answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return FALLBACK_ANSWER
        response = self.llm.invoke(
            [
                SystemMessage(content=ANSWER_SYSTEM),
                HumanMessage(content=answer_user_prompt(question, chunks, overview=is_overview_question(question))),
            ]
        )
        return _message_content(response.content)

    def critique(self, question: str, chunks: list[RetrievedChunk], answer: str) -> CriticResult:
        if not chunks:
            return CriticResult(accepted=False, reason="No retrieved chunks were available.")
        response = self.llm.invoke(
            [
                SystemMessage(content=CRITIC_SYSTEM),
                HumanMessage(content=critic_user_prompt(question, chunks, answer)),
            ]
        )
        return parse_critic_json(_message_content(response.content), answer=answer, chunks=chunks)

    def reformulate(self, question: str, current_query: str, critic: CriticResult) -> str:
        response = self.llm.invoke(
            [
                SystemMessage(content=REWRITE_SYSTEM),
                HumanMessage(content=rewrite_user_prompt(question, current_query, critic)),
            ]
        )
        rewritten = _message_content(response.content).strip().strip('"')
        return rewritten or question


class SelfHealingRag:
    def __init__(
        self,
        settings: Settings,
        components: RagComponents | None = None,
        *,
        use_checkpointer: bool = True,
    ) -> None:
        self.settings = settings
        self.components = components or OllamaRagComponents(settings)
        self._checkpoint_conn: sqlite3.Connection | None = None
        self.graph = self._build_graph(use_checkpointer=use_checkpointer)

    def ask(
        self,
        question: str,
        *,
        collection: str | None = None,
        max_attempts: int | None = None,
        thread_id: str | None = None,
        focus_sources: list[str] | None = None,
    ) -> AskResponse:
        thread_id = thread_id or str(uuid4())
        attempts = max(1, max_attempts or self.settings.max_attempts)
        initial: RagState = {
            "question": question,
            "collection": collection or self.settings.default_collection,
            "current_query": question,
            "attempt_count": 0,
            "max_attempts": attempts,
            "attempts": [],
            "error_notes": [],
            "focus_sources": focus_sources or [],
        }
        result = self.graph.invoke(initial, config={"configurable": {"thread_id": thread_id}})
        return _response_from_state(result, thread_id)

    def _build_graph(self, *, use_checkpointer: bool):
        builder = StateGraph(RagState)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate", self._generate)
        builder.add_node("critique", self._critique)
        builder.add_node("reformulate", self._reformulate)
        builder.add_node("finalize", self._finalize)

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", "critique")
        builder.add_conditional_edges(
            "critique",
            self._route_after_critique,
            {"finalize": "finalize", "reformulate": "reformulate"},
        )
        builder.add_edge("reformulate", "retrieve")
        builder.add_edge("finalize", END)

        if not use_checkpointer:
            return builder.compile()
        self.settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_conn = sqlite3.connect(str(self.settings.checkpoint_db), check_same_thread=False)
        return builder.compile(checkpointer=SqliteSaver(self._checkpoint_conn))

    def _retrieve(self, state: RagState) -> RagState:
        query = state.get("current_query") or state["question"]
        attempt_count = int(state.get("attempt_count", 0)) + 1
        try:
            chunks = self.components.retrieve(
                query,
                collection=state.get("collection", self.settings.default_collection),
                top_k=self.settings.top_k,
                fetch_k=self.settings.fetch_k,
                focus_sources=state.get("focus_sources", []),
            )
        except Exception as exc:
            message = f"Retrieval failed: {exc}"
            notes = list(state.get("error_notes", []))
            notes.append(message)
            return {
                "attempt_count": attempt_count,
                "retrieved_chunks": [],
                "answer": FALLBACK_ANSWER,
                "error_notes": notes,
                "runtime_error": message,
                "retrieval_strategy": "error",
                "retrieval_confidence": 0.0,
            }
        strategy = _retrieval_strategy(query, state.get("focus_sources", []))
        return {
            "attempt_count": attempt_count,
            "retrieved_chunks": [chunk.model_dump() for chunk in chunks],
            "runtime_error": "",
            "retrieval_strategy": strategy,
            "retrieval_confidence": _retrieval_confidence(chunks),
        }

    def _generate(self, state: RagState) -> RagState:
        if state.get("runtime_error"):
            return {"answer": FALLBACK_ANSWER}
        chunks = _chunks_from_state(state)
        try:
            answer = self.components.generate_answer(state["question"], chunks)
        except Exception as exc:
            message = f"Generation failed: {exc}"
            notes = list(state.get("error_notes", []))
            notes.append(message)
            return {"answer": FALLBACK_ANSWER, "error_notes": notes, "runtime_error": message}
        normalized = normalize_answer_citations(answer.strip() or FALLBACK_ANSWER, chunks)
        return {"answer": normalized}

    def _critique(self, state: RagState) -> RagState:
        chunks = _chunks_from_state(state)
        answer = state.get("answer", FALLBACK_ANSWER)
        if state.get("runtime_error"):
            critic = CriticResult(accepted=False, reason=state["runtime_error"])
        else:
            try:
                critic = self.components.critique(state["question"], chunks, answer)
            except Exception as exc:
                critic = CriticResult(
                    accepted=False,
                    reason=f"Critic rejected because it returned malformed output or failed: {exc}",
                )

        attempt = AttemptTrace(
            attempt=int(state.get("attempt_count", 1)),
            query=state.get("current_query", state["question"]),
            retrieved_count=len(chunks),
            answer=answer,
            critic_accepted=critic.accepted,
            critic_reason=critic.reason,
            missing_claims=critic.missing_claims,
            invalid_citations=critic.invalid_citations,
            retrieval_strategy=state.get("retrieval_strategy", "vector"),
            retrieval_confidence=state.get("retrieval_confidence"),
        )
        return {
            "critic": critic.model_dump(),
            "attempts": list(state.get("attempts", [])) + [attempt.model_dump()],
        }

    def _route_after_critique(self, state: RagState) -> Literal["finalize", "reformulate"]:
        critic = CriticResult.model_validate(state.get("critic", {"accepted": False, "reason": "Missing critic result."}))
        if critic.accepted:
            return "finalize"
        if state.get("runtime_error"):
            return "finalize"
        if int(state.get("attempt_count", 0)) < int(state.get("max_attempts", self.settings.max_attempts)):
            return "reformulate"
        return "finalize"

    def _reformulate(self, state: RagState) -> RagState:
        critic = CriticResult.model_validate(state.get("critic", {"accepted": False, "reason": "Missing critic result."}))
        current_query = state.get("current_query", state["question"])
        try:
            rewritten = self.components.reformulate(state["question"], current_query, critic)
        except Exception as exc:
            notes = list(state.get("error_notes", []))
            notes.append(f"Query reformulation failed: {exc}")
            rewritten = f"{state['question']} {critic.reason}".strip()
            return {"current_query": sanitize_retrieval_query(rewritten), "error_notes": notes}
        return {"current_query": sanitize_retrieval_query(rewritten) or state["question"]}

    def _finalize(self, state: RagState) -> RagState:
        critic = CriticResult.model_validate(state.get("critic", {"accepted": False, "reason": "Missing critic result."}))
        if critic.accepted:
            return {"status": "answered"}
        return {"status": "insufficient_info", "answer": FALLBACK_ANSWER}


def _response_from_state(state: RagState, thread_id: str) -> AskResponse:
    attempts = [AttemptTrace.model_validate(attempt) for attempt in state.get("attempts", [])]
    status = state.get("status", "insufficient_info")
    if status != "answered":
        return AskResponse(
            status="insufficient_info",
            answer=FALLBACK_ANSWER,
            citations=[],
            attempts=attempts,
            thread_id=thread_id,
            focus_sources=state.get("focus_sources", []),
        )

    answer = state.get("answer", FALLBACK_ANSWER)
    citations = _citations_for_answer(answer, _chunks_from_state(state))
    return AskResponse(
        status="answered",
        answer=answer,
        citations=citations,
        attempts=attempts,
        thread_id=thread_id,
        focus_sources=state.get("focus_sources", []),
    )


def _citations_for_answer(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    used = cited_ids(answer)
    return [chunk.to_citation() for chunk in chunks if chunk.citation_id in used]


def _chunks_from_state(state: RagState) -> list[RetrievedChunk]:
    return [RetrievedChunk.model_validate(chunk) for chunk in state.get("retrieved_chunks", [])]


def _message_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", ""))))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def is_overview_question(question: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\s]", " ", question.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return False

    overview_phrases = (
        "what is this document about",
        "what is the document about",
        "what s this document about",
        "whats this document about",
        "what is this about",
        "what s this about",
        "whats this about",
        "what is it about",
        "what are these documents about",
        "what is this file about",
        "what is this pdf about",
        "what s this pdf about",
        "whats this pdf about",
        "summarize this document",
        "summarise this document",
        "summarize this file",
        "summarise this file",
        "summarize this pdf",
        "summarise this pdf",
        "summarize the document",
        "summarise the document",
        "tell me about this document",
        "tell me about this file",
        "tell me about this pdf",
        "give me an overview",
        "document overview",
        "main topic of this document",
        "main idea of this document",
    )
    if any(phrase in normalized for phrase in overview_phrases):
        return True

    tokens = normalized.split()
    if len(tokens) <= 10 and {"document", "about"}.issubset(tokens):
        return True
    if len(tokens) <= 10 and ("summary" in tokens or "summarize" in tokens or "summarise" in tokens):
        return True
    return False


def sanitize_retrieval_query(query: str, *, max_length: int = 240) -> str:
    sanitized = re.sub(r"^\s*(rewritten query|query|search query)\s*:\s*", "", query, flags=re.IGNORECASE)
    sanitized = sanitized.strip().strip('"').strip("'")
    sanitized = re.sub(r"\s+", " ", sanitized)
    if len(sanitized) <= max_length:
        return sanitized
    truncated = sanitized[:max_length].rsplit(" ", 1)[0].strip()
    return truncated or sanitized[:max_length].strip()


def _retrieval_strategy(query: str, focus_sources: list[str]) -> str:
    strategy = "overview" if is_overview_question(query) else "hybrid"
    if focus_sources:
        return f"focused-{strategy}"
    return strategy


def _retrieval_confidence(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    relevance_scores = [chunk.relevance for chunk in chunks if chunk.relevance is not None]
    if relevance_scores:
        top_scores = relevance_scores[: min(3, len(relevance_scores))]
        return round(sum(top_scores) / len(top_scores), 3)
    return 1.0

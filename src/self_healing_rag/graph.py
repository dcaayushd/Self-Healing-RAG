from __future__ import annotations

import sqlite3
import re
from time import perf_counter
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
    validate_critic_result,
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
    retrieval_ms: float
    generation_ms: float
    answerability_error: str


class OllamaRagComponents:
    def __init__(self, settings: Settings, vector_store: VectorStoreManager | None = None) -> None:
        self.settings = settings
        self.vector_store = vector_store or VectorStoreManager(settings)
        common_llm_kwargs = {
            "model": settings.chat_model,
            "base_url": settings.ollama_base_url,
            "temperature": 0,
            "num_ctx": settings.llm_num_ctx,
            "keep_alive": settings.ollama_keep_alive,
        }
        self.answer_llm = ChatOllama(
            **common_llm_kwargs,
            num_predict=settings.answer_num_predict,
        )
        self.critic_llm = ChatOllama(
            **common_llm_kwargs,
            format="json",
            num_predict=settings.critic_num_predict,
        )
        self.rewrite_llm = ChatOllama(
            **common_llm_kwargs,
            num_predict=settings.rewrite_num_predict,
        )
        self.llm = self.answer_llm

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
        response = self.answer_llm.invoke(
            [
                SystemMessage(content=ANSWER_SYSTEM),
                HumanMessage(content=answer_user_prompt(question, chunks, overview=is_overview_question(question))),
            ]
        )
        return _message_content(response.content)

    def critique(self, question: str, chunks: list[RetrievedChunk], answer: str) -> CriticResult:
        if not chunks:
            return CriticResult(accepted=False, reason="No retrieved chunks were available.")
        response = self.critic_llm.invoke(
            [
                SystemMessage(content=CRITIC_SYSTEM),
                HumanMessage(content=critic_user_prompt(question, chunks, answer)),
            ]
        )
        return parse_critic_json(_message_content(response.content), answer=answer, chunks=chunks)

    def reformulate(self, question: str, current_query: str, critic: CriticResult) -> str:
        response = self.rewrite_llm.invoke(
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
        started = perf_counter()
        result = self.graph.invoke(initial, config={"configurable": {"thread_id": thread_id}})
        return _response_from_state(result, thread_id, total_ms=_elapsed_ms(started))

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
        started = perf_counter()
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
                "retrieval_ms": _elapsed_ms(started),
                "answerability_error": message,
            }
        strategy = _retrieval_strategy(query, state.get("focus_sources", []))
        confidence = _retrieval_confidence(query, chunks)
        answerability_error = _answerability_error(
            query,
            chunks,
            confidence=confidence,
            min_confidence=self.settings.min_retrieval_confidence,
        )
        return {
            "attempt_count": attempt_count,
            "retrieved_chunks": [chunk.model_dump() for chunk in chunks],
            "runtime_error": "",
            "retrieval_strategy": strategy,
            "retrieval_confidence": confidence,
            "retrieval_ms": _elapsed_ms(started),
            "answerability_error": answerability_error,
        }

    def _generate(self, state: RagState) -> RagState:
        started = perf_counter()
        if state.get("runtime_error"):
            return {"answer": FALLBACK_ANSWER, "generation_ms": _elapsed_ms(started)}
        if state.get("answerability_error"):
            return {"answer": FALLBACK_ANSWER, "generation_ms": _elapsed_ms(started)}
        chunks = _chunks_from_state(state)
        try:
            answer = self.components.generate_answer(state["question"], chunks)
        except Exception as exc:
            message = f"Generation failed: {exc}"
            notes = list(state.get("error_notes", []))
            notes.append(message)
            return {
                "answer": FALLBACK_ANSWER,
                "error_notes": notes,
                "runtime_error": message,
                "generation_ms": _elapsed_ms(started),
            }
        normalized = normalize_answer_citations(answer.strip() or FALLBACK_ANSWER, chunks)
        return {"answer": normalized, "generation_ms": _elapsed_ms(started)}

    def _critique(self, state: RagState) -> RagState:
        started = perf_counter()
        chunks = _chunks_from_state(state)
        answer = state.get("answer", FALLBACK_ANSWER)
        if state.get("runtime_error"):
            critic = CriticResult(accepted=False, reason=state["runtime_error"])
        elif state.get("answerability_error"):
            critic = CriticResult(accepted=False, reason=state["answerability_error"])
        else:
            critic = _local_critic(answer, chunks)
            if self.settings.critic_mode in {"hybrid", "llm"} and (critic.accepted or self.settings.critic_mode == "llm"):
                try:
                    llm_critic = self.components.critique(state["question"], chunks, answer)
                except Exception as exc:
                    critic = CriticResult(
                        accepted=False,
                        reason=f"Critic rejected because it returned malformed output or failed: {exc}",
                    )
                else:
                    critic = validate_critic_result(llm_critic, answer=answer, chunks=chunks)

        critique_ms = _elapsed_ms(started)
        retrieval_ms = state.get("retrieval_ms")
        generation_ms = state.get("generation_ms")
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
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            critique_ms=critique_ms,
            total_ms=_sum_ms(retrieval_ms, generation_ms, critique_ms),
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
        if not self.settings.use_llm_rewrite:
            rewritten = _fallback_rewrite(state["question"], current_query, critic)
            return {"current_query": sanitize_retrieval_query(rewritten) or state["question"]}
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


def _response_from_state(state: RagState, thread_id: str, *, total_ms: float | None = None) -> AskResponse:
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
            total_ms=total_ms,
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
        total_ms=total_ms,
    )


def _citations_for_answer(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    used = cited_ids(answer)
    return [chunk.to_citation() for chunk in chunks if chunk.citation_id in used]


def _chunks_from_state(state: RagState) -> list[RetrievedChunk]:
    return [RetrievedChunk.model_validate(chunk) for chunk in state.get("retrieved_chunks", [])]


def _local_critic(answer: str, chunks: list[RetrievedChunk]) -> CriticResult:
    result = validate_critic_result(CriticResult(accepted=True, reason=""), answer=answer, chunks=chunks)
    if result.accepted:
        result.reason = "Accepted by local grounding validator."
    elif not result.reason or result.reason == "Rejected by critic.":
        result.reason = "Rejected by local grounding validator."
    return result


def _fallback_rewrite(question: str, current_query: str, critic: CriticResult) -> str:
    missing = " ".join(critic.missing_claims[:2])
    invalid = " ".join(critic.invalid_citations[:2])
    parts = [question, current_query, critic.reason, missing, invalid]
    return " ".join(part for part in parts if part).strip()


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
        "what are the key facts",
        "what are key facts",
        "key facts",
        "what are the key aspects",
        "key aspects",
        "what are the main points",
        "main points",
        "important points",
        "what should i know first",
        "what should i know",
        "give me the gist",
        "high level summary",
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
    if len(tokens) <= 10 and {"key", "facts"}.issubset(tokens):
        return True
    if len(tokens) <= 10 and {"key", "aspects"}.issubset(tokens):
        return True
    if len(tokens) <= 10 and {"main", "points"}.issubset(tokens):
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


def _retrieval_confidence(query: str, chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    relevance_scores = [chunk.relevance for chunk in chunks if chunk.relevance is not None]
    if not relevance_scores:
        return 1.0
    if is_overview_question(query):
        return 1.0

    ordered_scores = sorted(relevance_scores, reverse=True)
    top_score = ordered_scores[0]
    mean_top3 = sum(ordered_scores[: min(3, len(ordered_scores))]) / min(3, len(ordered_scores))
    calibrated_relevance = _calibrate_relevance(top_score, mean_top3)
    coverage = _query_evidence_coverage(query, chunks)
    supporting_chunk_ratio = _supporting_chunk_ratio(query, chunks)

    confidence = (0.52 * calibrated_relevance) + (0.34 * coverage) + (0.14 * supporting_chunk_ratio)
    if top_score >= 0.35 and coverage >= 0.80:
        confidence = max(confidence, 0.72)
    elif top_score >= 0.35 and coverage >= 0.50:
        confidence = max(confidence, 0.62)
    elif top_score >= 0.25 and coverage >= 0.50:
        confidence = max(confidence, 0.52)
    if coverage == 0:
        confidence = min(confidence, 0.30)
    return round(max(0.0, min(1.0, confidence)), 3)


def _calibrate_relevance(top_score: float, mean_top3: float) -> float:
    blended = max(0.0, min(1.0, (0.72 * top_score) + (0.28 * mean_top3)))
    return round(blended**0.58, 3)


def _query_evidence_coverage(query: str, chunks: list[RetrievedChunk]) -> float:
    query_tokens = _confidence_tokens(query)
    if not query_tokens:
        return 1.0
    evidence_tokens: set[str] = set()
    for chunk in chunks:
        evidence_tokens |= _confidence_tokens(f"{chunk.content} {chunk.source} {chunk.citation_label}")
    return len(query_tokens & evidence_tokens) / len(query_tokens)


def _supporting_chunk_ratio(query: str, chunks: list[RetrievedChunk]) -> float:
    query_tokens = _confidence_tokens(query)
    if not query_tokens:
        return 1.0
    supporting = 0
    for chunk in chunks:
        chunk_tokens = _confidence_tokens(f"{chunk.content} {chunk.source} {chunk.citation_label}")
        if not chunk_tokens:
            continue
        overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
        relevance = chunk.relevance or 0.0
        if overlap >= 0.35 or relevance >= 0.35:
            supporting += 1
    return min(1.0, supporting / min(3, max(1, len(chunks))))


def _confidence_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]{3,}", text.lower()) if token not in _CONFIDENCE_STOPWORDS}


def _answerability_error(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    confidence: float,
    min_confidence: float,
) -> str:
    if not chunks:
        return "No retrieved chunks were available."
    if is_overview_question(query):
        return ""
    has_scored_chunks = any(chunk.relevance is not None for chunk in chunks)
    if has_scored_chunks and confidence < min_confidence:
        return (
            f"Retrieved evidence confidence {confidence:.2f} is below the configured "
            f"minimum {min_confidence:.2f}."
        )
    return ""


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 1)


def _sum_ms(*values: float | None) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return round(sum(present), 1)


_CONFIDENCE_STOPWORDS = {
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
    "document",
    "documents",
}

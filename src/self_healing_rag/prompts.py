from __future__ import annotations

import json
import re
from pathlib import Path

from self_healing_rag.constants import FALLBACK_ANSWER
from self_healing_rag.schemas import CriticResult, RetrievedChunk


ANSWER_SYSTEM = f"""You are a grounded retrieval assistant.
Answer only with facts supported by the retrieved chunks.
Every factual sentence must include at least one inline citation like [C1].
Use only the citation IDs shown in the context.
For broad overview questions such as "what is this document about?", synthesize the document's main purpose, workflow, and important components from the retrieved chunks.
Prefer a concise 2-4 sentence answer unless the question asks for more detail.
If the context does not contain enough information, respond exactly:
{FALLBACK_ANSWER}
"""

CRITIC_SYSTEM = """You are a strict grounding critic for a RAG system.
Decide whether the answer is fully grounded in the retrieved chunks.
Reject answers with unsupported claims, missing citations, invalid citation IDs, or claims that go beyond the context.
Return JSON only with this shape:
{"accepted": true, "reason": "short reason", "missing_claims": [], "invalid_citations": []}
"""

REWRITE_SYSTEM = """You reformulate retrieval queries for a RAG system.
Use the original user question and the critic feedback to create a more specific search query.
Return only the rewritten query.
"""

_CITATION_RE = re.compile(r"\[(C\d+)\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for chunk in chunks:
        location = chunk.citation_label
        blocks.append(f"[{chunk.citation_id}] Source: {location}\n{chunk.content}")
    return "\n\n".join(blocks)


def answer_user_prompt(question: str, chunks: list[RetrievedChunk], *, overview: bool = False) -> str:
    task = ""
    if overview:
        task = (
            "\nTask hint:\n"
            "This is a broad overview question. Summarize the retrieved evidence as a whole. "
            "If chunks come from multiple sources, describe the collection as multiple documents and cover the main topics across sources. "
            "Do not summarize only the first chunk unless the retrieved context only contains one source.\n"
        )
    return f"Question:\n{question}{task}\nRetrieved chunks:\n{build_context(chunks)}"


def critic_user_prompt(question: str, chunks: list[RetrievedChunk], answer: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Retrieved chunks:\n{build_context(chunks)}\n\n"
        f"Answer to critique:\n{answer}"
    )


def rewrite_user_prompt(question: str, current_query: str, critic: CriticResult) -> str:
    feedback = {
        "reason": critic.reason,
        "missing_claims": critic.missing_claims,
        "invalid_citations": critic.invalid_citations,
    }
    return (
        f"Original question:\n{question}\n\n"
        f"Previous retrieval query:\n{current_query}\n\n"
        f"Critic feedback:\n{json.dumps(feedback, ensure_ascii=True)}"
    )


def cited_ids(answer: str) -> set[str]:
    return set(_CITATION_RE.findall(answer))


def normalize_answer_citations(answer: str, chunks: list[RetrievedChunk]) -> str:
    if answer.strip() == FALLBACK_ANSWER or not chunks:
        return answer
    normalized_lines = []
    for line in answer.splitlines():
        if not line.strip():
            normalized_lines.append(line)
            continue
        sentences = _split_sentences(line)
        if len(sentences) <= 1:
            normalized_lines.append(_ensure_sentence_cited(line, chunks))
        else:
            normalized_lines.append(
                " ".join(_ensure_sentence_cited(sentence, chunks) for sentence in sentences)
            )
    return "\n".join(normalized_lines)


def parse_critic_json(raw: str, *, answer: str, chunks: list[RetrievedChunk]) -> CriticResult:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Critic did not return a JSON object.")
    data = json.loads(raw[start : end + 1])
    result = CriticResult.model_validate(data)

    valid_ids = {chunk.citation_id for chunk in chunks}
    used_ids = cited_ids(answer)
    invalid = sorted(used_ids - valid_ids)
    missing = list(result.missing_claims)

    if answer.strip() == FALLBACK_ANSWER:
        result.accepted = False
        if not result.reason:
            result.reason = "The generator returned the insufficient-information fallback."
    elif not used_ids:
        result.accepted = False
        missing.append("The answer contains no inline citations.")

    uncited = _uncited_claims(answer)
    if uncited:
        result.accepted = False
        missing.extend(f"Uncited sentence: {sentence}" for sentence in uncited)

    unsupported = _unsupported_cited_claims(answer, chunks)
    if unsupported:
        result.accepted = False
        missing.extend(f"Unsupported cited sentence: {sentence}" for sentence in unsupported)

    if invalid:
        result.accepted = False
        result.invalid_citations = sorted(set(result.invalid_citations + invalid))
    if missing:
        result.missing_claims = sorted(set(missing))
    if result.accepted and result.invalid_citations:
        result.accepted = False
    if result.accepted and result.missing_claims:
        result.accepted = False
    if not result.reason:
        result.reason = "Accepted by critic." if result.accepted else "Rejected by critic."
    return result


def _split_sentences(answer: str) -> list[str]:
    return [sentence.strip() for sentence in _SENTENCE_RE.split(answer.strip()) if sentence.strip()]


def _ensure_sentence_cited(sentence: str, chunks: list[RetrievedChunk]) -> str:
    desired_ids = _citation_ids_for_text(sentence, chunks)
    existing_ids = cited_ids(sentence)
    missing_ids = [citation_id for citation_id in desired_ids if citation_id not in existing_ids]
    if existing_ids and not missing_ids:
        return sentence
    citation = " ".join(f"[{citation_id}]" for citation_id in (missing_ids or desired_ids))
    stripped = sentence.rstrip()
    if stripped.endswith((".", "!", "?")):
        return f"{stripped[:-1]} {citation}{stripped[-1]}"
    return f"{stripped} {citation}"


def _uncited_claims(answer: str) -> list[str]:
    if answer.strip() == FALLBACK_ANSWER:
        return []
    uncited: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped or not re.search(r"[A-Za-z]", stripped):
            continue
        if stripped.startswith(("-", "*")) and not _CITATION_RE.search(stripped):
            uncited.append(stripped)
            continue
        uncited.extend(sentence for sentence in _split_sentences(stripped) if not _CITATION_RE.search(sentence))
    return uncited


def _unsupported_cited_claims(answer: str, chunks: list[RetrievedChunk]) -> list[str]:
    chunk_by_id = {chunk.citation_id: chunk for chunk in chunks}
    unsupported: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped or stripped == FALLBACK_ANSWER:
            continue
        for sentence in _split_sentences(stripped):
            citation_ids = cited_ids(sentence)
            if not citation_ids:
                continue
            cited_chunks = [chunk_by_id[citation_id] for citation_id in citation_ids if citation_id in chunk_by_id]
            if not cited_chunks:
                continue
            if not _sentence_supported_by_chunks(sentence, cited_chunks):
                unsupported.append(sentence)
    return unsupported


def _sentence_supported_by_chunks(sentence: str, chunks: list[RetrievedChunk]) -> bool:
    claim_tokens = _tokens(re.sub(_CITATION_RE, "", sentence))
    if len(claim_tokens) <= 2:
        return True
    evidence_tokens: set[str] = set()
    source_tokens: set[str] = set()
    for chunk in chunks:
        evidence_tokens |= _tokens(chunk.content)
        source_tokens |= _tokens(chunk.source) | _tokens(chunk.citation_label)
    overlap = claim_tokens & (evidence_tokens | source_tokens)
    if len(overlap) >= min(3, len(claim_tokens)):
        return True
    return (len(overlap) / len(claim_tokens)) >= 0.34


def _citation_ids_for_text(text: str, chunks: list[RetrievedChunk]) -> list[str]:
    mentioned = _source_mention_citations(text, chunks)
    if mentioned:
        return mentioned
    return [_best_citation_id(text, chunks)]


def _source_mention_citations(text: str, chunks: list[RetrievedChunk]) -> list[str]:
    normalized_text = _normalize_source_text(text)
    citations: list[str] = []
    for chunk in chunks:
        for name in _source_names(chunk):
            normalized_name = _normalize_source_text(name)
            if normalized_name and normalized_name in normalized_text and chunk.citation_id not in citations:
                citations.append(chunk.citation_id)
                break
    return citations


def _source_names(chunk: RetrievedChunk) -> set[str]:
    source = chunk.source.removeprefix("upload:")
    citation_label = re.sub(r"\s+p\.\d+$", "", chunk.citation_label)
    return {source, Path(source).name, citation_label, Path(citation_label).name}


def _normalize_source_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _best_citation_id(text: str, chunks: list[RetrievedChunk]) -> str:
    text_tokens = _tokens(text)
    if not text_tokens:
        return chunks[0].citation_id
    best_chunk = chunks[0]
    best_score = -1
    for chunk in chunks:
        source_tokens = _tokens(chunk.citation_label) | _tokens(chunk.source)
        content_tokens = _tokens(chunk.content)
        score = (len(text_tokens & source_tokens) * 4) + len(text_tokens & content_tokens)
        if score > best_score:
            best_score = score
            best_chunk = chunk
    return best_chunk.citation_id


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
    "also",
    "include",
    "includes",
    "including",
    "document",
    "documents",
}

from __future__ import annotations

import json
import re

from self_healing_rag.constants import FALLBACK_ANSWER
from self_healing_rag.schemas import CriticResult, RetrievedChunk


ANSWER_SYSTEM = f"""You are a grounded retrieval assistant.
Answer only with facts supported by the retrieved chunks.
Every factual sentence must include at least one inline citation like [C1].
Use only the citation IDs shown in the context.
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


def build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for chunk in chunks:
        location = chunk.citation_label
        blocks.append(f"[{chunk.citation_id}] Source: {location}\n{chunk.content}")
    return "\n\n".join(blocks)


def answer_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return f"Question:\n{question}\n\nRetrieved chunks:\n{build_context(chunks)}"


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


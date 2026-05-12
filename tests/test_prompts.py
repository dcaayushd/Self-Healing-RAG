import pytest

from self_healing_rag.prompts import answer_user_prompt, normalize_answer_citations, parse_critic_json
from self_healing_rag.schemas import RetrievedChunk


def chunk() -> RetrievedChunk:
    return RetrievedChunk(
        id="1",
        content="The document describes a grounded RAG workflow.",
        source="doc.md",
        source_type="markdown",
        citation_label="doc.md",
        chunk_index=0,
        citation_id="C1",
    )


def chunk_two() -> RetrievedChunk:
    return RetrievedChunk(
        id="2",
        content="Hantavirus infection can cause respiratory disease.",
        source="hantavirus.pdf",
        source_type="pdf",
        citation_label="hantavirus.pdf",
        chunk_index=0,
        citation_id="C2",
    )


def chunk_three() -> RetrievedChunk:
    return RetrievedChunk(
        id="3",
        content="A second disease PDF also discusses hantavirus.",
        source="upload:disease.pdf",
        source_type="pdf",
        citation_label="disease.pdf p.1",
        chunk_index=0,
        citation_id="C3",
    )


def test_normalize_answer_citations_adds_missing_sentence_citation():
    answer = "The document is about grounded RAG [C1]. It uses a critic."

    normalized = normalize_answer_citations(answer, [chunk()])

    assert normalized == "The document is about grounded RAG [C1]. It uses a critic [C1]."


def test_parse_critic_rejects_uncited_sentence():
    answer = "The document is about grounded RAG [C1]. It uses a critic."

    result = parse_critic_json(
        '{"accepted": true, "reason": "ok", "missing_claims": [], "invalid_citations": []}',
        answer=answer,
        chunks=[chunk()],
    )

    assert not result.accepted
    assert result.missing_claims


def test_parse_critic_rejects_unsupported_cited_sentence():
    result = parse_critic_json(
        '{"accepted": true, "reason": "ok", "missing_claims": [], "invalid_citations": []}',
        answer="The document recommends Kubernetes autoscaling [C1].",
        chunks=[chunk()],
    )

    assert not result.accepted
    assert any("Unsupported cited sentence" in claim for claim in result.missing_claims)


def test_parse_critic_rejects_missing_required_surface_fact():
    result = parse_critic_json(
        '{"accepted": true, "reason": "ok", "missing_claims": [], "invalid_citations": []}',
        answer="The document describes a grounded RAG workflow in 2026 [C1].",
        chunks=[chunk()],
    )

    assert not result.accepted
    assert any("Unsupported cited sentence" in claim for claim in result.missing_claims)


def test_normalize_answer_citations_uses_best_matching_chunk_for_bullets():
    answer = "Topics include:\n\n* Hantavirus infection symptoms"

    normalized = normalize_answer_citations(answer, [chunk(), chunk_two()])

    assert "Topics include:" in normalized
    assert "* Hantavirus infection symptoms [C2]" in normalized


def test_normalize_answer_citations_cites_named_sources():
    answer = "Sources include hantavirus.pdf and disease.pdf."

    normalized = normalize_answer_citations(answer, [chunk_two(), chunk_three()])

    assert "hantavirus.pdf and disease.pdf [C2] [C3]." in normalized


def test_answer_prompt_includes_overview_hint():
    prompt = answer_user_prompt("What is this document about?", [chunk()], overview=True)

    assert "broad overview question" in prompt
    assert "Retrieved chunks" in prompt

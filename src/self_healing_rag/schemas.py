from typing import Literal

from pydantic import BaseModel, Field, model_validator

from self_healing_rag.constants import FALLBACK_ANSWER


class Citation(BaseModel):
    id: str
    source: str
    source_type: str
    citation_label: str
    page: int | None = None
    chunk_index: int
    score: float | None = None
    relevance: float | None = None
    excerpt: str


class AttemptTrace(BaseModel):
    attempt: int
    query: str
    retrieved_count: int
    answer: str
    critic_accepted: bool
    critic_reason: str
    missing_claims: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    retrieval_strategy: str = "vector"
    retrieval_confidence: float | None = None
    retrieval_ms: float | None = None
    generation_ms: float | None = None
    critique_ms: float | None = None
    total_ms: float | None = None


class CriticResult(BaseModel):
    accepted: bool
    reason: str
    missing_claims: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)


class IngestRequest(BaseModel):
    source: str | None = None
    sources: list[str] = Field(default_factory=list)
    collection: str = "default"

    @model_validator(mode="after")
    def require_source(self) -> "IngestRequest":
        if self.source:
            self.sources.insert(0, self.source)
            self.source = None
        self.sources = [source for source in self.sources if source]
        if not self.sources:
            raise ValueError("At least one source is required.")
        return self


class IngestResponse(BaseModel):
    collection: str
    chunks_added: int
    sources: list[str]
    ids: list[str]


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    collection: str = "default"
    max_attempts: int = Field(default=3, ge=1)
    thread_id: str | None = None
    focus_sources: list[str] = Field(default_factory=list)


class AskResponse(BaseModel):
    status: Literal["answered", "insufficient_info"]
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    attempts: list[AttemptTrace] = Field(default_factory=list)
    thread_id: str
    focus_sources: list[str] = Field(default_factory=list)
    total_ms: float | None = None


class HealthResponse(BaseModel):
    ok: bool
    chroma_path: str
    checkpoint_db: str
    default_collection: str
    chat_model: str
    embedding_model: str


class IndexStats(BaseModel):
    collection: str
    chunk_count: int
    source_count: int
    sources: list[str] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)
    embedding_model: str | None = None
    is_empty: bool = True


class RetrievedChunk(BaseModel):
    id: str
    content: str
    source: str
    source_type: str
    citation_label: str
    page: int | None = None
    chunk_index: int
    score: float | None = None
    relevance: float | None = None
    retrieval_rank: int | None = None
    citation_id: str

    def to_citation(self) -> Citation:
        return Citation(
            id=self.citation_id,
            source=self.source,
            source_type=self.source_type,
            citation_label=self.citation_label,
            page=self.page,
            chunk_index=self.chunk_index,
            score=self.score,
            relevance=self.relevance,
            excerpt=self.content[:500],
        )


def insufficient_response(thread_id: str, attempts: list[AttemptTrace] | None = None) -> AskResponse:
    return AskResponse(
        status="insufficient_info",
        answer=FALLBACK_ANSWER,
        citations=[],
        attempts=attempts or [],
        thread_id=thread_id,
    )

from self_healing_rag.config import Settings
from self_healing_rag.constants import FALLBACK_ANSWER
from self_healing_rag.graph import SelfHealingRag, _retrieval_confidence, is_overview_question, sanitize_retrieval_query
from self_healing_rag.schemas import CriticResult, RetrievedChunk


def chunk() -> RetrievedChunk:
    return RetrievedChunk(
        id="1",
        content="The system retries after a critic rejects an ungrounded answer.",
        source="doc.md",
        source_type="markdown",
        citation_label="doc.md",
        chunk_index=0,
        citation_id="C1",
    )


class FakeComponents:
    def __init__(self, verdicts, chunks=None, raises_critic=False):
        self.verdicts = list(verdicts)
        self.chunks = chunks if chunks is not None else [chunk()]
        self.queries = []
        self.raises_critic = raises_critic
        self.critic_calls = 0

    def retrieve(self, query, *, collection, top_k, fetch_k, focus_sources=None):
        self.queries.append(query)
        return self.chunks

    def generate_answer(self, question, chunks):
        if not chunks:
            return FALLBACK_ANSWER
        return "The system retries after critic rejection [C1]."

    def critique(self, question, chunks, answer):
        self.critic_calls += 1
        if self.raises_critic:
            raise ValueError("bad json")
        accepted = self.verdicts.pop(0) if self.verdicts else False
        return CriticResult(accepted=accepted, reason="ok" if accepted else "not grounded")

    def reformulate(self, question, current_query, critic):
        return f"{question} retry"


class FailingRetrieveComponents(FakeComponents):
    def __init__(self):
        super().__init__([])

    def retrieve(self, query, *, collection, top_k, fetch_k, focus_sources=None):
        raise ConnectionError("Failed to connect to Ollama")


class UnsupportedAnswerComponents(FakeComponents):
    def generate_answer(self, question, chunks):
        return "The system uses Kubernetes autoscaling [C1]."


def settings(tmp_path, **kwargs):
    return Settings(chroma_path=tmp_path / "chroma", checkpoint_db=tmp_path / "checkpoints.sqlite", **kwargs)


def test_graph_accepts_first_answer(tmp_path):
    fake = FakeComponents([True])
    engine = SelfHealingRag(settings(tmp_path), components=fake, use_checkpointer=False)

    response = engine.ask("How does it retry?", max_attempts=3, thread_id="t1")

    assert response.status == "answered"
    assert fake.critic_calls == 0
    assert response.citations[0].id == "C1"
    assert len(response.attempts) == 1
    assert response.attempts[0].retrieval_strategy == "hybrid"
    assert response.attempts[0].retrieval_confidence == 1.0
    assert response.total_ms is not None
    assert response.attempts[0].retrieval_ms is not None
    assert response.attempts[0].generation_ms is not None
    assert response.attempts[0].critique_ms is not None
    assert response.attempts[0].total_ms is not None


def test_graph_retries_then_accepts(tmp_path):
    fake = FakeComponents([False, True])
    engine = SelfHealingRag(
        settings(tmp_path, critic_mode="llm", use_llm_rewrite=True),
        components=fake,
        use_checkpointer=False,
    )

    response = engine.ask("How does it retry?", max_attempts=3, thread_id="t1")

    assert response.status == "answered"
    assert len(response.attempts) == 2
    assert fake.queries == ["How does it retry?", "How does it retry? retry"]


def test_graph_exhausts_retries_to_insufficient_info(tmp_path):
    engine = SelfHealingRag(
        settings(tmp_path, critic_mode="llm", use_llm_rewrite=True),
        components=FakeComponents([False, False]),
        use_checkpointer=False,
    )

    response = engine.ask("How does it retry?", max_attempts=2, thread_id="t1")

    assert response.status == "insufficient_info"
    assert response.answer == FALLBACK_ANSWER
    assert len(response.attempts) == 2


def test_graph_handles_no_chunks(tmp_path):
    engine = SelfHealingRag(settings(tmp_path), components=FakeComponents([False], chunks=[]), use_checkpointer=False)

    response = engine.ask("Unknown?", max_attempts=1, thread_id="t1")

    assert response.status == "insufficient_info"
    assert response.attempts[0].retrieved_count == 0


def test_graph_rejects_malformed_critic_output(tmp_path):
    engine = SelfHealingRag(
        settings(tmp_path, critic_mode="llm"),
        components=FakeComponents([], raises_critic=True),
        use_checkpointer=False,
    )

    response = engine.ask("How does it retry?", max_attempts=1, thread_id="t1")

    assert response.status == "insufficient_info"
    assert "malformed" in response.attempts[0].critic_reason


def test_graph_applies_local_grounding_validation_after_critic_accepts(tmp_path):
    engine = SelfHealingRag(
        settings(tmp_path),
        components=UnsupportedAnswerComponents([True]),
        use_checkpointer=False,
    )

    response = engine.ask("How does it retry?", max_attempts=1, thread_id="t1")

    assert response.status == "insufficient_info"
    assert response.attempts[0].missing_claims
    assert "Unsupported cited sentence" in response.attempts[0].missing_claims[0]


def test_graph_stops_on_retrieval_runtime_error(tmp_path):
    engine = SelfHealingRag(settings(tmp_path), components=FailingRetrieveComponents(), use_checkpointer=False)

    response = engine.ask("How does it retry?", max_attempts=3, thread_id="t1")

    assert response.status == "insufficient_info"
    assert len(response.attempts) == 1
    assert "Retrieval failed" in response.attempts[0].critic_reason


def test_overview_question_detection():
    assert is_overview_question("What is this document about?")
    assert is_overview_question("What's this PDF about?")
    assert is_overview_question("Tell me about this file")
    assert is_overview_question("summarize this document")
    assert is_overview_question("What are the key facts?")
    assert is_overview_question("key aspects")
    assert is_overview_question("What should I know first?")
    assert not is_overview_question("What does the critic do after a rejection?")


def test_sanitize_retrieval_query_strips_prefix_and_truncates():
    query = sanitize_retrieval_query("Rewritten query: " + ("hantavirus transmission " * 30), max_length=80)

    assert not query.lower().startswith("rewritten query")
    assert len(query) <= 80


def test_retrieval_confidence_calibrates_supported_evidence_above_raw_scores():
    chunks = [
        RetrievedChunk(
            id="1",
            content="Hantavirus transmission occurs through infected rodent excreta.",
            source="hantavirus.pdf",
            source_type="pdf",
            citation_label="hantavirus.pdf",
            chunk_index=0,
            citation_id="C1",
            relevance=0.38,
        ),
        RetrievedChunk(
            id="2",
            content="Prevention guidance discusses rodent control and transmission risk.",
            source="guidance.pdf",
            source_type="pdf",
            citation_label="guidance.pdf",
            chunk_index=1,
            citation_id="C2",
            relevance=0.31,
        ),
    ]

    confidence = _retrieval_confidence("hantavirus transmission rodent", chunks)

    assert confidence >= 0.72


def test_retrieval_confidence_keeps_uncovered_queries_low():
    chunks = [
        RetrievedChunk(
            id="1",
            content="The document discusses project setup instructions.",
            source="setup.md",
            source_type="markdown",
            citation_label="setup.md",
            chunk_index=0,
            citation_id="C1",
            relevance=0.30,
        )
    ]

    confidence = _retrieval_confidence("hantavirus transmission rodent", chunks)

    assert confidence <= 0.30

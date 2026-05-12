from self_healing_rag.streamlit_app import (
    _attempt_summary,
    _load_persisted_chat_sessions,
    _ms_label,
    _new_chat_collection,
    _page_label,
    _percent_label,
    _pluralize,
    _score_label,
    _session_title,
    _source_lines,
    _workspace_label,
)


def test_source_lines_ignores_blank_lines():
    assert _source_lines("\n./docs\n\nhttps://example.com\n") == ["./docs", "https://example.com"]


def test_page_label_is_human_one_indexed():
    assert _page_label(None) == ""
    assert _page_label(0) == "1"


def test_attempt_summary_includes_verdict_and_query():
    summary = _attempt_summary(
        {
            "attempt": 2,
            "critic_accepted": False,
            "retrieved_count": 3,
            "query": "retry query",
            "critic_reason": "Missing support.",
        }
    )

    assert "Attempt 2" in summary
    assert "rejected" in summary
    assert "retry query" in summary


def test_score_label_formats_optional_score():
    assert _score_label(None) == ""
    assert _score_label(0.123456) == "0.1235"


def test_percent_label_formats_optional_confidence():
    assert _percent_label(None) == "confidence n/a"
    assert _percent_label(0.874) == "87% confidence"


def test_ms_label_formats_optional_latency():
    assert _ms_label(None) == "latency n/a"
    assert _ms_label(12.6) == "13 ms"


def test_pluralize_formats_file_counts():
    assert _pluralize(1, "selected file") == "1 selected file"
    assert _pluralize(2, "selected file") == "2 selected files"


def test_new_chat_collection_is_ui_scoped():
    collection = _new_chat_collection()

    assert collection.startswith("ui_")
    assert len(collection) > 10
    assert " " not in collection
    assert "Chat workspace" in _workspace_label(collection)


def test_session_title_uses_first_user_message():
    title = _session_title(
        [
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "What is this document about and why does it matter?"},
        ]
    )

    assert title.startswith("What is this document")


def test_load_persisted_chat_sessions_ignores_invalid_payload(tmp_path):
    path = tmp_path / "logs.json"
    path.write_text('{"sessions": [], "order": "bad"}', encoding="utf-8")

    sessions, order = _load_persisted_chat_sessions(path)

    assert sessions == {}
    assert order == []

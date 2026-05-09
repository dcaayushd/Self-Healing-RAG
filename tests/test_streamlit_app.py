from self_healing_rag.streamlit_app import _attempt_summary, _page_label, _source_lines


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

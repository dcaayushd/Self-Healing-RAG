from self_healing_rag.source_policy import display_source_name, preferred_default_sources, source_filter_where


def test_preferred_default_sources_excludes_system_sample_when_possible():
    sources = ["/tmp/self_healing_rag.md", "upload:paper.pdf"]

    assert preferred_default_sources(sources) == ["upload:paper.pdf"]


def test_display_source_name_strips_upload_prefix_and_paths():
    assert display_source_name("upload:/tmp/report.pdf") == "report.pdf"


def test_source_filter_where_builds_chroma_filters():
    assert source_filter_where([]) is None
    assert source_filter_where(["a"]) == {"source": "a"}
    assert source_filter_where(["a", "b"]) == {"source": {"$in": ["a", "b"]}}

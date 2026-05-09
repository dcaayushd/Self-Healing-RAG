from typer.testing import CliRunner

from self_healing_rag import cli
from self_healing_rag.constants import FALLBACK_ANSWER
from self_healing_rag.diagnostics import DiagnosticResult
from self_healing_rag.schemas import AskResponse, IngestResponse


class FakeService:
    def ingest_sources(self, sources, *, collection=None):
        return IngestResponse(collection=collection or "default", chunks_added=1, sources=sources, ids=["1"])

    def ask(self, question, *, collection=None, max_attempts=None, thread_id=None):
        return AskResponse(status="insufficient_info", answer=FALLBACK_ANSWER, citations=[], attempts=[], thread_id="t1")

    def reset_collection(self, collection=None):
        self.reset = collection


def test_doctor_success(monkeypatch):
    monkeypatch.setattr(cli, "run_diagnostics", lambda settings: DiagnosticResult(True, [("python", True, "3")]))
    result = CliRunner().invoke(cli.app, ["doctor"])
    assert result.exit_code == 0
    assert "ok" in result.output


def test_ingest_command(monkeypatch):
    monkeypatch.setattr(cli, "build_service", lambda: FakeService())
    result = CliRunner().invoke(cli.app, ["ingest", "doc.md", "--collection", "docs"])
    assert result.exit_code == 0
    assert '"chunks_added": 1' in result.output


def test_ask_command(monkeypatch):
    monkeypatch.setattr(cli, "build_service", lambda: FakeService())
    result = CliRunner().invoke(cli.app, ["ask", "What?", "--collection", "docs"])
    assert result.exit_code == 0
    assert FALLBACK_ANSWER in result.output


def test_reset_command(monkeypatch):
    monkeypatch.setattr(cli, "build_service", lambda: FakeService())
    result = CliRunner().invoke(cli.app, ["reset", "--collection", "docs"])
    assert result.exit_code == 0
    assert '{"reset": "docs"}' in result.output


def test_ui_command(monkeypatch):
    calls = []

    class Result:
        returncode = 0

    def fake_run(command):
        calls.append(command)
        return Result()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "_available_port", lambda host, port: port)
    result = CliRunner().invoke(cli.app, ["ui", "--host", "0.0.0.0", "--port", "8502"])

    assert result.exit_code == 0
    assert calls
    assert calls[0][1:4] == ["-m", "streamlit", "run"]
    assert "8502" in calls[0]


def test_ingest_command_reports_clean_error(monkeypatch):
    class FailingService:
        def ingest_sources(self, sources, *, collection=None):
            raise ValueError("No supported documents found")

    monkeypatch.setattr(cli, "build_service", lambda: FailingService())
    result = CliRunner().invoke(cli.app, ["ingest", "empty-dir"])

    assert result.exit_code == 1
    assert "Error: No supported documents found" in result.stderr

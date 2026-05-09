import requests

from self_healing_rag.config import Settings
from self_healing_rag.diagnostics import _model_available, run_diagnostics


def test_model_available_matches_implicit_latest():
    assert _model_available("nomic-embed-text", ["nomic-embed-text:latest"])
    assert _model_available("llama3:latest", ["llama3:latest"])
    assert not _model_available("llama3.1:8b", ["llama3:latest"])


def test_diagnostics_reports_missing_model(monkeypatch, tmp_path):
    class Response:
        ok = True
        status_code = 200
        reason = "OK"

        def json(self):
            return {"models": [{"name": "nomic-embed-text:latest"}]}

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    settings = Settings(
        chroma_path=tmp_path / "chroma",
        checkpoint_db=tmp_path / "checkpoints.sqlite",
        chat_model="missing-model",
    )

    result = run_diagnostics(settings)

    assert not result.ok
    assert any(check[0] == "chat_model" and not check[1] for check in result.checks)

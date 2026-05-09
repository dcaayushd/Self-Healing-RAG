from __future__ import annotations

import sys
from dataclasses import dataclass

import requests

from self_healing_rag.config import Settings


@dataclass
class DiagnosticResult:
    ok: bool
    checks: list[tuple[str, bool, str]]


def run_diagnostics(settings: Settings) -> DiagnosticResult:
    checks: list[tuple[str, bool, str]] = []
    ollama_models: list[str] = []
    ollama_available = False

    checks.append(("python", True, sys.version.split()[0]))

    try:
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        checks.append(("chroma_path", True, str(settings.chroma_path)))
    except Exception as exc:
        checks.append(("chroma_path", False, str(exc)))

    try:
        settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        checks.append(("checkpoint_db", True, str(settings.checkpoint_db)))
    except Exception as exc:
        checks.append(("checkpoint_db", False, str(exc)))

    try:
        response = requests.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=2)
        ollama_available = response.ok
        if response.ok:
            payload = response.json()
            ollama_models = [str(model.get("name", "")) for model in payload.get("models", [])]
        checks.append(("ollama_server", response.ok, f"{response.status_code} {response.reason}"))
    except Exception as exc:
        checks.append(("ollama_server", False, str(exc)))

    if ollama_available:
        checks.append(_model_check("chat_model", settings.chat_model, ollama_models))
        checks.append(_model_check("embedding_model", settings.embedding_model, ollama_models))
    else:
        checks.append(("chat_model", False, "Ollama server unavailable; cannot verify model."))
        checks.append(("embedding_model", False, "Ollama server unavailable; cannot verify model."))

    ok = all(item[1] for item in checks)
    return DiagnosticResult(ok=ok, checks=checks)


def _model_check(label: str, model: str, available_models: list[str]) -> tuple[str, bool, str]:
    if _model_available(model, available_models):
        return (label, True, model)
    available = ", ".join(available_models) or "none"
    return (label, False, f"Missing `{model}`. Run `ollama pull {model}`. Available: {available}")


def _model_available(model: str, available_models: list[str]) -> bool:
    normalized_target = _normalize_model_name(model)
    return any(_normalize_model_name(candidate) == normalized_target for candidate in available_models)


def _normalize_model_name(model: str) -> str:
    return model if ":" in model else f"{model}:latest"

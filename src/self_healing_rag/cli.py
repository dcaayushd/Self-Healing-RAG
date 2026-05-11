from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

import typer
import uvicorn

from self_healing_rag.config import get_settings
from self_healing_rag.diagnostics import run_diagnostics

app = typer.Typer(help="Self-Healing RAG CLI")


def build_service():
    from self_healing_rag.service import build_service as _build_service

    return _build_service()


def _fail(message: str) -> None:
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _open_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((_open_host(host), port)) == 0


def _available_port(host: str, preferred_port: int) -> int:
    port = preferred_port
    while _port_in_use(host, port):
        port += 1
    if port != preferred_port:
        typer.secho(f"Port {preferred_port} is in use; using {port} instead.", fg=typer.colors.YELLOW)
    return port


@app.command()
def doctor() -> None:
    """Check local paths and Ollama connectivity."""
    settings = get_settings()
    result = run_diagnostics(settings)
    for name, ok, detail in result.checks:
        marker = "ok" if ok else "fail"
        typer.echo(f"{marker:4} {name}: {detail}")
    if not result.ok:
        raise typer.Exit(code=1)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the FastAPI server."""
    port = _available_port(host, port)
    uvicorn.run("self_healing_rag.api:app", host=host, port=port, reload=reload)


@app.command()
def ui(host: str = "127.0.0.1", port: int = 8501) -> None:
    """Run the Streamlit frontend."""
    port = _available_port(host, port)
    app_path = Path(__file__).with_name("streamlit_app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]
    raise typer.Exit(subprocess.run(command).returncode)


@app.command()
def ingest(
    source: str = typer.Argument(..., help="Local file, local directory, or exact URL to ingest."),
    collection: str = typer.Option(None, "--collection", "-c", help="Chroma collection name."),
) -> None:
    """Ingest a local source or exact single-page URL."""
    settings = get_settings()
    try:
        result = build_service().ingest_sources([source], collection=collection or settings.default_collection)
    except Exception as exc:
        _fail(str(exc))
    else:
        typer.echo(result.model_dump_json(indent=2))


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer from the indexed documents."),
    collection: str = typer.Option(None, "--collection", "-c", help="Chroma collection name."),
    max_attempts: int = typer.Option(None, "--max-attempts", "-m", help="Maximum retrieve/generate/critique attempts."),
    source: list[str] = typer.Option(None, "--source", "-s", help="Limit retrieval to one or more exact source IDs."),
    json_output: bool = typer.Option(False, "--json", help="Emit the full JSON response."),
) -> None:
    """Ask a question through the self-healing LangGraph workflow."""
    settings = get_settings()
    try:
        response = build_service().ask(
            question,
            collection=collection or settings.default_collection,
            max_attempts=max_attempts or settings.max_attempts,
            focus_sources=source or [],
        )
    except Exception as exc:
        _fail(str(exc))
        return
    if json_output:
        typer.echo(response.model_dump_json(indent=2))
        return
    typer.echo(response.answer)
    if response.citations:
        typer.echo("")
        typer.echo("Citations:")
        for citation in response.citations:
            page = f" page {citation.page + 1}" if citation.page is not None else ""
            typer.echo(f"- [{citation.id}] {citation.citation_label}{page}")


@app.command()
def reset(collection: str = typer.Option(None, "--collection", "-c", help="Chroma collection name.")) -> None:
    """Delete a Chroma collection."""
    settings = get_settings()
    target = collection or settings.default_collection
    try:
        build_service().reset_collection(target)
    except Exception as exc:
        _fail(str(exc))
    typer.echo(json.dumps({"reset": target}))


@app.command()
def delete_source(
    source: list[str] = typer.Option(..., "--source", "-s", help="Exact source ID to delete from the collection."),
    collection: str = typer.Option(None, "--collection", "-c", help="Chroma collection name."),
) -> None:
    """Delete one or more indexed sources from a collection."""
    settings = get_settings()
    target = collection or settings.default_collection
    try:
        deleted = build_service().delete_sources(source, collection=target)
    except Exception as exc:
        _fail(str(exc))
    typer.echo(json.dumps({"collection": target, "deleted_chunks": deleted, "sources": source}))


if __name__ == "__main__":
    app()

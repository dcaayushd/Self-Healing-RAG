from __future__ import annotations

from fastapi import Depends, FastAPI, File, Form, UploadFile

from self_healing_rag.config import get_settings
from self_healing_rag.schemas import AskRequest, AskResponse, HealthResponse, IndexStats, IngestRequest, IngestResponse
from self_healing_rag.service import RagService, build_service

app = FastAPI(title="Self-Healing RAG", version="0.1.0")


def get_rag_service() -> RagService:
    return build_service()


@app.get("/health", response_model=HealthResponse)
def health(service: RagService = Depends(get_rag_service)) -> HealthResponse:
    return service.health()


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest, service: RagService = Depends(get_rag_service)) -> IngestResponse:
    collection = request.collection or get_settings().default_collection
    return service.ingest_sources(request.sources, collection=collection)


@app.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    file: UploadFile = File(...),
    collection: str | None = Form(default=None),
    service: RagService = Depends(get_rag_service),
) -> IngestResponse:
    content = await file.read()
    return service.ingest_upload(file.filename or "upload", content, collection=collection)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, service: RagService = Depends(get_rag_service)) -> AskResponse:
    return service.ask(
        request.question,
        collection=request.collection,
        max_attempts=request.max_attempts,
        thread_id=request.thread_id,
        focus_sources=request.focus_sources,
    )


@app.get("/collections/{collection}/stats", response_model=IndexStats)
def collection_stats(collection: str, service: RagService = Depends(get_rag_service)) -> IndexStats:
    return service.collection_stats(collection)

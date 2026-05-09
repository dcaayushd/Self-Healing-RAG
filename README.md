# Self-Healing RAG

A local Retrieval-Augmented Generation pipeline that retrieves evidence, generates an answer, critiques whether the answer is grounded in the retrieved chunks, and retries with a reformulated query before returning a safe fallback.

The graph is modeled with LangGraph as a cyclical workflow:

```text
retrieve -> generate -> critique -> reformulate -> retrieve
                              \-> finalize
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Start Ollama and pull the default models:

```bash
ollama serve
ollama pull llama3:latest
ollama pull nomic-embed-text
```

In another terminal:

```bash
rag doctor
```

## CLI

Put documents in `data/docs` first, or use the included sample document:

```bash
rag ingest ./data/docs --collection default
rag ask "What does the document say about retries?" --collection default
rag reset --collection default
```

You can also ingest one exact URL:

```bash
rag ingest https://example.com/single-page --collection default
```

Run the API or UI:

```bash
rag serve --host 127.0.0.1 --port 8000
rag ui --host 127.0.0.1 --port 8501
```

## Streamlit UI

Run the local frontend:

```bash
rag ui
```

Then open `http://127.0.0.1:8501`.

The sidebar supports PDF/TXT/Markdown/HTML uploads, server-local paths, exact single-page URLs, collection reset, and retry limit control. The chat panel shows the final answer, citations, and critic attempt history.

## API

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/ingest \
  -H "content-type: application/json" \
  -d '{"sources":["./data/docs"],"collection":"default"}'

curl -X POST http://127.0.0.1:8000/ask \
  -H "content-type: application/json" \
  -d '{"question":"What does the document say about retries?","collection":"default","max_attempts":3}'
```

Upload ingestion supports PDF, TXT, Markdown, and HTML:

```bash
curl -X POST http://127.0.0.1:8000/ingest/upload \
  -F "collection=default" \
  -F "file=@paper.pdf"
```

## Configuration

Settings are read from environment variables prefixed with `RAG_`. See `.env.example`.

Defaults:

- Chroma path: `data/chroma`
- LangGraph checkpoint DB: `data/checkpoints.sqlite`
- Chat model: `llama3:latest`
- Embedding model: `nomic-embed-text`
- Retrieval: `top_k=6`, `fetch_k=20`
- Chunking: `chunk_size=1000`, `chunk_overlap=150`
- Retry limit: `max_attempts=3`

URL ingestion fetches only the exact URLs supplied. It blocks `file://`, localhost, and private-network targets unless `RAG_ALLOW_PRIVATE_URLS=true`.

## Tests

```bash
pytest
```

Optional local Ollama checks can be added behind `RUN_OLLAMA_TESTS=1`; the default test suite uses fakes and does not require Ollama to be running.

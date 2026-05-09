# Self-Healing RAG

This document describes a Retrieval-Augmented Generation pipeline that retrieves relevant chunks, generates an answer, critiques whether the answer is grounded in the retrieved evidence, and retries with a reformulated query when the critic rejects the answer.

The pipeline uses LangGraph to model the process as a cyclical stateful workflow instead of a simple linear chain. A normal question flows through retrieval, generation, critique, optional query reformulation, and finalization.

If the system cannot produce an answer that is supported by the retrieved chunks, it should respond with: I don't have enough information in the provided documents to answer that.

The local implementation exposes a CLI, FastAPI API, and Streamlit UI. It uses Chroma for vector storage, Ollama for local chat and embeddings, and a SQLite LangGraph checkpointer for thread history.

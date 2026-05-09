from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    chroma_path: Path = Path("data/chroma")
    checkpoint_db: Path = Path("data/checkpoints.sqlite")
    default_collection: str = "default"
    chat_model: str = "llama3:latest"
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    top_k: int = Field(default=6, ge=1)
    fetch_k: int = Field(default=20, ge=1)
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)
    max_attempts: int = Field(default=3, ge=1)
    allow_private_urls: bool = False
    url_timeout_seconds: float = Field(default=20.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()

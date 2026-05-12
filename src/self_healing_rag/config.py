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
    min_retrieval_confidence: float = Field(default=0.08, ge=0.0, le=1.0)
    max_lexical_scan: int = Field(default=5000, ge=100)
    retrieval_cache_size: int = Field(default=128, ge=0)
    stats_cache_size: int = Field(default=32, ge=0)
    answer_cache_size: int = Field(default=64, ge=0)
    llm_num_ctx: int = Field(default=8192, ge=1024)
    answer_num_predict: int = Field(default=512, ge=64)
    critic_num_predict: int = Field(default=256, ge=64)
    rewrite_num_predict: int = Field(default=96, ge=16)
    critic_mode: str = Field(default="local", pattern="^(local|hybrid|llm)$")
    use_llm_rewrite: bool = False
    ollama_keep_alive: str = "10m"
    allow_private_urls: bool = False
    url_timeout_seconds: float = Field(default=20.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()

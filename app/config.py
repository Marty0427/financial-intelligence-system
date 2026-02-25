from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # Financial data
    alpha_vantage_api_key: str = ""

    # News / Search
    news_api_key: str = ""
    tavily_api_key: str = ""

    # Vector DB
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "sec_filings"

    # Server
    log_level: str = "INFO"
    rate_limit_per_minute: int = 30
    host: str = "0.0.0.0"
    port: int = 8000


# Singleton — import this everywhere
settings = Settings()  # type: ignore[call-arg]

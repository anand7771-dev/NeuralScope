"""
app/config.py
─────────────
Application configuration via pydantic-settings.
All values can be overridden through environment variables or a .env file.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    """NeuralScope application settings.

    Every field may be overridden by a matching environment variable
    (case-insensitive) or by entries in an optional ``.env`` file placed
    beside this module.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    app_name: str = "NeuralScope"
    app_version: str = "0.1.0"

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cache_dir: str = "./model_cache"
    default_model: str = "gpt2"

    # ── Session limits ────────────────────────────────────────────────────────
    max_sessions: int = 10

    # ── Tensor store ──────────────────────────────────────────────────────────
    tensor_store_max_mb: int = 512

    # ── Instrumentation ───────────────────────────────────────────────────────
    # Allowed values: "lightweight" | "standard" | "deep"
    instrumentation_level: str = "standard"

    # ── Generation ────────────────────────────────────────────────────────────
    max_new_tokens: int = 200

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` singleton.

    Using ``@lru_cache`` ensures the ``.env`` file and environment variables
    are parsed exactly once during the application lifecycle.
    """
    return Settings()

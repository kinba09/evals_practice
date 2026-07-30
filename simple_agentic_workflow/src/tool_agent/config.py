"""Environment-backed application configuration."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "qwen3-30b-a3b-instruct-2507"
    openai_reasoning_effort: str | None = None
    openai_temperature: float = 0.0
    openai_timeout_seconds: float = 30.0
    openai_max_retries: int = 2


def get_settings() -> Settings:
    """Read runtime settings from this project's environment."""

    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_model=os.getenv("OPENAI_MODEL", "qwen3-30b-a3b-instruct-2507"),
        openai_reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT") or None,
        openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
        openai_timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")),
        openai_max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
    )

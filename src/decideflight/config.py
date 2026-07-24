"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./decideflight.db",
    )
    openweathermap_api_key: str = os.getenv("OPENWEATHERMAP_API_KEY", "")
    weatherapi_api_key: str = os.getenv("WEATHERAPI_API_KEY", "")
    debug: bool = _to_bool(os.getenv("DEBUG"), default=False)


settings = Settings()

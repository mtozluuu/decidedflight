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


def _float(env: str, default: float) -> float:
    raw = os.getenv(env)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./decideflight.db",
    )
    openweathermap_api_key: str = os.getenv("OPENWEATHERMAP_API_KEY", "")
    weatherapi_api_key: str = os.getenv("WEATHERAPI_API_KEY", "")
    windy_api_key: str = os.getenv("WINDY_API_KEY", "")
    debug: bool = _to_bool(os.getenv("DEBUG"), default=False)

    # --- Drone flight limits (configurable via .env) ---
    # Wind speed (knots)
    wind_ok_max_knots: float = _float("WIND_OK_MAX_KNOTS", 15.0)
    wind_risky_max_knots: float = _float("WIND_RISKY_MAX_KNOTS", 25.0)
    # Visibility (km)
    visibility_ok_min_km: float = _float("VISIBILITY_OK_MIN_KM", 5.0)
    visibility_risky_min_km: float = _float("VISIBILITY_RISKY_MIN_KM", 1.0)
    # Temperature (°C)
    temp_ok_min_c: float = _float("TEMP_OK_MIN_C", 0.0)
    temp_ok_max_c: float = _float("TEMP_OK_MAX_C", 40.0)
    temp_risky_min_c: float = _float("TEMP_RISKY_MIN_C", -5.0)
    temp_risky_max_c: float = _float("TEMP_RISKY_MAX_C", 45.0)
    # Humidity (%)
    humidity_ok_max_pct: float = _float("HUMIDITY_OK_MAX_PCT", 85.0)
    humidity_risky_max_pct: float = _float("HUMIDITY_RISKY_MAX_PCT", 95.0)
    # Cloud base (ft AGL)
    cloud_base_ok_min_ft: float = _float("CLOUD_BASE_OK_MIN_FT", 500.0)
    cloud_base_risky_min_ft: float = _float("CLOUD_BASE_RISKY_MIN_FT", 200.0)
    # Cloud ceiling (ft)
    cloud_ceiling_ok_min_ft: float = _float("CLOUD_CEILING_OK_MIN_FT", 1000.0)
    cloud_ceiling_risky_min_ft: float = _float("CLOUD_CEILING_RISKY_MIN_FT", 500.0)


settings = Settings()

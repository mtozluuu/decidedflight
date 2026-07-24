"""Weather fetcher service.

Retrieves current weather data from up to four sources and normalises them
into ``WeatherSourceData`` objects:

1. **Open-Meteo**  – always attempted (free, no key).
2. **OpenWeatherMap** – requires ``OPENWEATHERMAP_API_KEY``.
3. **WeatherAPI**    – requires ``WEATHERAPI_API_KEY``.
4. **Windy**         – requires ``WINDY_API_KEY``.

Sources without a configured API key are silently skipped.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import httpx

from decideflight.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TIMEOUT = 15.0

# Precipitation level constants
PRECIP_NONE = 0
PRECIP_LIGHT = 1
PRECIP_HEAVY = 2


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class WeatherSourceData:
    """Normalised weather observation from a single source."""

    source: str
    wind_speed_knots: float
    temperature_c: float
    humidity_pct: float
    visibility_km: float
    precipitation_level: int  # 0 = none, 1 = light, 2 = moderate/heavy
    cloud_base_ft: float | None
    cloud_ceiling_ft: float | None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


def _kmh_to_knots(kmh: float) -> float:
    return kmh * 0.539957


def _ms_to_knots(ms: float) -> float:
    return ms * 1.94384


def _kelvin_to_celsius(k: float) -> float:
    return k - 273.15


def _m_to_ft(m: float) -> float:
    return m * 3.28084


def _m_to_km(m: float) -> float:
    return m / 1000.0


def _dew_point_c(temp_c: float, rh_pct: float) -> float | None:
    """Return estimated dew point (°C) using the Magnus formula.

    Returns *None* when *rh_pct* is not a physically meaningful value
    (i.e. ≤ 0 or > 100), so that callers can skip cloud-base estimation
    rather than propagating a nonsensical result.
    """
    if rh_pct <= 0 or rh_pct > 100:
        return None
    a, b = 17.625, 243.04
    alpha = math.log(rh_pct / 100.0) + (a * temp_c) / (b + temp_c)
    return b * alpha / (a - alpha)


def _cloud_base_ft(temp_c: float, dew_point_c: float) -> float:
    """Estimate cloud base AGL in feet from temperature / dew-point spread."""
    spread = max(temp_c - dew_point_c, 0.0)
    # Standard approximation: ≈122.5 m per °C of spread, then convert to ft
    base_m = spread * 122.5
    return _m_to_ft(base_m)


def _precip_level(mm: float) -> int:
    if mm <= 0:
        return PRECIP_NONE
    if mm < 2.5:
        return PRECIP_LIGHT
    return PRECIP_HEAVY


# ---------------------------------------------------------------------------
# Open-Meteo
# ---------------------------------------------------------------------------


async def _fetch_open_meteo(
    lat: float,
    lon: float,
    client: httpx.AsyncClient,
) -> WeatherSourceData | None:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "dew_point_2m,"
            "wind_speed_10m,"
            "precipitation,"
            "visibility,"
            "cloud_cover"
        ),
        "wind_speed_unit": "kmh",
        "timezone": "UTC",
    }
    try:
        resp = await client.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Open-Meteo fetch failed: %s", exc)
        return None

    cur = data.get("current", {})
    temp = cur.get("temperature_2m")
    dew = cur.get("dew_point_2m")
    wind_kmh = cur.get("wind_speed_10m", 0.0)
    rh = cur.get("relative_humidity_2m", 0.0)
    precip_mm = cur.get("precipitation", 0.0)
    vis_m = cur.get("visibility", 10000.0)
    cloud_pct = cur.get("cloud_cover", 0.0)

    if temp is None:
        logger.warning("Open-Meteo returned no temperature")
        return None

    base_ft: float | None = None
    if dew is not None:
        base_ft = _cloud_base_ft(temp, dew)

    # Estimate cloud ceiling: if cloud cover > 50 % use base; else unlimited
    ceiling_ft: float | None = None
    if base_ft is not None and cloud_pct is not None and cloud_pct > 50:
        # Rough heuristic: ceiling slightly above base
        ceiling_ft = base_ft + 200.0

    return WeatherSourceData(
        source="Open-Meteo",
        wind_speed_knots=_kmh_to_knots(wind_kmh),
        temperature_c=temp,
        humidity_pct=rh,
        visibility_km=_m_to_km(vis_m),
        precipitation_level=_precip_level(precip_mm),
        cloud_base_ft=base_ft,
        cloud_ceiling_ft=ceiling_ft,
        raw=cur,
    )


# ---------------------------------------------------------------------------
# OpenWeatherMap
# ---------------------------------------------------------------------------


async def _fetch_owm(
    lat: float,
    lon: float,
    client: httpx.AsyncClient,
) -> WeatherSourceData | None:
    if not settings.openweathermap_api_key:
        return None

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": settings.openweathermap_api_key,
    }
    try:
        resp = await client.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("OpenWeatherMap fetch failed: %s", exc)
        return None

    main = data.get("main", {})
    wind = data.get("wind", {})
    temp_k = main.get("temp", 273.15)
    temp_c = _kelvin_to_celsius(temp_k)
    wind_ms = wind.get("speed", 0.0)
    rh = main.get("humidity", 0.0)
    vis_m = data.get("visibility", 10000)
    rain_mm = data.get("rain", {}).get("1h", 0.0)
    snow_mm = data.get("snow", {}).get("1h", 0.0)
    precip_mm = rain_mm + snow_mm
    cloud_pct = data.get("clouds", {}).get("all", 0.0)

    # Estimate dew point from Magnus formula
    dew_c = _dew_point_c(temp_c, rh)
    base_ft = _cloud_base_ft(temp_c, dew_c) if dew_c is not None else None
    ceiling_ft = (base_ft + 200.0) if (base_ft is not None and cloud_pct > 50) else None

    return WeatherSourceData(
        source="OpenWeatherMap",
        wind_speed_knots=_ms_to_knots(wind_ms),
        temperature_c=temp_c,
        humidity_pct=rh,
        visibility_km=_m_to_km(vis_m),
        precipitation_level=_precip_level(precip_mm),
        cloud_base_ft=base_ft,
        cloud_ceiling_ft=ceiling_ft,
        raw=data,
    )


# ---------------------------------------------------------------------------
# WeatherAPI
# ---------------------------------------------------------------------------


async def _fetch_weatherapi(
    lat: float,
    lon: float,
    client: httpx.AsyncClient,
) -> WeatherSourceData | None:
    if not settings.weatherapi_api_key:
        return None

    url = "https://api.weatherapi.com/v1/current.json"
    params = {
        "key": settings.weatherapi_api_key,
        "q": f"{lat},{lon}",
    }
    try:
        resp = await client.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("WeatherAPI fetch failed: %s", exc)
        return None

    cur = data.get("current", {})
    temp_c = cur.get("temp_c", 20.0)
    wind_kmh = cur.get("wind_kph", 0.0)
    rh = cur.get("humidity", 0.0)
    vis_km = cur.get("vis_km", 10.0)
    precip_mm = cur.get("precip_mm", 0.0)
    cloud_pct = cur.get("cloud", 0.0)

    # Estimate dew point
    dew_c = _dew_point_c(temp_c, rh)
    base_ft = _cloud_base_ft(temp_c, dew_c) if dew_c is not None else None
    ceiling_ft = (base_ft + 200.0) if (base_ft is not None and cloud_pct > 50) else None

    return WeatherSourceData(
        source="WeatherAPI",
        wind_speed_knots=_kmh_to_knots(wind_kmh),
        temperature_c=temp_c,
        humidity_pct=rh,
        visibility_km=vis_km,
        precipitation_level=_precip_level(precip_mm),
        cloud_base_ft=base_ft,
        cloud_ceiling_ft=ceiling_ft,
        raw=cur,
    )


# ---------------------------------------------------------------------------
# Windy Point Forecast API
# ---------------------------------------------------------------------------


async def _fetch_windy(
    lat: float,
    lon: float,
    client: httpx.AsyncClient,
) -> WeatherSourceData | None:
    if not settings.windy_api_key:
        return None

    url = "https://api.windy.com/api/point-forecast/v2"
    body = {
        "lat": lat,
        "lon": lon,
        "model": "gfs",
        "parameters": ["wind", "temp", "rh", "lclouds", "mclouds", "hclouds"],
        "levels": ["surface"],
        "key": settings.windy_api_key,
    }
    try:
        resp = await client.post(url, json=body, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Windy fetch failed: %s", exc)
        return None

    # Take first forecast step (index 0)
    def _first(key: str) -> float:
        arr = data.get(key, [0.0])
        return float(arr[0]) if arr else 0.0

    wind_u = _first("wind_u-surface")
    wind_v = _first("wind_v-surface")
    wind_ms = math.sqrt(wind_u**2 + wind_v**2)
    temp_k = _first("temp-surface")
    temp_c = _kelvin_to_celsius(temp_k)
    rh = _first("rh-surface")
    lclouds = _first("lclouds-surface")
    mclouds = _first("mclouds-surface")

    # Estimate dew point
    dew_c = _dew_point_c(temp_c, rh)
    base_ft = _cloud_base_ft(temp_c, dew_c) if dew_c is not None else None
    cloud_pct = max(lclouds, mclouds)
    ceiling_ft = (base_ft + 200.0) if (base_ft is not None and cloud_pct > 50) else None

    return WeatherSourceData(
        source="Windy",
        wind_speed_knots=_ms_to_knots(wind_ms),
        temperature_c=temp_c,
        humidity_pct=rh,
        visibility_km=10.0,  # Windy doesn't provide visibility directly
        precipitation_level=PRECIP_NONE,
        cloud_base_ft=base_ft,
        cloud_ceiling_ft=ceiling_ft,
        raw=data,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def fetch_all_sources(lat: float, lon: float) -> list[WeatherSourceData]:
    """Fetch from all available sources and return non-None results.

    Open-Meteo is always attempted.  The other three require API keys in
    ``settings``; they are skipped (not raising) if the key is absent or
    the request fails.

    Raises ``RuntimeError`` if *no* source succeeds.
    """
    async with httpx.AsyncClient() as client:
        results = [
            await _fetch_open_meteo(lat, lon, client),
            await _fetch_owm(lat, lon, client),
            await _fetch_weatherapi(lat, lon, client),
            await _fetch_windy(lat, lon, client),
        ]

    sources = [r for r in results if r is not None]
    if not sources:
        raise RuntimeError(
            "All weather sources failed.  Check connectivity and API keys."
        )
    return sources

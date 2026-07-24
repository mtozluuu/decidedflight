"""Weather fetcher service.

Retrieves current weather data from up to five sources and normalises them
into ``WeatherSourceData`` objects:

1. **Open-Meteo**  – always attempted (free, no key).
2. **OpenWeatherMap** – requires ``OPENWEATHERMAP_API_KEY``.
3. **WeatherAPI**    – requires ``WEATHERAPI_API_KEY``.
4. **Windy**         – requires ``WINDY_API_KEY``.
5. **METAR (AVWX)**  – requires ``AVWX_API_KEY``.

Sources without a configured API key are silently skipped.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import httpx

from decideflight.config import settings
from decideflight.services.metar_fetcher import (
    estimate_relative_humidity,
    fetch_metar_taf_for_coords,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TIMEOUT = 15.0
_MISSING_VALUE_SENTINEL = -9999

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
    reliability_weight: float = 1.0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class AirQualityData:
    """Normalised air-quality observation."""

    aqi_score: int
    pm25: float | None = None
    pm10: float | None = None
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


def _to_float(value: Any) -> float | None:
    if value in (None, "", _MISSING_VALUE_SENTINEL, str(_MISSING_VALUE_SENTINEL)):
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_first_number(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                parsed = _to_float(item)
                if parsed is not None:
                    return parsed
            continue
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _pick_payload_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("records", "result", "data", "items", "hourlyForecast"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return [payload]


def _pick_closest_row(
    rows: list[dict[str, Any]],
    lat: float,
    lon: float,
) -> dict[str, Any] | None:
    if not rows:
        return None

    def _distance(row: dict[str, Any]) -> float:
        row_lat = _pick_first_number(row, "lat", "latitude", "enlem")
        row_lon = _pick_first_number(row, "lon", "longitude", "boylam")
        if row_lat is None or row_lon is None:
            return float("inf")
        return (row_lat - lat) ** 2 + (row_lon - lon) ** 2

    best = min(rows, key=_distance)
    return best if _distance(best) != float("inf") else rows[0]


def _is_turkey(lat: float, lon: float) -> bool:
    return 36.0 <= lat <= 42.0 and 26.0 <= lon <= 45.0


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


async def _fetch_mgm(
    lat: float,
    lon: float,
    client: httpx.AsyncClient,
) -> WeatherSourceData | None:
    if not _is_turkey(lat, lon):
        return None

    try:
        current_resp = await client.get(
            "https://servis.mgm.gov.tr/web/sondurumlar",
            timeout=_TIMEOUT,
        )
        current_resp.raise_for_status()
        current_rows = _pick_payload_rows(current_resp.json())
    except Exception as exc:
        logger.warning("MGM current conditions fetch failed: %s", exc)
        return None

    current_row = _pick_closest_row(current_rows, lat, lon)
    if current_row is None:
        return None

    hourly_row: dict[str, Any] | None = None
    station_id = current_row.get("istNo") or current_row.get("istno")
    try:
        hourly_resp = await client.get(
            "https://servis.mgm.gov.tr/web/tahminler/saatlik",
            params={"istno": station_id} if station_id is not None else None,
            timeout=_TIMEOUT,
        )
        hourly_resp.raise_for_status()
        hourly_row = _pick_closest_row(_pick_payload_rows(hourly_resp.json()), lat, lon)
    except Exception as exc:
        logger.warning("MGM hourly forecast fetch failed: %s", exc)

    merged = dict(hourly_row or {})
    merged.update(current_row)

    wind_ms = _pick_first_number(merged, "ruzgarHiz", "windSpeed") or 0.0
    temp_c = _pick_first_number(merged, "sicaklik", "temperature") or 20.0
    humidity = _pick_first_number(merged, "nem", "humidity") or 0.0
    visibility_m = _pick_first_number(merged, "gorus", "visibility") or 10000.0
    precip_mm = (
        _pick_first_number(
            merged,
            "yagis1Saat",
            "yagis00Now",
            "yagis10Dk",
            "precipitation",
        )
        or 0.0
    )

    dew_c = _dew_point_c(temp_c, humidity)
    base_ft = _cloud_base_ft(temp_c, dew_c) if dew_c is not None else None

    return WeatherSourceData(
        source="MGM (Türkiye)",
        wind_speed_knots=_ms_to_knots(wind_ms),
        temperature_c=temp_c,
        humidity_pct=humidity,
        visibility_km=_m_to_km(visibility_m),
        precipitation_level=_precip_level(precip_mm),
        cloud_base_ft=base_ft,
        cloud_ceiling_ft=base_ft + 200.0 if base_ft is not None else None,
        reliability_weight=1.15,
        raw=merged,
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
# METAR (AVWX)
# ---------------------------------------------------------------------------


async def _fetch_metar_avwx(
    lat: float,
    lon: float,
    client: httpx.AsyncClient,
) -> WeatherSourceData | None:
    metar = await fetch_metar_taf_for_coords(lat=lat, lon=lon, client=client)
    if metar is None:
        return None

    temp_c = metar.temperature_c if metar.temperature_c is not None else 20.0
    rh = estimate_relative_humidity(metar.temperature_c, metar.dewpoint_c)
    raw_payload: dict[str, Any] = {
        "icao": metar.icao,
        "raw_metar": metar.raw_metar,
        "visibility_text": metar.visibility_text,
        "taf_summary_next_6h": metar.taf_summary_next_6h,
        "temperature_c": metar.temperature_c,
        "dewpoint_c": metar.dewpoint_c,
    }
    raw_payload.update(metar.raw)

    return WeatherSourceData(
        source="METAR (AVWX)",
        wind_speed_knots=metar.wind_speed_kt,
        temperature_c=temp_c,
        humidity_pct=rh,
        visibility_km=metar.visibility_km,
        precipitation_level=PRECIP_NONE,
        cloud_base_ft=metar.cloud_base_ft,
        cloud_ceiling_ft=metar.cloud_base_ft,
        reliability_weight=1.8,
        raw=raw_payload,
    )


async def _fetch_air_quality(
    lat: float,
    lon: float,
    client: httpx.AsyncClient,
) -> AirQualityData | None:
    try:
        resp = await client.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "european_aqi,pm2_5,pm10",
                "timezone": "UTC",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Open-Meteo air-quality fetch failed: %s", exc)
        return None

    current = data.get("current", {})
    aqi = _pick_first_number(current, "european_aqi")
    if aqi is None:
        hourly = data.get("hourly", {})
        aqi = _pick_first_number(hourly, "european_aqi")
        current = hourly
    if aqi is None:
        return None

    return AirQualityData(
        aqi_score=int(round(aqi)),
        pm25=_pick_first_number(current, "pm2_5"),
        pm10=_pick_first_number(current, "pm10"),
        raw=data,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def fetch_all_sources(lat: float, lon: float) -> list[WeatherSourceData]:
    """Fetch from all available sources and return non-None results.

    Open-Meteo is always attempted. The other four require API keys in
    ``settings``; they are skipped (not raising) if the key is absent or
    the request fails.

    Raises ``RuntimeError`` if *no* source succeeds.
    """
    async with httpx.AsyncClient() as client:
        results = [
            await _fetch_open_meteo(lat, lon, client),
            await _fetch_owm(lat, lon, client),
            await _fetch_weatherapi(lat, lon, client),
            await _fetch_mgm(lat, lon, client),
            await _fetch_windy(lat, lon, client),
            await _fetch_metar_avwx(lat, lon, client),
        ]

    sources = [r for r in results if r is not None]
    if not sources:
        raise RuntimeError(
            "All weather sources failed.  Check connectivity and API keys."
        )
    return sources


async def fetch_air_quality(lat: float, lon: float) -> AirQualityData | None:
    """Fetch the current AQI context for the given coordinates."""
    async with httpx.AsyncClient() as client:
        return await _fetch_air_quality(lat, lon, client)

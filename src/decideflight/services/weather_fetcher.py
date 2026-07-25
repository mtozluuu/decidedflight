"""Weather fetcher service.

Retrieves current weather data from multiple forecast and METAR sources and
normalises them into ``WeatherSourceData`` objects. Optional sources without a
configured API key are silently skipped.
"""

from __future__ import annotations

import asyncio
import csv
import logging
import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
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
_MISSING_VALUE_SENTINEL_STR = str(_MISSING_VALUE_SENTINEL)
_OURAIRPORTS_MAX_DISTANCE_KM = 100.0
_STATIC_DATA_DIR = Path(__file__).resolve().parent.parent / "static" / "data"
_OURAIRPORTS_AIRPORTS_CSV = _STATIC_DATA_DIR / "airports.csv"
_OURAIRPORTS_RUNWAYS_CSV = _STATIC_DATA_DIR / "runways.csv"

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
    wind_direction_deg: float | None = None
    reliability_weight: float = 1.0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class AirQualityData:
    """Normalised air-quality observation."""

    aqi_score: int
    pm25: float | None = None
    pm10: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class _AirportRecord:
    icao: str
    name: str
    latitude_deg: float
    longitude_deg: float


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


async def _fetch_with_retry(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> httpx.Response:
    """GET *url* with exponential back-off on 429 Too Many Requests.

    Delays: 1 s, 2 s, 4 s (base_delay * 2^attempt).
    On non-429 HTTP errors the exception is re-raised immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = await client.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429 and attempt < max_retries:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "Open-Meteo rate limited (429) – attempt %d/%d; "
                    "retrying in %.1fs",
                    attempt + 1,
                    max_retries + 1,
                    delay,
                )
                await asyncio.sleep(delay)
                last_exc = exc
            else:
                raise
    # Reached only when all retries were exhausted on 429
    raise (
        last_exc
        if last_exc is not None
        else RuntimeError("Open-Meteo rate limit retries exhausted")  # pragma: no cover
    )


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


def _sm_to_km(sm: float) -> float:
    return sm * 1.609344


def _to_float(value: Any) -> float | None:
    if value in (None, "", _MISSING_VALUE_SENTINEL, _MISSING_VALUE_SENTINEL_STR):
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_icao(value: Any) -> str:
    ident = str(value or "").strip().upper()
    if len(ident) == 4 and ident.isalpha():
        return ident
    return ""


def _parse_number(value: Any) -> float | None:
    parsed = _to_float(value)
    if parsed is not None:
        return parsed

    if value in (None, ""):
        return None

    total = 0.0
    found = False
    for part in str(value).replace("+", " ").replace("SM", " ").split():
        try:
            total += float(Fraction(part))
            found = True
            continue
        except (TypeError, ValueError, ZeroDivisionError):
            pass
        try:
            total += float(part)
            found = True
        except (TypeError, ValueError):
            continue
    return total if found else None


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


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    d_lat = lat2_rad - lat1_rad
    d_lon = lon2_rad - lon1_rad
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(d_lon / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _load_ourairports_airports() -> list[_AirportRecord]:
    airports: list[_AirportRecord] = []
    try:
        with _OURAIRPORTS_AIRPORTS_CSV.open(
            newline="",
            encoding="utf-8",
        ) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                icao = _to_icao(row.get("ident"))
                latitude_deg = _to_float(row.get("latitude_deg"))
                longitude_deg = _to_float(row.get("longitude_deg"))
                if not icao or latitude_deg is None or longitude_deg is None:
                    continue
                airports.append(
                    _AirportRecord(
                        icao=icao,
                        name=str(row.get("name") or icao).strip() or icao,
                        latitude_deg=latitude_deg,
                        longitude_deg=longitude_deg,
                    )
                )
    except Exception as exc:
        logger.warning("Failed to load OurAirports airports data: %s", exc)
    return airports


def _load_ourairports_runways() -> dict[str, list[dict[str, Any]]]:
    runways_by_airport: dict[str, list[dict[str, Any]]] = {}
    try:
        with _OURAIRPORTS_RUNWAYS_CSV.open(
            newline="",
            encoding="utf-8",
        ) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if str(row.get("closed") or "").strip() == "1":
                    continue
                icao = _to_icao(row.get("airport_ident"))
                if not icao:
                    continue
                airport_runways = runways_by_airport.setdefault(icao, [])
                for ident_key, heading_key in (
                    ("le_ident", "le_heading_degT"),
                    ("he_ident", "he_heading_degT"),
                ):
                    runway_ident = str(row.get(ident_key) or "").strip()
                    heading_true = _to_float(row.get(heading_key))
                    if not runway_ident or heading_true is None:
                        continue
                    airport_runways.append(
                        {
                            "runway_ident": runway_ident,
                            "heading_true": round(heading_true, 1),
                        }
                    )
    except Exception as exc:
        logger.warning("Failed to load OurAirports runways data: %s", exc)
    return runways_by_airport


_OURAIRPORTS_AIRPORTS = _load_ourairports_airports()
_OURAIRPORTS_RUNWAYS = _load_ourairports_runways()


def _find_nearest_airports(
    lat: float,
    lon: float,
    *,
    max_results: int,
    max_distance_km: float | None = None,
) -> list[tuple[_AirportRecord, float]]:
    candidates: list[tuple[_AirportRecord, float]] = []
    for airport in _OURAIRPORTS_AIRPORTS:
        distance_km = _haversine_km(
            lat,
            lon,
            airport.latitude_deg,
            airport.longitude_deg,
        )
        if max_distance_km is not None and distance_km > max_distance_km:
            continue
        candidates.append((airport, distance_km))

    candidates.sort(key=lambda item: item[1])
    return candidates[:max_results]


def _find_nearest_airport(lat: float, lon: float) -> _AirportRecord | None:
    nearest = _find_nearest_airports(lat, lon, max_results=1)
    return nearest[0][0] if nearest else None


def _fallback_runways_from_ourairports(lat: float, lon: float) -> list[dict]:
    try:
        results: list[dict] = []
        nearest_airports = _find_nearest_airports(
            lat,
            lon,
            max_results=3,
            max_distance_km=_OURAIRPORTS_MAX_DISTANCE_KM,
        )
        for airport, distance_km in nearest_airports:
            for runway in _OURAIRPORTS_RUNWAYS.get(airport.icao, []):
                results.append(
                    {
                        "airport_icao": airport.icao,
                        "airport_name": airport.name,
                        "runway_ident": runway["runway_ident"],
                        "heading_true": runway["heading_true"],
                        "distance_km": round(distance_km, 1),
                    }
                )
        return results
    except Exception as exc:
        logger.warning("OurAirports runway fallback failed: %s", exc)
        return []


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
            "wind_direction_10m,"
            "precipitation,"
            "visibility,"
            "cloud_cover"
        ),
        "wind_speed_unit": "kmh",
        "timezone": "UTC",
    }
    try:
        resp = await _fetch_with_retry(client, url, params)
        data = resp.json()
    except Exception as exc:
        logger.warning("Open-Meteo fetch failed: %s", exc)
        return None

    cur = data.get("current", {})
    temp = cur.get("temperature_2m")
    dew = cur.get("dew_point_2m")
    wind_kmh = cur.get("wind_speed_10m", 0.0)
    wind_dir = cur.get("wind_direction_10m")
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
        wind_direction_deg=float(wind_dir) if wind_dir is not None else None,
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
    wind_dir_owm = wind.get("deg")
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
        wind_direction_deg=float(wind_dir_owm) if wind_dir_owm is not None else None,
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
    wind_dir_wa = cur.get("wind_degree")
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
        wind_direction_deg=float(wind_dir_wa) if wind_dir_wa is not None else None,
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
    wind_dir_mgm = _pick_first_number(merged, "ruzgarYon", "windDirection")
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
        wind_direction_deg=float(wind_dir_mgm) if wind_dir_mgm is not None else None,
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
    # Derive meteorological wind direction from u/v components
    # (direction wind comes FROM, 0° = North, clockwise)
    wind_dir_windy: float | None = None
    if wind_ms > 0:
        wind_dir_windy = (270.0 - math.degrees(math.atan2(wind_v, wind_u))) % 360.0
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
        wind_direction_deg=wind_dir_windy,
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

    # Extract wind direction from raw METAR data
    wind_dir_metar: float | None = None
    metar_data = metar.raw.get("metar", {}) or {}
    wind_dir_raw = metar_data.get("wind_direction")
    if isinstance(wind_dir_raw, dict):
        wd_val = wind_dir_raw.get("value")
        if wd_val is not None:
            try:
                wind_dir_metar = float(wd_val)
            except (TypeError, ValueError):
                pass
    elif wind_dir_raw is not None:
        try:
            wind_dir_metar = float(wind_dir_raw)
        except (TypeError, ValueError):
            pass

    return WeatherSourceData(
        source="METAR (AVWX)",
        wind_speed_knots=metar.wind_speed_kt,
        temperature_c=temp_c,
        humidity_pct=rh,
        visibility_km=metar.visibility_km,
        precipitation_level=PRECIP_NONE,
        cloud_base_ft=metar.cloud_base_ft,
        cloud_ceiling_ft=metar.cloud_base_ft,
        wind_direction_deg=wind_dir_metar,
        reliability_weight=1.8,
        raw=raw_payload,
    )


async def _fetch_avwx_gov_metar(
    icao: str,
    client: httpx.AsyncClient,
) -> WeatherSourceData | None:
    try:
        resp = await client.get(
            "https://aviationweather.gov/api/data/metar",
            params={"ids": icao, "format": "json"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        logger.warning("aviationweather.gov METAR fetch failed for %s: %s", icao, exc)
        return None

    rows = _pick_payload_rows(payload)
    if not rows:
        return None

    metar = rows[0]
    temp_c = _pick_first_number(metar, "temp")
    dewpoint_c = _pick_first_number(metar, "dwpt")
    visibility_sm = _parse_number(metar.get("visib"))
    visibility_km = _sm_to_km(visibility_sm) if visibility_sm is not None else 10.0

    return WeatherSourceData(
        source="METAR (NOAA/aviationweather.gov)",
        wind_speed_knots=_pick_first_number(metar, "wspd") or 0.0,
        temperature_c=temp_c if temp_c is not None else 20.0,
        humidity_pct=estimate_relative_humidity(temp_c, dewpoint_c),
        visibility_km=visibility_km,
        precipitation_level=PRECIP_NONE,
        cloud_base_ft=None,
        cloud_ceiling_ft=None,
        wind_direction_deg=_pick_first_number(metar, "wdir"),
        reliability_weight=1.8,
        raw={"icao": icao, **metar},
    )


async def _fetch_metar_aviationweather(
    lat: float,
    lon: float,
    client: httpx.AsyncClient,
) -> WeatherSourceData | None:
    airport = _find_nearest_airport(lat, lon)
    if airport is None:
        return None
    return await _fetch_avwx_gov_metar(airport.icao, client)


async def _fetch_metar_checkwx(
    lat: float,
    lon: float,
    client: httpx.AsyncClient,
) -> WeatherSourceData | None:
    if not settings.checkwx_api_key:
        return None

    airport = _find_nearest_airport(lat, lon)
    if airport is None:
        return None

    try:
        resp = await client.get(
            f"https://api.checkwx.com/metar/{airport.icao}/decoded",
            headers={"X-API-Key": settings.checkwx_api_key},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        logger.warning("CheckWX METAR fetch failed for %s: %s", airport.icao, exc)
        return None

    rows = _pick_payload_rows(payload)
    if not rows:
        return None

    metar = rows[0]
    wind = metar.get("wind") or {}
    visibility = metar.get("visibility") or {}
    temperature = metar.get("temperature") or {}
    dewpoint = metar.get("dewpoint") or {}

    visibility_km: float
    meters_visibility = _parse_number(visibility.get("meters"))
    miles_visibility = _parse_number(visibility.get("miles"))
    if meters_visibility is not None:
        visibility_km = _m_to_km(meters_visibility)
    elif miles_visibility is not None:
        visibility_km = _sm_to_km(miles_visibility)
    else:
        visibility_km = 10.0

    temp_c = _parse_number(temperature.get("celsius"))
    dewpoint_c = _parse_number(dewpoint.get("celsius"))

    return WeatherSourceData(
        source="METAR (CheckWX)",
        wind_speed_knots=_parse_number(wind.get("speed_kts")) or 0.0,
        temperature_c=temp_c if temp_c is not None else 20.0,
        humidity_pct=estimate_relative_humidity(temp_c, dewpoint_c),
        visibility_km=visibility_km,
        precipitation_level=PRECIP_NONE,
        cloud_base_ft=None,
        cloud_ceiling_ft=None,
        wind_direction_deg=_parse_number(wind.get("degrees")),
        reliability_weight=1.7,
        raw={"icao": airport.icao, **metar},
    )


async def _fetch_air_quality(
    lat: float,
    lon: float,
    client: httpx.AsyncClient,
) -> AirQualityData | None:
    try:
        resp = await _fetch_with_retry(
            client,
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            {
                "latitude": lat,
                "longitude": lon,
                "current": "european_aqi,pm2_5,pm10",
                "timezone": "UTC",
            },
        )
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

    Open-Meteo is always attempted. The other optional sources require API
    keys in ``settings`` when applicable; they are skipped (not raising) if
    the key is absent or
    the request fails.

    Raises ``RuntimeError`` if *no* source succeeds.
    """
    async with httpx.AsyncClient() as client:
        gathered = await asyncio.gather(
            _fetch_open_meteo(lat, lon, client),
            _fetch_owm(lat, lon, client),
            _fetch_weatherapi(lat, lon, client),
            _fetch_mgm(lat, lon, client),
            _fetch_windy(lat, lon, client),
            _fetch_metar_avwx(lat, lon, client),
            _fetch_metar_aviationweather(lat, lon, client),
            _fetch_metar_checkwx(lat, lon, client),
            return_exceptions=True,
        )

    results: list[WeatherSourceData | None] = []
    for item in gathered:
        if isinstance(item, Exception):
            logger.warning("Unexpected weather source fetch failure: %s", item)
            continue
        results.append(item)

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


async def fetch_nearby_runways(lat: float, lon: float) -> list[dict]:
    """Fetch nearby airport runways from AVWX API.

    Returns a list of dicts like::

        [
          {
            "airport_icao": "LTFM",
            "airport_name": "Istanbul Airport",
            "runway_ident": "35L",
            "heading_true": 350.0,
            "distance_km": 12.3,
          },
          ...
        ]

    Falls back to bundled OurAirports CSV data when AVWX returns no runways.
    Returns an empty list if the AVWX key is missing or any request fails.
    """
    if not settings.avwx_api_key:
        return []

    headers = {"Authorization": settings.avwx_api_key}
    results: list[dict] = []

    try:
        async with httpx.AsyncClient() as client:
            near_url = f"https://avwx.rest/api/station/near/{lat},{lon}"
            near_params = {"n": 3, "airport": "true"}
            try:
                near_resp = await client.get(
                    near_url,
                    params=near_params,
                    headers=headers,
                    timeout=_TIMEOUT,
                )
                near_resp.raise_for_status()
                near_data = near_resp.json()
            except Exception as exc:
                logger.warning("AVWX nearby stations fetch failed: %s", exc)
                return []

            if not isinstance(near_data, list):
                return []

            for item in near_data:
                station = item.get("station") or {}
                icao = station.get("icao") or station.get("ident") or ""
                if not icao:
                    continue
                airport_name = station.get("name", icao)
                distance_km = float(item.get("kilometers") or 0.0)

                runways = station.get("runways") or []
                for rwy in runways:
                    for side, bearing_key in (
                        ("ident1", "bearing1"),
                        ("ident2", "bearing2"),
                    ):
                        ident = rwy.get(side, "")
                        if not ident:
                            continue
                        bearing = rwy.get(bearing_key)
                        if bearing is None:
                            continue
                        results.append(
                            {
                                "airport_icao": icao,
                                "airport_name": airport_name,
                                "runway_ident": ident,
                                "heading_true": round(float(bearing), 1),
                                "distance_km": round(distance_km, 1),
                            }
                        )
    except Exception as exc:
        logger.warning("fetch_nearby_runways failed: %s", exc)
        return []

    return results if results else _fallback_runways_from_ourairports(lat, lon)

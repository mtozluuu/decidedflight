"""Gridded wind and cloud-cover samples for the map overlay (Open-Meteo)."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import httpx

AIRPORT_TYPES = {"large_airport", "medium_airport", "small_airport"}
_airport_cache: dict[str, Any] = {"airports": None}

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_TTL_SECONDS = 15 * 60
LAT_RANGE = (36.0, 42.0)
LON_RANGE = (26.0, 44.0)
GRID_STEP = 1.5

LEVELS: dict[str, dict[str, Any]] = {
    "surface": {"label": "Yüzey (10 m)", "suffix": "", "wind": "10m", "cloud": "cloud_cover"},
    "850": {"label": "850 hPa (~FL050)", "suffix": "_850hPa"},
    "700": {"label": "700 hPa (~FL100)", "suffix": "_700hPa"},
    "500": {"label": "500 hPa (~FL180)", "suffix": "_500hPa"},
}

_cache: dict[str, tuple[float, dict[str, Any]]] = {}


class MapLayerUnavailableError(RuntimeError):
    """Raised when the weather provider cannot supply grid data."""


def _read_turkish_airports() -> list[dict[str, Any]]:
    path = Path(__file__).resolve().parents[1] / "static" / "data" / "airports.csv"
    airports: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            ident = (row.get("ident") or "").strip().upper()
            if (
                row.get("iso_country") != "TR"
                or row.get("type") not in AIRPORT_TYPES
                or not ident.startswith("LT")
                or len(ident) != 4
            ):
                continue
            try:
                lat, lon = float(row["latitude_deg"]), float(row["longitude_deg"])
            except (KeyError, ValueError):
                continue
            airports.append({"icao": ident, "name": row.get("name") or ident, "lat": lat, "lon": lon})
    return airports


def turkish_airports() -> list[dict[str, Any]]:
    if _airport_cache["airports"] is None:
        _airport_cache["airports"] = _read_turkish_airports()
    return _airport_cache["airports"]


def grid_points() -> list[tuple[float, float]]:
    lats = _frange(*LAT_RANGE)
    lons = _frange(*LON_RANGE)
    return [(lat, lon) for lat in lats for lon in lons]


def _frange(start: float, stop: float) -> list[float]:
    count = int(round((stop - start) / GRID_STEP))
    return [round(start + i * GRID_STEP, 4) for i in range(count + 1)]


def _variables(level: str) -> list[str]:
    if level == "surface":
        return ["wind_speed_10m", "wind_direction_10m", "cloud_cover"]
    suffix = LEVELS[level]["suffix"]
    return [f"wind_speed{suffix}", f"wind_direction{suffix}", f"cloud_cover{suffix}"]


def parse_response(level: str, points: list[tuple[float, float]], payload: Any) -> dict[str, Any]:
    speed_key, dir_key, cloud_key = _variables(level)
    entries = payload if isinstance(payload, list) else [payload]
    if len(entries) != len(points):
        raise MapLayerUnavailableError("Beklenmeyen Open-Meteo yanıtı.")
    samples = []
    valid_time = None
    for (lat, lon), entry in zip(points, entries):
        hourly = entry.get("hourly") or {}
        times = hourly.get("time") or []
        if times and valid_time is None:
            valid_time = times[0]

        def first(key: str) -> float | None:
            values = hourly.get(key) or []
            return values[0] if values else None

        samples.append(
            {
                "lat": lat,
                "lon": lon,
                "wind_speed_knots": first(speed_key),
                "wind_direction_deg": first(dir_key),
                "cloud_cover_pct": first(cloud_key),
            }
        )
    return {"level": level, "label": LEVELS[level]["label"], "valid_time_utc": valid_time, "points": samples}


async def load_layer(level: str) -> dict[str, Any]:
    now = time.time()
    cached = _cache.get(level)
    if cached and now - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]
    points = grid_points()
    params = {
        "latitude": ",".join(str(lat) for lat, _ in points),
        "longitude": ",".join(str(lon) for _, lon in points),
        "hourly": ",".join(_variables(level)),
        "wind_speed_unit": "kn",
        "forecast_hours": 1,
        "timezone": "GMT",
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(OPEN_METEO_URL, params=params)
            response.raise_for_status()
            data = parse_response(level, points, response.json())
    except (httpx.HTTPError, ValueError) as exc:
        if cached:
            return cached[1]
        raise MapLayerUnavailableError("Open-Meteo harita verisi alınamadı.") from exc
    except MapLayerUnavailableError:
        if cached:
            return cached[1]
        raise
    _cache[level] = (now, data)
    return data

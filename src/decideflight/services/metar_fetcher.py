"""AVWX-backed METAR/TAF fetcher.

This module provides resilient helpers for retrieving aviation weather from
AVWX. All failures are handled gracefully and return ``None``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from decideflight.config import settings
from decideflight.services.geocoding import geocode_city

logger = logging.getLogger(__name__)

_BASE_URL = "https://avwx.rest/api"
_TIMEOUT = 12.0


@dataclass
class MetarTafData:
    """Parsed METAR + TAF payload from AVWX."""

    icao: str
    wind_speed_kt: float
    visibility_text: str
    visibility_km: float
    cloud_base_ft: float | None
    temperature_c: float | None
    dewpoint_c: float | None
    raw_metar: str
    taf_summary_next_6h: str | None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, dict):
        for key in ("value", "repr"):
            number = _to_float(value.get(key))
            if number is not None:
                return number
    if isinstance(value, str):
        cleaned = value.strip().upper().replace("KT", "")
        if cleaned == "":
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _extract_icao(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in ("icao", "station", "ident", "id"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().upper()
        for key in ("station", "stations", "results", "data"):
            value = payload.get(key)
            code = _extract_icao(value)
            if code:
                return code
    if isinstance(payload, list):
        for item in payload:
            code = _extract_icao(item)
            if code:
                return code
    return None


def _parse_visibility(visibility: Any) -> tuple[str, float]:
    if isinstance(visibility, dict):
        value = _to_float(visibility)
        unit = str(visibility.get("units") or visibility.get("unit") or "").lower()
        repr_value = str(visibility.get("repr") or "").strip()
        if value is not None and unit in {"sm", "mile", "miles"}:
            return (repr_value or f"{value:g}SM", value * 1.60934)
        if value is not None:
            if unit in {"km", "kilometer", "kilometers"}:
                return (repr_value or f"{value:g}km", value)
            return (repr_value or f"{int(value)}m", value / 1000.0)
        return (repr_value or "N/A", 10.0)

    if isinstance(visibility, str):
        text = visibility.strip().upper()
        if text.endswith("SM"):
            number = _to_float(text[:-2])
            if number is not None:
                return (text, number * 1.60934)
        number = _to_float(text)
        if number is not None:
            return (text, number / 1000.0)
        return (text or "N/A", 10.0)

    value = _to_float(visibility)
    if value is None:
        return ("N/A", 10.0)
    return (str(int(value)), value / 1000.0)


def _parse_cloud_base_ft(clouds: Any) -> float | None:
    if not isinstance(clouds, list):
        return None

    candidates: list[float] = []
    for cloud in clouds:
        if not isinstance(cloud, dict):
            continue
        base = _to_float(cloud.get("base"))
        if base is None:
            continue
        unit = str(cloud.get("unit") or cloud.get("units") or "ft").lower()
        if unit in {"m", "meter", "meters"}:
            base *= 3.28084
        candidates.append(base)

    return min(candidates) if candidates else None


def _parse_taf_next_6h(taf_payload: dict[str, Any]) -> str | None:
    now = datetime.now(timezone.utc)
    until = now + timedelta(hours=6)
    selected: list[str] = []

    for period in taf_payload.get("forecast", []):
        if not isinstance(period, dict):
            continue
        raw = str(period.get("sanitized") or period.get("raw") or "").strip()
        if not raw:
            continue

        start_dt = None
        end_dt = None
        for key in ("start_time", "end_time"):
            item = period.get(key)
            if isinstance(item, dict):
                dt = item.get("dt")
                if isinstance(dt, str):
                    try:
                        parsed = datetime.fromisoformat(dt.replace("Z", "+00:00"))
                    except ValueError:
                        parsed = None
                    if parsed is not None:
                        if key == "start_time":
                            start_dt = parsed
                        else:
                            end_dt = parsed

        overlaps = True
        if start_dt is not None and start_dt > until:
            overlaps = False
        if end_dt is not None and end_dt < now:
            overlaps = False

        if overlaps:
            selected.append(raw)

    if selected:
        return " | ".join(selected[:3])

    fallback = taf_payload.get("raw")
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()[:240]
    return None


async def _resolve_nearest_icao(
    lat: float,
    lon: float,
    client: httpx.AsyncClient,
) -> str | None:
    url = f"{_BASE_URL}/station/near/{lat},{lon}"
    try:
        resp = await client.get(
            url,
            params={"n": 1, "token": settings.avwx_api_key},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return _extract_icao(resp.json())
    except Exception as exc:
        logger.warning("AVWX nearest station lookup failed: %s", exc)
        return None


async def _fetch_metar(
    icao: str,
    client: httpx.AsyncClient,
) -> dict[str, Any] | None:
    url = f"{_BASE_URL}/metar/{icao}"
    try:
        resp = await client.get(
            url, params={"token": settings.avwx_api_key}, timeout=_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning("AVWX METAR fetch failed for %s: %s", icao, exc)
        return None


async def _fetch_taf(
    icao: str,
    client: httpx.AsyncClient,
) -> dict[str, Any] | None:
    url = f"{_BASE_URL}/taf/{icao}"
    try:
        resp = await client.get(
            url, params={"token": settings.avwx_api_key}, timeout=_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning("AVWX TAF fetch failed for %s: %s", icao, exc)
        return None


def _relative_humidity_from_temp_dew(
    temp_c: float | None,
    dewpoint_c: float | None,
) -> float:
    if temp_c is None or dewpoint_c is None:
        return 50.0

    a = 17.625
    b = 243.04
    sat = math.exp((a * temp_c) / (b + temp_c))
    actual = math.exp((a * dewpoint_c) / (b + dewpoint_c))
    rh = max(0.0, min(100.0, (actual / sat) * 100.0))
    return rh


async def fetch_metar_taf_for_coords(
    lat: float,
    lon: float,
    client: httpx.AsyncClient | None = None,
) -> MetarTafData | None:
    """Fetch AVWX METAR/TAF for the nearest station to given coordinates."""
    if not settings.avwx_api_key:
        return None

    owned_client = client is None
    if owned_client:
        client = httpx.AsyncClient()

    try:
        icao = await _resolve_nearest_icao(lat, lon, client)
        if not icao:
            return None

        metar = await _fetch_metar(icao, client)
        if metar is None:
            return None

        taf = await _fetch_taf(icao, client)

        wind_speed = _to_float(metar.get("wind_speed")) or 0.0
        visibility_text, visibility_km = _parse_visibility(metar.get("visibility"))
        cloud_base_ft = _parse_cloud_base_ft(metar.get("clouds"))
        temperature_c = _to_float(metar.get("temperature"))
        dewpoint_c = _to_float(metar.get("dewpoint"))
        raw_metar = str(metar.get("raw") or metar.get("sanitized") or "").strip()
        taf_summary = _parse_taf_next_6h(taf or {})

        return MetarTafData(
            icao=icao,
            wind_speed_kt=wind_speed,
            visibility_text=visibility_text,
            visibility_km=visibility_km,
            cloud_base_ft=cloud_base_ft,
            temperature_c=temperature_c,
            dewpoint_c=dewpoint_c,
            raw_metar=raw_metar,
            taf_summary_next_6h=taf_summary,
            raw={"metar": metar, "taf": taf},
        )
    except Exception:
        # Keep this service silent by design.
        return None
    finally:
        if owned_client:
            await client.aclose()


async def fetch_metar_taf_for_location(
    location: str,
    client: httpx.AsyncClient | None = None,
) -> MetarTafData | None:
    """Resolve location to coordinates via geocoding, then fetch METAR/TAF."""
    if not settings.avwx_api_key:
        return None

    try:
        lat, lon, _ = await geocode_city(location)
    except Exception:
        return None

    return await fetch_metar_taf_for_coords(lat=lat, lon=lon, client=client)


def estimate_relative_humidity(temp_c: float | None, dewpoint_c: float | None) -> float:
    """Public helper for humidity estimate from METAR temperature/dew point."""
    return _relative_humidity_from_temp_dew(temp_c, dewpoint_c)

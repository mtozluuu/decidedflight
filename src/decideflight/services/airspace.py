"""Turkish airspace (TMA/CTR/FIR/...) data from the public openAIP export."""

from __future__ import annotations

import time
from typing import Any

import httpx

OPENAIP_AIRSPACE_URL = "https://storage.openaip.net/openaip-system-exports/tr_asp.geojson"
CACHE_TTL_SECONDS = 24 * 60 * 60

AIRSPACE_TYPES: dict[int, tuple[str, str]] = {
    1: ("R", "Kısıtlı"),
    2: ("D", "Tehlikeli"),
    3: ("P", "Yasak"),
    4: ("CTR", "Kontrol bölgesi"),
    7: ("TMA", "Terminal kontrol sahası"),
    8: ("TRA", "Geçici ayrılmış saha"),
    9: ("TSA", "Geçici ayrı saha"),
    10: ("FIR", "Uçuş bilgi bölgesi"),
    13: ("ATZ", "Hava trafik bölgesi"),
    14: ("MATZ", "Askerî hava trafik bölgesi"),
    25: ("MTA", "Askerî terminal sahası"),
}
DEFAULT_TYPES = ("TMA", "CTR", "FIR")

_UNITS = {0: "m", 1: "ft", 6: "FL"}
_DATUMS = {0: "AGL", 1: "AMSL", 2: ""}

_cache: dict[str, Any] = {"fetched_at": 0.0, "features": None}


class AirspaceUnavailableError(RuntimeError):
    """Raised when airspace data cannot be retrieved and no cache exists."""


def format_limit(limit: dict[str, Any] | None) -> str:
    """Render an openAIP vertical limit such as ``FL240`` or ``3500 ft AMSL``."""
    if not limit or limit.get("value") is None:
        return "—"
    value = limit["value"]
    unit = limit.get("unit")
    datum = limit.get("referenceDatum")
    if unit == 6:
        return f"FL{int(value):03d}"
    if value == 0 and datum == 0:
        return "GND"
    unit_text = _UNITS.get(unit, "")
    datum_text = _DATUMS.get(datum, "")
    return " ".join(part for part in (f"{value:g}", unit_text, datum_text) if part)


def _round_ring(ring: list[list[float]]) -> list[list[float]]:
    return [[round(point[0], 5), round(point[1], 5)] for point in ring]


def _normalize_feature(raw: dict[str, Any]) -> dict[str, Any] | None:
    props = raw.get("properties") or {}
    geometry = raw.get("geometry") or {}
    type_info = AIRSPACE_TYPES.get(props.get("type"))
    if type_info is None or geometry.get("type") != "Polygon":
        return None
    return {
        "type": "Feature",
        "properties": {
            "id": props.get("_id"),
            "name": props.get("name") or "—",
            "category": type_info[0],
            "category_label": type_info[1],
            "lower": format_limit(props.get("lowerLimit")),
            "upper": format_limit(props.get("upperLimit")),
            "updated_at": props.get("updatedAt"),
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [_round_ring(ring) for ring in geometry["coordinates"]],
        },
    }


def normalize_collection(payload: dict[str, Any]) -> list[dict[str, Any]]:
    features = (_normalize_feature(raw) for raw in payload.get("features", []))
    return [feature for feature in features if feature is not None]


async def load_airspaces(force: bool = False) -> tuple[list[dict[str, Any]], float]:
    """Return normalized features and the time they were fetched (epoch seconds)."""
    now = time.time()
    cached = _cache["features"]
    if cached is not None and not force and now - _cache["fetched_at"] < CACHE_TTL_SECONDS:
        return cached, _cache["fetched_at"]
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(OPENAIP_AIRSPACE_URL)
            response.raise_for_status()
            features = normalize_collection(response.json())
    except (httpx.HTTPError, ValueError) as exc:
        if cached is not None:
            return cached, _cache["fetched_at"]
        raise AirspaceUnavailableError("openAIP hava sahası verisi alınamadı.") from exc
    _cache["features"] = features
    _cache["fetched_at"] = now
    return features, now


def filter_features(features: list[dict[str, Any]], categories: set[str]) -> list[dict[str, Any]]:
    return [f for f in features if f["properties"]["category"] in categories]

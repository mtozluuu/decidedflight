"""Geocoding service using the Open-Meteo geocoding API (no key required)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_TIMEOUT = 10.0
_CITIES_PATH = Path(__file__).resolve().parent.parent / "static" / "data" / "cities.json"


async def geocode_city(city: str) -> tuple[float, float, str]:
    """Return *(lat, lon, display_name)* for *city*.

    Raises ``ValueError`` if the city cannot be found.
    """
    name, separator, country = city.partition(",")
    params = {"name": name.strip(), "count": 1, "language": "en", "format": "json"}
    if separator:
        country = country.strip().casefold()
        countries = json.loads(_CITIES_PATH.read_text(encoding="utf-8"))["countries"]
        match = next(
            (
                item for item in countries
                if country in {
                    item["code"].casefold(),
                    item["name"].casefold(),
                    item.get("queryName", "").casefold(),
                }
            ),
            None,
        )
        if match is None:
            raise ValueError(f"Country not found: {country!r}")
        params["countryCode"] = match["code"]

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(
            _GEOCODING_URL,
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results")
    if not results:
        raise ValueError(f"City not found: {city!r}")

    hit = results[0]
    lat: float = hit["latitude"]
    lon: float = hit["longitude"]
    display_name = (f"{hit.get('name', city)}, {hit.get('country', '')}").strip(", ")
    return lat, lon, display_name

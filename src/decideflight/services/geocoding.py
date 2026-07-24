"""Geocoding service using the Open-Meteo geocoding API (no key required)."""

from __future__ import annotations

import httpx

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_TIMEOUT = 10.0


async def geocode_city(city: str) -> tuple[float, float, str]:
    """Return *(lat, lon, display_name)* for *city*.

    Raises ``ValueError`` if the city cannot be found.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(
            _GEOCODING_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
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

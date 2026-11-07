"""Hourly pressure-level profiles for the point-analysis Airgram."""

import math

import httpx

LEVELS = (1000, 925, 850, 700, 500, 300)


async def fetch_airgram(lat: float, lon: float) -> dict:
    fields = [
        f"{name}_{level}hPa"
        for level in LEVELS
        for name in ("temperature", "relative_humidity", "wind_speed", "wind_direction",
                     "geopotential_height")
    ]
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon, "hourly": ",".join(fields),
                "forecast_hours": 48, "timezone": "UTC", "wind_speed_unit": "kn",
            },
        )
        response.raise_for_status()
    hourly = response.json().get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        raise ValueError("Open-Meteo saatlik profil döndürmedi.")

    def values(name: str, level: int) -> list:
        result = hourly.get(f"{name}_{level}hPa")
        if not isinstance(result, list) or len(result) != len(times):
            raise ValueError("Open-Meteo profil alanları eksik.")
        return [v if isinstance(v, (float, int)) and math.isfinite(v) else None
                for v in result]

    return {
        "times": times, "timezone": "UTC",
        "levels": [
            {
                "pressure_hpa": level,
                "temperature_c": values("temperature", level),
                "humidity_pct": values("relative_humidity", level),
                "wind_knots": values("wind_speed", level),
                "wind_direction_deg": values("wind_direction", level),
                "height_m": values("geopotential_height", level),
            }
            for level in LEVELS
        ],
    }

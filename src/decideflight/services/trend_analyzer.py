"""Wind trend analyzer.

Fetches the last 6 hours of hourly wind data from Open-Meteo and computes a
simple linear trend (increasing / decreasing / stable).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class WindTrend:
    """Result of a wind trend computation."""

    direction: str  # "artıyor" | "azalıyor" | "sabit"
    change_per_hour_kt: float  # signed kt/h (positive = increasing)
    description: str  # Human-readable Turkish description


async def fetch_wind_trend(lat: float, lon: float) -> WindTrend | None:
    """Fetch the last 6 hours of hourly wind data and compute the trend.

    Returns *None* if the fetch fails.  Uses Open-Meteo ``past_hours=6``.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_speed_10m",
        "wind_speed_unit": "kn",
        "past_hours": 6,
        "forecast_hours": 1,
        "timezone": "UTC",
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("Wind trend fetch failed: %s", exc)
        return None

    hourly = data.get("hourly", {})
    speeds: list[float] = [
        float(v) for v in hourly.get("wind_speed_10m", []) if v is not None
    ]

    if len(speeds) < 2:
        return None

    return _compute_trend(speeds)


def _compute_trend(speeds: list[float]) -> WindTrend:
    """Compute a simple linear regression slope over the speed values."""
    n = len(speeds)
    if n < 2:
        return WindTrend(
            direction="sabit",
            change_per_hour_kt=0.0,
            description="Rüzgar trendi: sabit",
        )

    # Simple linear regression: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    x_vals = list(range(n))
    sum_x = sum(x_vals)
    sum_y = sum(speeds)
    sum_xy = sum(x * y for x, y in zip(x_vals, speeds))
    sum_x2 = sum(x * x for x in x_vals)

    denom = n * sum_x2 - sum_x**2
    if denom == 0:
        slope = 0.0
    else:
        slope = (n * sum_xy - sum_x * sum_y) / denom

    # Threshold: ±0.3 kt/h is considered stable
    if slope > 0.3:
        direction = "artıyor"
    elif slope < -0.3:
        direction = "azalıyor"
    else:
        direction = "sabit"

    abs_slope = abs(slope)
    description = f"Son 6 saatte rüzgar: {abs_slope:.1f} kt/h {direction}"

    return WindTrend(
        direction=direction,
        change_per_hour_kt=round(slope, 2),
        description=description,
    )

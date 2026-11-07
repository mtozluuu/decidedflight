from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from decideflight.main import app
from decideflight.services.airgram import LEVELS, fetch_airgram


@pytest.mark.asyncio
async def test_airgram_pressure_profile_units_and_missing_values():
    hourly = {"time": ["2026-11-07T00:00", "2026-11-07T01:00"]}
    for level in LEVELS:
        for field in ("temperature", "relative_humidity", "wind_speed",
                      "wind_direction", "geopotential_height"):
            hourly[f"{field}_{level}hPa"] = [10, None]
    response = httpx.Response(
        200, json={"hourly": hourly},
        request=httpx.Request("GET", "https://api.open-meteo.com/v1/forecast"),
    )
    with patch("decideflight.services.airgram.httpx.AsyncClient") as factory:
        client = factory.return_value.__aenter__.return_value
        client.get = AsyncMock(return_value=response)
        result = await fetch_airgram(37, 38)
    assert result["levels"][0]["wind_knots"] == [10, None]
    assert result["timezone"] == "UTC"
    params = client.get.call_args.kwargs["params"]
    assert params["latitude"] == 37
    assert params["longitude"] == 38
    assert params["wind_speed_unit"] == "kn"


@pytest.mark.parametrize("icao", ["LTAJ", "LTAC"])
def test_mgm_airgram_uses_selected_airport(icao):
    response = httpx.Response(
        200, content=b"\x89PNG\r\n\x1a\nexample",
        request=httpx.Request("GET", "https://hezarfen.mgm.gov.tr/Genel/pngGetir.ashx"),
    )
    with patch("decideflight.api.weather.httpx.AsyncClient") as factory:
        client = factory.return_value.__aenter__.return_value
        client.get = AsyncMock(return_value=response)
        result = TestClient(app).get(f"/api/v1/weather/airport-airgram/{icao}")
    assert result.status_code == 200
    assert result.headers["content-type"] == "image/png"
    assert client.get.call_args.kwargs["params"] == {"syol": f"{icao}.png"}


def test_mgm_airgram_does_not_return_html_as_image():
    response = httpx.Response(
        200, text="not available",
        request=httpx.Request("GET", "https://hezarfen.mgm.gov.tr/Genel/pngGetir.ashx"),
    )
    with patch("decideflight.api.weather.httpx.AsyncClient") as factory:
        client = factory.return_value.__aenter__.return_value
        client.get = AsyncMock(return_value=response)
        result = TestClient(app).get("/api/v1/weather/airport-airgram/LTAC")
    assert result.status_code == 404


def test_airgram_rejects_invalid_coordinates():
    assert TestClient(app).get("/api/v1/weather/airgram?lat=91&lon=35").status_code == 422


def test_airgram_reports_provider_failure():
    with patch(
        "decideflight.api.weather.fetch_airgram",
        new=AsyncMock(side_effect=ValueError("missing fields")),
    ):
        result = TestClient(app).get("/api/v1/weather/airgram?lat=37&lon=38")
    assert result.status_code == 503

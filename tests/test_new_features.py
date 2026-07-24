from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from decideflight.main import app
from decideflight.services.weather_fetcher import (
    PRECIP_NONE,
    WeatherSourceData,
    _fetch_air_quality,
    _fetch_mgm,
)


class _MockResponse:
    def __init__(self, payload: dict | list):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _source() -> WeatherSourceData:
    return WeatherSourceData(
        source="Open-Meteo",
        wind_speed_knots=9.0,
        temperature_c=22.0,
        humidity_pct=55.0,
        visibility_km=10.0,
        precipitation_level=PRECIP_NONE,
        cloud_base_ft=2500.0,
        cloud_ceiling_ft=3200.0,
    )


def _create_report(client: TestClient, *, location: str = "Test Location") -> int:
    with patch(
        "decideflight.api.weather.fetch_all_sources",
        new=AsyncMock(return_value=[_source()]),
    ):
        response = client.post(
            "/api/v1/weather/report",
            json={"lat": 41.0, "lon": 28.9, "location": location},
        )
    assert response.status_code == 201
    return response.json()["report_id"]


class TestNewWeatherEndpoints:
    def test_chat_endpoint_returns_reply(self, client: TestClient):
        report_id = _create_report(client, location="Chat Noktası")
        with patch(
            "decideflight.api.weather.chat_about_report",
            new=AsyncMock(return_value="500 ft için rüzgar tekrar kontrol edilmeli."),
        ):
            response = client.post(
                f"/api/v1/weather/report/{report_id}/chat",
                json={"message": "500ft irtifada uçsam ne olur?"},
            )

        assert response.status_code == 200
        assert response.json() == {
            "reply": "500 ft için rüzgar tekrar kontrol edilmeli.",
        }

    def test_forecast_change_endpoint_returns_six_hours(self, client: TestClient):
        report_id = _create_report(client, location="Tahmin Noktası")
        hourly_points = [
            {
                "wind_speed_knots": 8 + offset,
                "temperature_c": 20.0,
                "humidity_pct": 60.0,
                "visibility_km": 9.0,
                "cloud_base_ft": 1800.0,
                "cloud_ceiling_ft": 2000.0,
                "precipitation_level": 0,
            }
            for offset in range(6)
        ]

        with patch(
            "decideflight.api.weather._fetch_grid_point_weather",
            new=AsyncMock(side_effect=hourly_points),
        ):
            response = client.post(
                f"/api/v1/weather/report/{report_id}/forecast-change"
            )

        assert response.status_code == 200
        body = response.json()
        assert len(body["hours"]) == 6
        assert body["hours"][0]["offset"] == 1
        assert body["hours"][-1]["wind_kt"] == pytest.approx(13.0)

    def test_flight_plan_endpoint_returns_best_window(self, client: TestClient):
        today = datetime.now(timezone.utc).date()
        payload = {
            "hourly": {
                "time": [
                    f"{today.isoformat()}T09:00",
                    f"{today.isoformat()}T10:00",
                    f"{today.isoformat()}T11:00",
                ],
                "temperature_2m": [20.0, 20.0, 20.0],
                "relative_humidity_2m": [50.0, 50.0, 50.0],
                "dew_point_2m": [10.0, 10.0, 10.0],
                "wind_speed_10m": [10.0, 12.0, 35.0],
                "precipitation": [0.0, 0.0, 0.0],
                "visibility": [10000.0, 10000.0, 10000.0],
                "cloud_cover": [10.0, 20.0, 30.0],
            }
        }
        with patch(
            "decideflight.api.weather._fetch_open_meteo_hourly",
            new=AsyncMock(return_value=payload),
        ):
            response = client.post(
                "/api/v1/weather/flight-plan",
                json={
                    "lat": 41.0,
                    "lon": 28.9,
                    "date_offset": 0,
                    "start_hour": 9,
                    "end_hour": 11,
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert [hour["hour"] for hour in body["hours"]] == [9, 10, 11]
        assert body["best_window"] == {
            "start_hour": 9,
            "end_hour": 10,
            "length": 2,
        }

    def test_history_endpoint_lists_recent_reports(self, client: TestClient):
        _create_report(client, location="Geçmiş A")
        _create_report(client, location="Geçmiş B")

        response = client.get("/api/v1/weather/history?limit=30")

        assert response.status_code == 200
        reports = response.json()["reports"]
        locations = {report["location"] for report in reports}
        assert "Geçmiş A" in locations
        assert "Geçmiş B" in locations

    def test_seasonal_endpoint_returns_twelve_months(self, client: TestClient):
        payload = {
            "monthly": {
                "wind_speed_10m": [12.0] * 12,
                "visibility": [10.0] * 12,
            }
        }

        class _MockAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, *args, **kwargs):
                return _MockResponse(payload)

        with patch(
            "decideflight.api.weather.httpx.AsyncClient",
            return_value=_MockAsyncClient(),
        ):
            response = client.post(
                "/api/v1/weather/seasonal",
                json={"lat": 41.0, "lon": 28.9},
            )

        assert response.status_code == 200
        months = response.json()["months"]
        assert len(months) == 12
        assert months[0]["month"] == "Ocak"


@pytest.mark.asyncio
async def test_fetch_air_quality_parses_open_meteo_payload():
    async with httpx.AsyncClient() as client:
        client.get = AsyncMock(
            return_value=_MockResponse(
                {
                    "current": {
                        "european_aqi": 87,
                        "pm2_5": 12.5,
                        "pm10": 21.0,
                    }
                }
            )
        )
        result = await _fetch_air_quality(41.0, 28.9, client)

    assert result is not None
    assert result.aqi_score == 87
    assert result.pm25 == pytest.approx(12.5)
    assert result.pm10 == pytest.approx(21.0)


@pytest.mark.asyncio
async def test_fetch_mgm_parses_turkish_payloads():
    current_payload = [
        {
            "istNo": 17110,
            "enlem": 41.01,
            "boylam": 28.97,
            "ruzgarHiz": 5.5,
            "sicaklik": 24.0,
            "nem": 44,
            "gorus": 15000,
            "yagis1Saat": 0,
        }
    ]
    hourly_payload = {
        "records": [
            {
                "istNo": 17110,
                "ruzgarHiz": 6.0,
                "sicaklik": 23.5,
                "nem": 46,
                "gorus": 12000,
            }
        ]
    }

    async with httpx.AsyncClient() as client:
        client.get = AsyncMock(
            side_effect=[
                _MockResponse(current_payload),
                _MockResponse(hourly_payload),
            ]
        )
        result = await _fetch_mgm(41.0, 28.9, client)

    assert result is not None
    assert result.source == "MGM (Türkiye)"
    assert result.temperature_c == pytest.approx(24.0)
    assert result.humidity_pct == pytest.approx(44.0)
    assert result.visibility_km == pytest.approx(15.0)
    assert result.wind_speed_knots == pytest.approx(5.5 * 1.94384)

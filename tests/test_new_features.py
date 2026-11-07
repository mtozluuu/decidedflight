from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from decideflight.main import app
from decideflight.services.weather_fetcher import (
    AirQualityData,
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
    def test_source_status_reports_received_and_unconfigured_sources(
        self, client: TestClient
    ):
        with (
            patch(
                "decideflight.api.weather.fetch_all_sources",
                new=AsyncMock(return_value=[_source()]),
            ),
            patch(
                "decideflight.api.weather.fetch_air_quality",
                new=AsyncMock(
                    return_value=AirQualityData(aqi_score=42, pm25=4.0, pm10=8.0)
                ),
            ),
            patch(
                "decideflight.api.weather.fetch_wind_trend",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "decideflight.api.weather.settings",
                new=SimpleNamespace(
                    openweathermap_api_key="",
                    weatherapi_api_key="",
                    windy_api_key="",
                    checkwx_api_key="",
                ),
            ),
        ):
            response = client.get(
                "/api/v1/weather/sources/status?lat=41&lon=29"
            )

        assert response.status_code == 200
        checks = {
            (check["provider"], check["name"]): check
            for check in response.json()["checks"]
        }
        assert checks[("Open-Meteo", "Anlık hava durumu")]["status"] == "received"
        assert checks[("Open-Meteo", "Hava kalitesi")]["status"] == (
            "received"
        )
        assert checks[("Open-Meteo", "Rüzgar ve tahmin")]["status"] == "no_data"
        assert checks[("NOAA Aviation Weather", "METAR")]["status"] == "no_data"
        assert len(checks) == 8
        for provider in ("OpenWeatherMap", "WeatherAPI", "Windy", "CheckWX"):
            assert checks[(provider, next(
                check["name"]
                for check in response.json()["checks"]
                if check["provider"] == provider
            ))]["status"] == "not_configured"
        assert not any(
            provider == "MGM"
            for provider, _ in checks
        )

    def test_source_status_uses_configured_optional_providers(self, client: TestClient):
        sources = [
            _source(),
            WeatherSourceData(
                source="OpenWeatherMap",
                wind_speed_knots=9.0,
                temperature_c=22.0,
                humidity_pct=55.0,
                visibility_km=10.0,
                precipitation_level=PRECIP_NONE,
                cloud_base_ft=2500.0,
                cloud_ceiling_ft=3200.0,
            ),
            WeatherSourceData(
                source="WeatherAPI",
                wind_speed_knots=9.0,
                temperature_c=22.0,
                humidity_pct=55.0,
                visibility_km=10.0,
                precipitation_level=PRECIP_NONE,
                cloud_base_ft=2500.0,
                cloud_ceiling_ft=3200.0,
            ),
            WeatherSourceData(
                source="Windy",
                wind_speed_knots=9.0,
                temperature_c=22.0,
                humidity_pct=55.0,
                visibility_km=10.0,
                precipitation_level=PRECIP_NONE,
                cloud_base_ft=2500.0,
                cloud_ceiling_ft=3200.0,
            ),
            WeatherSourceData(
                source="METAR (CheckWX)",
                wind_speed_knots=9.0,
                temperature_c=22.0,
                humidity_pct=55.0,
                visibility_km=10.0,
                precipitation_level=PRECIP_NONE,
                cloud_base_ft=None,
                cloud_ceiling_ft=None,
            ),
        ]
        with (
            patch(
                "decideflight.api.weather.fetch_all_sources",
                new=AsyncMock(return_value=sources),
            ),
            patch(
                "decideflight.api.weather.fetch_air_quality",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "decideflight.api.weather.fetch_wind_trend",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "decideflight.api.weather.settings",
                new=SimpleNamespace(
                    openweathermap_api_key="configured",
                    weatherapi_api_key="configured",
                    windy_api_key="configured",
                    checkwx_api_key="configured",
                ),
            ),
        ):
            response = client.get(
                "/api/v1/weather/sources/status?lat=41&lon=29"
            )

        assert response.status_code == 200
        checks = {
            (check["provider"], check["name"]): check["status"]
            for check in response.json()["checks"]
        }
        for provider, name in (
            ("OpenWeatherMap", "Anlık hava durumu"),
            ("WeatherAPI", "Anlık hava durumu"),
            ("Windy", "Model tahmini"),
            ("CheckWX", "METAR"),
        ):
            assert checks[(provider, name)] == "received"

    def test_source_status_rejects_invalid_coordinates(self, client: TestClient):
        response = client.get(
            "/api/v1/weather/sources/status?lat=91&lon=29"
        )

        assert response.status_code == 422

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
        from datetime import timedelta

        start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        payload = {"hourly": {
            "time": [(start + timedelta(hours=i)).isoformat() for i in range(7)],
            "temperature_2m": [20.0] * 7,
            "wind_speed_10m": [i / 0.539957 for i in range(7, 14)],
            "relative_humidity_2m": [60.0] * 7,
            "visibility": [9000.0] * 7,
            "precipitation": [0.0] * 7,
            "cloud_cover": [10.0] * 7,
        }}

        with patch(
            "decideflight.api.weather._fetch_open_meteo_hourly",
            new=AsyncMock(return_value=payload),
        ):
            response = client.post(
                f"/api/v1/weather/report/{report_id}/forecast-change"
            )

        assert response.status_code == 200
        body = response.json()
        assert len(body["hours"]) == 6
        assert body["hours"][0]["offset"] == 1
        assert body["hours"][-1]["wind_kt"] == pytest.approx(13.0)
        assert datetime.fromisoformat(body["hours"][0]["time"]) == start + timedelta(hours=1)
        assert body["missing_offsets"] == []

    def test_flight_plan_endpoint_returns_best_window(self, client: TestClient):
        from datetime import timedelta

        today = datetime.now(timezone.utc).date() + timedelta(days=1)
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
                    "date_offset": 1,
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

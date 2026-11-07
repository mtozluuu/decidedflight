from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from decideflight.services.weather_fetcher import (
    PRECIP_NONE,
    WeatherSourceData,
    _AirportRecord,
    _fetch_metar_aviationweather,
    _fetch_metar_checkwx,
    _fetch_windy,
    fetch_all_sources,
    fetch_nearby_runways,
)


class _MockResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _MockAsyncClient:
    def __init__(self, response):
        self.get = AsyncMock(return_value=response)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _airport() -> _AirportRecord:
    return _AirportRecord(
        icao="HCMM",
        name="Aden Adde International Airport",
        latitude_deg=2.01444,
        longitude_deg=45.304699,
    )


@pytest.mark.asyncio
async def test_fetch_windy_logs_error_body_without_api_key(caplog):
    api_key = "windy-test-secret"
    response = httpx.Response(
        400,
        text=f'Invalid request for key "{api_key}"',
        request=httpx.Request(
            "POST", "https://api.windy.com/api/point-forecast/v2"
        ),
    )
    client = SimpleNamespace(post=AsyncMock(return_value=response))

    with patch(
        "decideflight.services.weather_fetcher.settings",
        new=SimpleNamespace(windy_api_key=api_key),
    ):
        result = await _fetch_windy(39.0, 35.0, client)

    assert result is None
    assert "HTTP 400" in caplog.text
    assert "[REDACTED]" in caplog.text
    assert api_key not in caplog.text


@pytest.mark.asyncio
async def test_fetch_nearby_runways_falls_back_to_ourairports():
    with (
        patch(
            "decideflight.services.weather_fetcher.settings",
            new=SimpleNamespace(avwx_api_key="k"),
        ),
        patch(
            "decideflight.services.weather_fetcher._OURAIRPORTS_AIRPORTS",
            new=[_airport()],
        ),
        patch(
            "decideflight.services.weather_fetcher._OURAIRPORTS_RUNWAYS",
            new={
                "HCMM": [
                    {"runway_ident": "05", "heading_true": 59.0},
                    {"runway_ident": "23", "heading_true": 239.0},
                ]
            },
        ),
        patch(
            "decideflight.services.weather_fetcher.httpx.AsyncClient",
            return_value=_MockAsyncClient(_MockResponse([])),
        ),
    ):
        result = await fetch_nearby_runways(2.01444, 45.304699)

    assert result == [
        {
            "airport_icao": "HCMM",
            "airport_name": "Aden Adde International Airport",
            "runway_ident": "05",
            "heading_true": 59.0,
            "distance_km": 0.0,
        },
        {
            "airport_icao": "HCMM",
            "airport_name": "Aden Adde International Airport",
            "runway_ident": "23",
            "heading_true": 239.0,
            "distance_km": 0.0,
        },
    ]


@pytest.mark.asyncio
async def test_fetch_nearby_runways_fills_null_bearing_from_ourairports():
    """AVWX bearing1/bearing2 null → filled from OurAirports CSV."""
    avwx_payload = [
        {
            "kilometers": 4.3,
            "station": {
                "icao": "HCMM",
                "name": "Aden Adde International Airport",
                "runways": [
                    {
                        "ident1": "05",
                        "ident2": "23",
                        "bearing1": None,
                        "bearing2": None,
                    }
                ],
            },
        }
    ]
    with (
        patch(
            "decideflight.services.weather_fetcher.settings",
            new=SimpleNamespace(avwx_api_key="k"),
        ),
        patch(
            "decideflight.services.weather_fetcher._OURAIRPORTS_RUNWAYS",
            new={
                "HCMM": [
                    {"runway_ident": "05", "heading_true": 50.0},
                    {"runway_ident": "23", "heading_true": 230.0},
                ]
            },
        ),
        patch(
            "decideflight.services.weather_fetcher.httpx.AsyncClient",
            return_value=_MockAsyncClient(_MockResponse(avwx_payload)),
        ),
    ):
        result = await fetch_nearby_runways(2.01444, 45.304699)

    assert result == [
        {
            "airport_icao": "HCMM",
            "airport_name": "Aden Adde International Airport",
            "runway_ident": "05",
            "heading_true": 50.0,
            "distance_km": 4.3,
        },
        {
            "airport_icao": "HCMM",
            "airport_name": "Aden Adde International Airport",
            "runway_ident": "23",
            "heading_true": 230.0,
            "distance_km": 4.3,
        },
    ]


@pytest.mark.asyncio
async def test_fetch_nearby_runways_prefers_hardcoded_bearing_override():
    avwx_payload = [
        {
            "kilometers": 4.3,
            "station": {
                "icao": "HCMM",
                "name": "Aden Adde International Airport",
                "runways": [
                    {
                        "ident1": "05",
                        "ident2": "23",
                        "bearing1": None,
                        "bearing2": None,
                    }
                ],
            },
        }
    ]
    with (
        patch(
            "decideflight.services.weather_fetcher.settings",
            new=SimpleNamespace(avwx_api_key="k"),
        ),
        patch(
            "decideflight.services.weather_fetcher._OURAIRPORTS_RUNWAYS",
            new={
                "HCMM": [
                    {"runway_ident": "05", "heading_true": 13.0},
                    {"runway_ident": "23", "heading_true": 30.0},
                ]
            },
        ),
        patch(
            "decideflight.services.weather_fetcher.httpx.AsyncClient",
            return_value=_MockAsyncClient(_MockResponse(avwx_payload)),
        ),
    ):
        result = await fetch_nearby_runways(2.01444, 45.304699)

    assert result == [
        {
            "airport_icao": "HCMM",
            "airport_name": "Aden Adde International Airport",
            "runway_ident": "05",
            "heading_true": 50.0,
            "distance_km": 4.3,
        },
        {
            "airport_icao": "HCMM",
            "airport_name": "Aden Adde International Airport",
            "runway_ident": "23",
            "heading_true": 230.0,
            "distance_km": 4.3,
        },
    ]


@pytest.mark.asyncio
async def test_fetch_metar_aviationweather_parses_payload(monkeypatch):
    monkeypatch.setattr(
        "decideflight.services.weather_fetcher._OURAIRPORTS_AIRPORTS",
        [_airport()],
    )

    async with _MockAsyncClient(
        _MockResponse(
            [
                {
                    "wspd": 14,
                    "wdir": 180,
                    "visib": "6.0",
                    "temp": 29,
                    "dwpt": 24,
                }
            ]
        )
    ) as client:
        result = await _fetch_metar_aviationweather(2.0, 45.3, client)

    assert result is not None
    assert result.source == "METAR (NOAA/aviationweather.gov)"
    assert result.wind_speed_knots == pytest.approx(14.0)
    assert result.wind_direction_deg == pytest.approx(180.0)
    assert result.visibility_km == pytest.approx(9.656064)
    assert result.temperature_c == pytest.approx(29.0)
    assert result.precipitation_level == PRECIP_NONE


@pytest.mark.asyncio
async def test_fetch_metar_checkwx_parses_payload(monkeypatch):
    monkeypatch.setattr(
        "decideflight.services.weather_fetcher._OURAIRPORTS_AIRPORTS",
        [_airport()],
    )

    payload = {
        "data": [
            {
                "wind": {"speed_kts": 18, "degrees": 220},
                "visibility": {"meters": 8000},
                "temperature": {"celsius": 28},
                "dewpoint": {"celsius": 24},
            }
        ]
    }

    async with _MockAsyncClient(_MockResponse(payload)) as client:
        with patch(
            "decideflight.services.weather_fetcher.settings",
            new=SimpleNamespace(checkwx_api_key="secret"),
        ):
            result = await _fetch_metar_checkwx(2.0, 45.3, client)

    assert result is not None
    assert result.source == "METAR (CheckWX)"
    assert result.wind_speed_knots == pytest.approx(18.0)
    assert result.wind_direction_deg == pytest.approx(220.0)
    assert result.visibility_km == pytest.approx(8.0)
    assert result.temperature_c == pytest.approx(28.0)
    assert result.precipitation_level == PRECIP_NONE


@pytest.mark.asyncio
async def test_fetch_all_sources_uses_active_providers_but_not_mgm():
    open_meteo = WeatherSourceData(
        source="Open-Meteo",
        wind_speed_knots=9.0,
        temperature_c=22.0,
        humidity_pct=55.0,
        visibility_km=10.0,
        precipitation_level=PRECIP_NONE,
        cloud_base_ft=2500.0,
        cloud_ceiling_ft=3200.0,
    )
    noaa = WeatherSourceData(
        source="METAR (NOAA/aviationweather.gov)",
        wind_speed_knots=12.0,
        temperature_c=24.0,
        humidity_pct=70.0,
        visibility_km=10.0,
        precipitation_level=PRECIP_NONE,
        cloud_base_ft=None,
        cloud_ceiling_ft=None,
    )

    optional_sources = [
        replace(open_meteo, source="OpenWeatherMap"),
        replace(open_meteo, source="WeatherAPI"),
        replace(open_meteo, source="Windy"),
        replace(noaa, source="METAR (CheckWX)"),
    ]

    with (
        patch(
            "decideflight.services.weather_fetcher._fetch_open_meteo",
            new=AsyncMock(return_value=open_meteo),
        ),
        patch(
            "decideflight.services.weather_fetcher._fetch_owm",
            new=AsyncMock(return_value=optional_sources[0]),
        ),
        patch(
            "decideflight.services.weather_fetcher._fetch_weatherapi",
            new=AsyncMock(return_value=optional_sources[1]),
        ),
        patch(
            "decideflight.services.weather_fetcher._fetch_windy",
            new=AsyncMock(return_value=optional_sources[2]),
        ),
        patch(
            "decideflight.services.weather_fetcher._fetch_metar_aviationweather",
            new=AsyncMock(return_value=noaa),
        ),
        patch(
            "decideflight.services.weather_fetcher._fetch_metar_checkwx",
            new=AsyncMock(return_value=optional_sources[3]),
        ),
        patch(
            "decideflight.services.weather_fetcher._fetch_mgm",
            new=AsyncMock(
                return_value=replace(open_meteo, source="MGM (Türkiye)")
            ),
        ) as fetch_mgm,
    ):
        result = await fetch_all_sources(2.0, 45.3)

    assert [source.source for source in result] == [
        "Open-Meteo",
        "OpenWeatherMap",
        "WeatherAPI",
        "Windy",
        "METAR (NOAA/aviationweather.gov)",
        "METAR (CheckWX)",
    ]
    fetch_mgm.assert_not_awaited()

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from decideflight.services.weather_fetcher import (
    PRECIP_NONE,
    WeatherSourceData,
    _AirportRecord,
    _fetch_metar_aviationweather,
    _fetch_metar_checkwx,
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
async def test_fetch_all_sources_includes_new_metar_sources():
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
    checkwx = WeatherSourceData(
        source="METAR (CheckWX)",
        wind_speed_knots=11.0,
        temperature_c=23.0,
        humidity_pct=72.0,
        visibility_km=9.0,
        precipitation_level=PRECIP_NONE,
        cloud_base_ft=None,
        cloud_ceiling_ft=None,
    )

    with (
        patch(
            "decideflight.services.weather_fetcher._fetch_open_meteo",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "decideflight.services.weather_fetcher._fetch_owm",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "decideflight.services.weather_fetcher._fetch_weatherapi",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "decideflight.services.weather_fetcher._fetch_mgm",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "decideflight.services.weather_fetcher._fetch_windy",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "decideflight.services.weather_fetcher._fetch_metar_avwx",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "decideflight.services.weather_fetcher._fetch_metar_aviationweather",
            new=AsyncMock(return_value=noaa),
        ),
        patch(
            "decideflight.services.weather_fetcher._fetch_metar_checkwx",
            new=AsyncMock(return_value=checkwx),
        ),
    ):
        result = await fetch_all_sources(2.0, 45.3)

    assert [source.source for source in result] == [
        "METAR (NOAA/aviationweather.gov)",
        "METAR (CheckWX)",
    ]

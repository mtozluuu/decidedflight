from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from decideflight.services.metar_fetcher import (
    MetarTafData,
    fetch_metar_taf_for_coords,
    fetch_metar_taf_for_location,
)


class _MockResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_fetch_metar_taf_for_coords_parses_avwx_payload():
    async with httpx.AsyncClient() as client:
        with patch(
            "decideflight.services.metar_fetcher.settings",
            new=SimpleNamespace(avwx_api_key="k"),
        ):
            client.get = AsyncMock(
                side_effect=[
                    _MockResponse({"stations": [{"icao": "LTFM"}]}),
                    _MockResponse(
                        {
                            "raw": "LTFM 241200Z 18015KT 9999 FEW030 25/14 Q1013",
                            "wind_speed": {"value": 15},
                            "wind_direction": {"value": 180},
                            "visibility": {"value": 9999, "repr": "9999", "units": "m"},
                            "clouds": [{"type": "FEW", "base": {"value": 3000}}],
                            "temperature": {"value": 25},
                            "dewpoint": {"value": 14},
                        }
                    ),
                    _MockResponse(
                        {
                            "forecast": [
                                {"sanitized": "BECMG 18020KT"},
                                {"sanitized": "TEMPO 3000 SHRA"},
                            ]
                        }
                    ),
                ]
            )

            result = await fetch_metar_taf_for_coords(41.0, 28.9, client=client)

    assert isinstance(result, MetarTafData)
    assert result.icao == "LTFM"
    assert result.wind_speed_kt == pytest.approx(15.0)
    assert result.visibility_km == pytest.approx(9.999)
    assert result.cloud_base_ft == pytest.approx(3000.0)
    assert result.temperature_c == pytest.approx(25.0)
    assert result.dewpoint_c == pytest.approx(14.0)
    assert "LTFM 241200Z" in result.raw_metar
    assert result.taf_summary_next_6h == "BECMG 18020KT | TEMPO 3000 SHRA"


@pytest.mark.asyncio
async def test_fetch_metar_taf_for_location_uses_geocoding():
    async with httpx.AsyncClient() as client:
        with (
            patch(
                "decideflight.services.metar_fetcher.settings",
                new=SimpleNamespace(avwx_api_key="k"),
            ),
            patch(
                "decideflight.services.metar_fetcher.geocode_city",
                new=AsyncMock(return_value=(41.0, 28.9, "Istanbul, Turkey")),
            ),
            patch(
                "decideflight.services.metar_fetcher.fetch_metar_taf_for_coords",
                new=AsyncMock(
                    return_value=MetarTafData(
                        icao="LTFM",
                        wind_speed_kt=10.0,
                        visibility_text="9999",
                        visibility_km=10.0,
                        cloud_base_ft=3000.0,
                        temperature_c=24.0,
                        dewpoint_c=14.0,
                        raw_metar="METAR TEXT",
                        taf_summary_next_6h="TAF SUMMARY",
                    )
                ),
            ) as mocked_fetch,
        ):
            result = await fetch_metar_taf_for_location("Istanbul", client=client)

    assert isinstance(result, MetarTafData)
    mocked_fetch.assert_awaited_once_with(lat=41.0, lon=28.9, client=client)

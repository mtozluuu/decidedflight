from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from decideflight.main import app
from decideflight.services.mgm_observations import (
    _ObservationParser,
    fetch_mgm_observations,
)


def test_parser_reads_latest_requested_station_and_ignores_scripts():
    parser = _ObservationParser("LTAJ")
    parser.feed(
        '<pre>METAR LTAC 071650Z CAVOK</pre>'
        '<pre>METAR LTAJ 071650Z VRB02KT <span>CAVOK</span> 12/06</pre>'
        '<pre>METAR LTAJ 071550Z CAVOK</pre>'
        '<pre>TAF AMD LTAJ 071640Z 0718/0818 CAVOK</pre>'
        '<script>METAR LTAJ 071750Z FAKE</script>'
    )
    assert parser.observations == {
        "metar": "METAR LTAJ 071650Z VRB02KT CAVOK 12/06",
        "taf": "TAF AMD LTAJ 071640Z 0718/0818 CAVOK",
    }


@pytest.mark.asyncio
async def test_fetch_uses_latest_mgm_query():
    response = httpx.Response(
        200, text="<pre>METAR LTAJ 071650Z CAVOK</pre>",
        request=httpx.Request("GET", "https://rasat.mgm.gov.tr/result"),
    )
    with patch("decideflight.services.mgm_observations.httpx.AsyncClient") as factory:
        client = factory.return_value.__aenter__.return_value
        client.get = AsyncMock(return_value=response)
        result = await fetch_mgm_observations("LTAJ")
    assert result == {"metar": "METAR LTAJ 071650Z CAVOK", "taf": None}
    assert client.get.call_args.kwargs["params"] == [
        ("stations", "LTAJ"), ("obsType", "1"), ("obsType", "2"), ("hours", "0"),
    ]


def test_endpoint_returns_observations_and_check_time():
    with patch(
        "decideflight.api.weather.fetch_mgm_observations",
        new=AsyncMock(return_value={"metar": "METAR LTAJ 071650Z CAVOK", "taf": None}),
    ) as fetch:
        response = TestClient(app).get("/api/v1/weather/airport-observations/ltaj")
    assert response.status_code == 200
    assert response.json()["icao"] == "LTAJ"
    assert response.json()["taf"] is None
    assert response.json()["checked_at"]
    assert response.json()["metar_timing"]["issued_at"]
    assert response.json()["taf_timing"]["status"] == "unknown"
    fetch.assert_awaited_once_with("LTAJ")


@pytest.mark.parametrize("icao", ["ABC", "1234", "LTAJJ"])
def test_endpoint_rejects_invalid_station(icao):
    with patch("decideflight.api.weather.fetch_mgm_observations") as fetch:
        response = TestClient(app).get(f"/api/v1/weather/airport-observations/{icao}")
    assert response.status_code == 422
    fetch.assert_not_called()


def test_endpoint_reports_no_observations():
    with patch(
        "decideflight.api.weather.fetch_mgm_observations",
        new=AsyncMock(return_value={"metar": None, "taf": None}),
    ):
        response = TestClient(app).get("/api/v1/weather/airport-observations/LTAJ")
    assert response.status_code == 404


def test_endpoint_reports_provider_failure():
    with patch(
        "decideflight.api.weather.fetch_mgm_observations",
        new=AsyncMock(side_effect=httpx.ConnectError("unavailable")),
    ):
        response = TestClient(app).get("/api/v1/weather/airport-observations/LTAJ")
    assert response.status_code == 503

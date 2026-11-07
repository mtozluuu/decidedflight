from unittest.mock import AsyncMock, patch

import httpx
import pytest

from decideflight.services.geocoding import geocode_city


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["Gaziantep,Turkey", "Gaziantep, Turkey", "Gaziantep, TR"])
async def test_geocoding_separates_city_and_country(location):
    response = httpx.Response(
        200,
        json={"results": [{"name": "Gaziantep", "country": "Turkey",
                           "latitude": 37.06, "longitude": 37.38}]},
        request=httpx.Request("GET", "https://geocoding-api.open-meteo.com/v1/search"),
    )
    with patch("decideflight.services.geocoding.httpx.AsyncClient") as factory:
        client = factory.return_value.__aenter__.return_value
        client.get = AsyncMock(return_value=response)
        result = await geocode_city(location)

    assert result == (37.06, 37.38, "Gaziantep, Turkey")
    assert client.get.call_args.kwargs["params"]["name"] == "Gaziantep"
    assert client.get.call_args.kwargs["params"]["countryCode"] == "TR"


@pytest.mark.asyncio
async def test_geocoding_city_only_preserves_request():
    response = httpx.Response(
        200,
        json={"results": [{"name": "Gaziantep", "country": "Turkey",
                           "latitude": 37.06, "longitude": 37.38}]},
        request=httpx.Request("GET", "https://geocoding-api.open-meteo.com/v1/search"),
    )
    with patch("decideflight.services.geocoding.httpx.AsyncClient") as factory:
        client = factory.return_value.__aenter__.return_value
        client.get = AsyncMock(return_value=response)
        await geocode_city("Gaziantep")

    assert client.get.call_args.kwargs["params"] == {
        "name": "Gaziantep", "count": 1, "language": "en", "format": "json",
    }


@pytest.mark.asyncio
async def test_geocoding_rejects_unknown_country():
    with pytest.raises(ValueError, match="Country not found"):
        await geocode_city("Gaziantep, Unknown")

from unittest.mock import patch

from fastapi.testclient import TestClient

from decideflight.main import app
from decideflight.services.weather_fetcher import _AirportRecord, airport_location


def test_selected_airport_location():
    with patch(
        "decideflight.services.weather_fetcher._OURAIRPORTS_AIRPORTS",
        [_AirportRecord("LTAJ", "Gaziantep", 36.95, 37.48)],
    ):
        assert airport_location("LTAJ") == {
            "icao": "LTAJ", "name": "Gaziantep", "lat": 36.95, "lon": 37.48,
        }
        assert airport_location("LTAC") is None


def test_airport_location_missing_and_invalid():
    client = TestClient(app)
    assert client.get("/api/v1/weather/airport-location/1234").status_code == 422
    with patch("decideflight.api.weather.airport_location", return_value=None):
        assert client.get("/api/v1/weather/airport-location/ZZZZ").status_code == 404


def test_airport_location_normalizes_icao():
    with patch(
        "decideflight.api.weather.airport_location",
        return_value={"icao": "LTAC", "name": "Ankara", "lat": 40, "lon": 33},
    ) as lookup:
        response = TestClient(app).get("/api/v1/weather/airport-location/ltac")
    assert response.status_code == 200
    lookup.assert_called_once_with("LTAC")

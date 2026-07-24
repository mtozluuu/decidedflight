"""Tests for the weather report feature."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from decideflight.main import app
from decideflight.services.decision_engine import (
    RISKLI,
    UYGUN,
    UYGUN_DEGIL,
    make_decision,
)
from decideflight.services.weather_fetcher import (
    PRECIP_HEAVY,
    PRECIP_LIGHT,
    PRECIP_NONE,
    WeatherSourceData,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _make_source(
    source: str = "Test",
    wind_knots: float = 10.0,
    temp_c: float = 20.0,
    humidity: float = 60.0,
    vis_km: float = 8.0,
    precip: int = PRECIP_NONE,
    cloud_base: float | None = 800.0,
    cloud_ceiling: float | None = 1200.0,
) -> WeatherSourceData:
    return WeatherSourceData(
        source=source,
        wind_speed_knots=wind_knots,
        temperature_c=temp_c,
        humidity_pct=humidity,
        visibility_km=vis_km,
        precipitation_level=precip,
        cloud_base_ft=cloud_base,
        cloud_ceiling_ft=cloud_ceiling,
    )


# ---------------------------------------------------------------------------
# Decision engine unit tests
# ---------------------------------------------------------------------------


class TestDecisionEngine:
    def test_all_ok(self):
        result = make_decision([_make_source()])
        assert result.decision == UYGUN
        assert "Tüm parametreler uygun" in result.detail

    def test_wind_makes_risky(self):
        result = make_decision([_make_source(wind_knots=20.0)])
        assert result.decision == RISKLI

    def test_wind_makes_not_suitable(self):
        result = make_decision([_make_source(wind_knots=30.0)])
        assert result.decision == UYGUN_DEGIL

    def test_low_visibility_risky(self):
        result = make_decision([_make_source(vis_km=3.0)])
        assert result.decision == RISKLI

    def test_very_low_visibility_not_suitable(self):
        result = make_decision([_make_source(vis_km=0.5)])
        assert result.decision == UYGUN_DEGIL

    def test_heavy_precipitation_not_suitable(self):
        result = make_decision([_make_source(precip=PRECIP_HEAVY)])
        assert result.decision == UYGUN_DEGIL

    def test_light_precipitation_risky(self):
        result = make_decision([_make_source(precip=PRECIP_LIGHT)])
        assert result.decision == RISKLI

    def test_high_humidity_risky(self):
        result = make_decision([_make_source(humidity=90.0)])
        assert result.decision == RISKLI

    def test_very_high_humidity_not_suitable(self):
        result = make_decision([_make_source(humidity=96.0)])
        assert result.decision == UYGUN_DEGIL

    def test_low_cloud_base_risky(self):
        result = make_decision([_make_source(cloud_base=300.0, cloud_ceiling=600.0)])
        assert result.decision == RISKLI

    def test_very_low_cloud_base_not_suitable(self):
        result = make_decision([_make_source(cloud_base=100.0, cloud_ceiling=400.0)])
        assert result.decision == UYGUN_DEGIL

    def test_temperature_too_cold_not_suitable(self):
        result = make_decision([_make_source(temp_c=-10.0)])
        assert result.decision == UYGUN_DEGIL

    def test_temperature_slightly_cold_risky(self):
        result = make_decision([_make_source(temp_c=-3.0)])
        assert result.decision == RISKLI

    def test_temperature_too_hot_not_suitable(self):
        result = make_decision([_make_source(temp_c=50.0)])
        assert result.decision == UYGUN_DEGIL

    def test_multiple_sources_averaged(self):
        # Average wind = (10 + 20) / 2 = 15 → boundary → RISKLI
        sources = [_make_source(wind_knots=10.0), _make_source(wind_knots=20.0)]
        result = make_decision(sources)
        assert result.avg_wind_knots == pytest.approx(15.0)

    def test_worst_precipitation_used(self):
        # One source says no rain, another says heavy
        sources = [
            _make_source(precip=PRECIP_NONE),
            _make_source(precip=PRECIP_HEAVY),
        ]
        result = make_decision(sources)
        assert result.decision == UYGUN_DEGIL

    def test_none_cloud_base_treated_as_ok(self):
        result = make_decision([_make_source(cloud_base=None, cloud_ceiling=None)])
        assert result.decision == UYGUN

    def test_parameters_list_length(self):
        result = make_decision([_make_source()])
        assert len(result.parameters) == 7


# ---------------------------------------------------------------------------
# PDF generation smoke test
# ---------------------------------------------------------------------------


class TestReportGenerator:
    def test_pdf_output_is_valid(self):
        from decideflight.services.report_generator import generate_pdf

        sources = [_make_source()]
        dr = make_decision(sources)
        pdf = generate_pdf("Test City", 41.0, 28.9, sources, dr)
        assert pdf[:4] == b"%PDF", "Output should be a valid PDF"
        assert len(pdf) > 1000

    def test_pdf_risky_decision(self):
        from decideflight.services.report_generator import generate_pdf

        sources = [_make_source(wind_knots=20.0)]
        dr = make_decision(sources)
        pdf = generate_pdf("Test City", 41.0, 28.9, sources, dr)
        assert pdf[:4] == b"%PDF"

    def test_pdf_not_suitable_decision(self):
        from decideflight.services.report_generator import generate_pdf

        sources = [_make_source(wind_knots=30.0, vis_km=0.5)]
        dr = make_decision(sources)
        pdf = generate_pdf("Test City", 41.0, 28.9, sources, dr)
        assert pdf[:4] == b"%PDF"


# ---------------------------------------------------------------------------
# API endpoint tests (mocked external calls)
# ---------------------------------------------------------------------------


class TestWeatherReportAPI:
    def _sources_payload(self) -> list[WeatherSourceData]:
        return [_make_source(source="Open-Meteo")]

    def test_create_report_with_coordinates(self, client: TestClient):
        sources = self._sources_payload()
        with patch(
            "decideflight.api.weather.fetch_all_sources",
            new=AsyncMock(return_value=sources),
        ):
            response = client.post(
                "/api/v1/weather/report",
                json={"lat": 41.0, "lon": 28.9},
            )
        assert response.status_code == 201
        body = response.json()
        assert "report_id" in body
        assert body["decision"] in (UYGUN, RISKLI, UYGUN_DEGIL)
        assert len(body["sources"]) == 1
        assert body["lat"] == pytest.approx(41.0)
        assert body["lon"] == pytest.approx(28.9)

    def test_create_report_with_city_name(self, client: TestClient):
        sources = self._sources_payload()
        with (
            patch(
                "decideflight.api.weather.geocode_city",
                new=AsyncMock(return_value=(41.0, 28.9, "Istanbul, Turkey")),
            ),
            patch(
                "decideflight.api.weather.fetch_all_sources",
                new=AsyncMock(return_value=sources),
            ),
        ):
            response = client.post(
                "/api/v1/weather/report",
                json={"location": "Istanbul"},
            )
        assert response.status_code == 201
        body = response.json()
        assert body["location"] == "Istanbul, Turkey"

    def test_create_report_missing_location(self, client: TestClient):
        response = client.post("/api/v1/weather/report", json={})
        assert response.status_code == 422

    def test_create_report_city_not_found(self, client: TestClient):
        with patch(
            "decideflight.api.weather.geocode_city",
            new=AsyncMock(side_effect=ValueError("City not found")),
        ):
            response = client.post(
                "/api/v1/weather/report",
                json={"location": "NonExistentCityXYZ"},
            )
        assert response.status_code == 422

    def test_download_pdf_for_existing_report(self, client: TestClient):
        sources = self._sources_payload()
        # First create a report
        with patch(
            "decideflight.api.weather.fetch_all_sources",
            new=AsyncMock(return_value=sources),
        ):
            create_resp = client.post(
                "/api/v1/weather/report",
                json={"lat": 41.0, "lon": 28.9},
            )
        assert create_resp.status_code == 201
        report_id = create_resp.json()["report_id"]

        # Then download the PDF
        pdf_resp = client.get(f"/api/v1/weather/report/{report_id}/pdf")
        assert pdf_resp.status_code == 200
        assert pdf_resp.headers["content-type"] == "application/pdf"
        assert pdf_resp.content[:4] == b"%PDF"

    def test_download_pdf_not_found(self, client: TestClient):
        response = client.get("/api/v1/weather/report/999999/pdf")
        assert response.status_code == 404

    def test_report_response_has_parameters(self, client: TestClient):
        sources = self._sources_payload()
        with patch(
            "decideflight.api.weather.fetch_all_sources",
            new=AsyncMock(return_value=sources),
        ):
            response = client.post(
                "/api/v1/weather/report",
                json={"lat": 41.0, "lon": 28.9},
            )
        body = response.json()
        assert len(body["parameters"]) == 7
        param_names = [p["name"] for p in body["parameters"]]
        assert "Rüzgar hızı" in param_names
        assert "Sıcaklık" in param_names

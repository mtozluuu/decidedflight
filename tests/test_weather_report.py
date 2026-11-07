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
    VERI_YETERSIZ,
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
    cloud_cover: float | None = None,
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
        cloud_cover_pct=cloud_cover,
    )


# ---------------------------------------------------------------------------
# Decision engine unit tests
# ---------------------------------------------------------------------------


class TestDecisionEngine:
    @pytest.mark.parametrize("wind, boundary", [(15, "15–25"), (25, "15–25"), (25.1, "> 25")])
    def test_wind_explanation_matches_boundary(self, wind, boundary):
        result = make_decision([_make_source(wind_knots=wind)])
        parameter = result.parameters[0]
        assert boundary in parameter.explanation
        assert parameter.explanation in result.detail

    def test_explanations_use_custom_limits_without_changing_decisions(self):
        from decideflight.services.flight_limits import DroneLimit

        limits = DroneLimit(wind_ok_max_knots=8, wind_risky_max_knots=12)
        result = make_decision([_make_source(wind_knots=10)], limits=limits)
        assert result.decision == RISKLI
        assert "8–12" in result.parameters[0].explanation

    def test_missing_cloud_data_explanation_is_not_current_or_safe_claim(self):
        result = make_decision([_make_source(cloud_base=None, cloud_ceiling=None)])
        assert result.decision == VERI_YETERSIZ
        clouds = [p for p in result.parameters if p.value == "N/A"]
        assert len(clouds) == 2
        assert all("Kullanılabilir veri yok" in p.explanation for p in clouds)

    @pytest.mark.parametrize(
        "cover, expected", [(0, UYGUN), (50, UYGUN), (50.1, RISKLI), (100, RISKLI)]
    )
    def test_cloud_cover_thresholds(self, cover, expected):
        result = make_decision([_make_source(cloud_cover=cover)])
        parameter = next(p for p in result.parameters if p.name == "Bulutluluk")
        assert result.decision == expected
        assert parameter.decision == expected
        assert parameter.value == f"{cover:.1f}%"
        assert result.avg_cloud_cover_pct == cover

    def test_cloud_cover_weighted_average_excludes_missing(self):
        first = _make_source(cloud_cover=40)
        second = _make_source(cloud_cover=80)
        second.reliability_weight = 3
        missing = _make_source()
        missing.reliability_weight = 10
        result = make_decision([first, second, missing])
        assert result.avg_cloud_cover_pct == 70
        assert result.decision == RISKLI

    def test_cloud_cover_does_not_override_other_unsuitable_parameters(self):
        result = make_decision([_make_source(cloud_cover=100, wind_knots=30)])
        assert result.decision == UYGUN_DEGIL

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

    def test_unknown_clouds_do_not_imply_suitable_flight(self):
        result = make_decision([_make_source(cloud_base=None, cloud_ceiling=None)])
        assert result.decision == VERI_YETERSIZ

    def test_parameters_list_length(self):
        result = make_decision([_make_source()])
        assert len(result.parameters) == 7

    def test_metar_source_adds_detail_and_confidence_bonus(self):
        metar_source = _make_source(
            source="METAR (AVWX)",
            wind_knots=15.0,
            temp_c=25.0,
            vis_km=10.0,
            cloud_base=3000.0,
        )
        metar_source.raw = {
            "raw_metar": "LTFM 241200Z 18015KT 9999 FEW030 25/14 Q1013",
            "visibility_text": "9999m (mükemmel)",
            "taf_summary_next_6h": "BECMG 18020KT",
            "metar": {"wind_direction": {"value": 180}},
        }
        result = make_decision([metar_source])

        assert "METAR (Resmi Havacılık Gözlemi)" in result.detail
        assert "TAF (Terminal Hava Tahmini - Sonraki 6 saat)" in result.detail
        assert result.confidence_score == 100


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
    def test_pdf_keeps_saved_decision_and_custom_limit_reference(self, client):
        from decideflight.database import SessionLocal
        from decideflight.models.weather_report import WeatherReport
        from decideflight.services.report_generator import _limits_rows

        with (
            patch(
                "decideflight.api.weather.fetch_all_sources",
                new=AsyncMock(return_value=[_make_source(cloud_cover=75)]),
            ),
            patch("decideflight.api.weather.fetch_air_quality", new=AsyncMock(return_value=None)),
            patch("decideflight.api.weather.fetch_wind_trend", new=AsyncMock(return_value=None)),
        ):
            body = client.post(
                "/api/v1/weather/report",
                json={"lat": 41, "lon": 29, "limits": {"cloud_cover_ok_max_pct": 80}},
            ).json()
        with SessionLocal() as db:
            report = db.get(WeatherReport, body["report_id"])
            report.decision = UYGUN_DEGIL
            report.decision_detail = "Kaydedilmiş ihtiyatlı karar"
            db.commit()
        with patch("decideflight.api.weather.generate_pdf", return_value=b"%PDF") as generate:
            response = client.get(f'/api/v1/weather/report/{body["report_id"]}/pdf')
        assert response.status_code == 200
        args = generate.call_args.kwargs
        assert args["decision_result"].decision == UYGUN_DEGIL
        assert args["decision_result"].detail == "Kaydedilmiş ihtiyatlı karar"
        assert args["limits"].cloud_cover_ok_max_pct == 80
        row = next(row for row in _limits_rows(args["limits"]) if row[0] == "Bulutluluk")
        assert row[1] == "≤ %80"

    def test_custom_limits_are_applied_and_preserved(self, client):
        from decideflight.api.weather import _build_report_decision
        from decideflight.database import SessionLocal
        from decideflight.models.weather_report import WeatherReport

        with (
            patch(
                "decideflight.api.weather.fetch_all_sources",
                new=AsyncMock(return_value=[_make_source(cloud_cover=75)]),
            ),
            patch("decideflight.api.weather.fetch_air_quality", new=AsyncMock(return_value=None)),
            patch("decideflight.api.weather.fetch_wind_trend", new=AsyncMock(return_value=None)),
        ):
            response = client.post(
                "/api/v1/weather/report",
                json={"lat": 41, "lon": 29, "limits": {"cloud_cover_ok_max_pct": 80}},
            )
        assert response.status_code == 201
        body = response.json()
        assert body["decision"] == UYGUN
        assert body["applied_limits"]["cloud_cover_ok_max_pct"] == 80
        with SessionLocal() as db:
            report = db.get(WeatherReport, body["report_id"])
            decision, payload = _build_report_decision(report)
        assert decision.decision == UYGUN
        assert payload["applied_limits"]["cloud_cover_ok_max_pct"] == 80

    def test_report_rejects_reversed_limits(self, client):
        response = client.post(
            "/api/v1/weather/report",
            json={"lat": 41, "lon": 29,
                  "limits": {"wind_ok_max_knots": 30, "wind_risky_max_knots": 20}},
        )
        assert response.status_code == 422

    def test_report_cloud_cover_survives_storage(self, client):
        from decideflight.api.weather import _load_report_sources
        from decideflight.database import SessionLocal
        from decideflight.models.weather_report import WeatherReport

        with patch(
            "decideflight.api.weather.fetch_all_sources",
            new=AsyncMock(return_value=[_make_source(cloud_cover=75)]),
        ):
            response = client.post(
                "/api/v1/weather/report", json={"lat": 41.0, "lon": 28.9}
            )
        assert response.status_code == 201
        body = response.json()
        assert body["sources"][0]["cloud_cover_pct"] == 75
        assert next(p for p in body["parameters"] if p["name"] == "Bulutluluk") == {
            "name": "Bulutluluk", "value": "75.0%", "decision": RISKLI,
            "explanation": "> %50; tek başına UYGUN DEĞİL yapmaz",
        }
        with SessionLocal() as db:
            report = db.get(WeatherReport, body["report_id"])
            assert _load_report_sources(report)[0].cloud_cover_pct == 75

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
        with (
            patch(
                "decideflight.api.weather.fetch_all_sources",
                new=AsyncMock(return_value=sources),
            ),
            patch(
                "decideflight.api.weather.fetch_air_quality",
                new=AsyncMock(return_value=None),
            ),
        ):
            response = client.post(
                "/api/v1/weather/report",
                json={"lat": 41.0, "lon": 28.9},
            )
        body = response.json()
        assert "confidence_score" in body
        assert len(body["parameters"]) == 7
        param_names = [p["name"] for p in body["parameters"]]
        assert "Rüzgar hızı" in param_names
        assert "Sıcaklık" in param_names

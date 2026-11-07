import json
from datetime import datetime, timezone
from dataclasses import replace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from decideflight.api import weather
from decideflight.database import Base
from decideflight.main import app
from decideflight.models.weather_report import WeatherReport
from decideflight.services.ai_decision_engine import _parse_gpt_response
from decideflight.services.decision_engine import make_decision
from decideflight.services.flight_limits import DroneLimit
from decideflight.services.weather_fetcher import WeatherSourceData


@pytest.fixture
def database():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    yield sessions
    engine.dispose()


@pytest.fixture
def client(database):
    def get_db():
        with database() as db:
            yield db

    app.dependency_overrides[weather.get_db] = get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(weather.get_db)


def create(client, monkeypatch, *, limits=None, ai=False):
    source = WeatherSourceData(
        source="Open-Meteo", wind_speed_knots=10, temperature_c=20,
        humidity_pct=50, visibility_km=10, precipitation_level=0,
        cloud_base_ft=1800, cloud_ceiling_ft=2000,
        raw={"current": {"time": "2026-11-07T10:00"}},
    )
    monkeypatch.setattr(weather, "fetch_all_sources", AsyncMock(return_value=[source]))
    monkeypatch.setattr(weather, "fetch_air_quality", AsyncMock(return_value=None))
    monkeypatch.setattr(weather, "fetch_wind_trend", AsyncMock(return_value=None))
    if ai:
        result = _parse_gpt_response(
            {"karar": "UYGUN_DEGIL", "ozet": "İhtiyatlı AI kararı", "guven_skoru": 37},
            make_decision([source]),
        )
        monkeypatch.setattr(weather, "make_ai_decision", AsyncMock(return_value=result))
    response = client.post(
        "/api/v1/weather/report",
        json={"lat": 37, "lon": 37, **({"limits": limits} if limits else {})},
    )
    assert response.status_code == 201
    return response.json()


def test_saved_ai_result_is_opened_without_fetching_or_recomputing(client, monkeypatch):
    original = create(client, monkeypatch, ai=True)
    monkeypatch.setattr(weather, "fetch_all_sources", AsyncMock(side_effect=AssertionError("No refetch")))
    monkeypatch.setattr(weather, "make_decision", lambda *args, **kwargs: pytest.fail("No recomputation"))
    monkeypatch.setattr(weather, "settings", replace(weather.settings, wind_ok_max_knots=1))
    response = client.get(f'/api/v1/weather/report/{original["report_id"]}')
    assert response.status_code == 200
    saved = response.json()
    for key in (
        "decision", "decision_detail", "parameters", "confidence_score", "confidence",
        "summary", "applied_limits", "source_checks",
    ):
        assert saved[key] == original[key]
    assert saved["decision"] == "UYGUN_DEGIL"
    assert saved["is_saved_report"] is True
    assert saved["parameter_snapshot_available"] is True
    assert saved["sources"][0]["data_time"] == original["sources"][0]["data_time"]
    with patch.object(weather, "generate_pdf", return_value=b"%PDF") as generate:
        assert client.get(f'/api/v1/weather/report/{original["report_id"]}/pdf').status_code == 200
    assert generate.call_args.kwargs["decision_result"].confidence_score == 37
    assert generate.call_args.kwargs["decision_result"].parameters[0].decision == "UYGUN"
    assert generate.call_args.kwargs["source_checks"] == original["source_checks"]
    with patch.object(weather, "chat_about_report", new=AsyncMock(return_value="Kayıtlı karar")) as chat:
        response = client.post(
            f'/api/v1/weather/report/{original["report_id"]}/chat',
            json={"message": "Neden uygun değil?"},
        )
    assert response.status_code == 200
    assert chat.call_args.kwargs["decision_result"].decision == "UYGUN_DEGIL"
    assert chat.call_args.kwargs["decision_result"].confidence_score == 37
    history = client.get("/api/v1/weather/history").json()["reports"]
    item = next(item for item in history if item["report_id"] == original["report_id"])
    assert item["decision"] == "UYGUN_DEGIL"
    assert item["confidence_score"] == 37


def test_custom_limits_and_parameters_survive_changed_defaults(client, monkeypatch):
    original = create(client, monkeypatch, limits={"wind_ok_max_knots": 8, "wind_risky_max_knots": 12})
    monkeypatch.setattr(weather, "settings", replace(weather.settings, wind_ok_max_knots=30, wind_risky_max_knots=40))
    saved = client.get(f'/api/v1/weather/report/{original["report_id"]}').json()
    assert saved["applied_limits"] == original["applied_limits"]
    assert saved["parameters"] == original["parameters"]
    assert saved["decision"] == "RISKLI"


def test_insufficient_report_preserves_quality_and_missing_fields(client, monkeypatch, database):
    source = WeatherSourceData(
        source="Open-Meteo", wind_speed_knots=20, temperature_c=20,
        humidity_pct=50, visibility_km=None, precipitation_level=None,
        cloud_base_ft=1800, cloud_ceiling_ft=2000, data_kind="forecast",
        raw={"time": datetime.now(timezone.utc).replace(microsecond=0).isoformat()},
    )
    monkeypatch.setattr(weather, "fetch_all_sources", AsyncMock(return_value=[source]))
    monkeypatch.setattr(weather, "fetch_air_quality", AsyncMock(return_value=None))
    monkeypatch.setattr(weather, "fetch_wind_trend", AsyncMock(return_value=None))
    monkeypatch.setattr("decideflight.services.ai_decision_engine._AsyncOpenAI", None)
    response = client.post("/api/v1/weather/report", json={"lat": 37, "lon": 37})
    assert response.status_code == 201
    original = response.json()
    assert original["decision"] == "VERI_YETERSIZ"
    assert original["known_decision"] == "RISKLI"
    assert original["missing_parameters"] == ["Görüş mesafesi", "Yağış"]
    assert original["data_quality"]["available_parameters"] == 5
    saved = client.get(f'/api/v1/weather/report/{original["report_id"]}').json()
    for key in ("decision", "missing_parameters", "known_decision", "data_quality", "sources"):
        assert saved[key] == original[key]
    with database() as db:
        report = db.get(WeatherReport, original["report_id"])
        assert json.loads(report.sources_data)[0]["raw"] == source.raw
    assert client.get(f'/api/v1/weather/report/{original["report_id"]}/pdf').content.startswith(b"%PDF")


def test_history_filters_apply_before_limit_and_include_whole_utc_day(client, database, monkeypatch):
    reports = [create(client, monkeypatch) for _ in range(4)]
    with database() as db:
        for report, location, created_at in zip(reports, (
            "Gaziantep", "Ankara", "Gaziantep", "Gaziantep",
        ), (
            datetime(2026, 11, 1, 0, 0), datetime(2026, 11, 2, 10),
            datetime(2026, 11, 3, 23, 59, 59), datetime(2026, 11, 4, 0, 0),
        )):
            saved = db.get(WeatherReport, report["report_id"])
            saved.location = location
            saved.created_at = created_at
        db.commit()
    response = client.get("/api/v1/weather/history", params={
        "location": "gaziantep", "start_date": "2026-11-01", "end_date": "2026-11-03",
    })
    assert response.status_code == 200
    assert [item["report_id"] for item in response.json()["reports"]] == [
        reports[2]["report_id"], reports[0]["report_id"],
    ]
    limited = client.get("/api/v1/weather/history", params={"location": "Ankara", "limit": 1})
    assert limited.json()["reports"][0]["report_id"] == reports[1]["report_id"]
    assert client.get("/api/v1/weather/history", params={"location": "%"}).json()["reports"] == []
    assert client.get("/api/v1/weather/history", params={
        "start_date": "2026-11-04", "end_date": "2026-11-01",
    }).status_code == 422
    assert client.get("/api/v1/weather/history", params={"start_date": "invalid"}).status_code == 422


def test_server_temperature_risk_range_is_saved_exactly(client, monkeypatch):
    monkeypatch.setattr(weather, "settings", replace(
        weather.settings, temp_risky_min_c=-20, temp_risky_max_c=60,
    ))
    original = create(client, monkeypatch)
    limits = DroneLimit.model_validate(original["applied_limits"])
    assert limits.temp_risky_min_c == -20
    assert limits.temp_risky_max_c == 60
    assert original["applied_limits"]["wind_ok_max_knots"] == weather.settings.wind_ok_max_knots
    assert DroneLimit(temp_ok_min_c=10, temp_ok_max_c=30).temp_risky_min_c == 5
    assert DroneLimit(temp_ok_min_c=10, temp_ok_max_c=30).temp_risky_max_c == 35


def test_legacy_report_preserves_final_decision_and_marks_missing_snapshots(client, monkeypatch, database):
    original = create(client, monkeypatch)
    with database() as db:
        report = db.get(WeatherReport, original["report_id"])
        report.ai_analysis_data = None
        report.decision = "UYGUN_DEGIL"
        report.decision_detail = "Eski kayıtlı karar"
        db.commit()
    response = client.get(f'/api/v1/weather/report/{original["report_id"]}')
    assert response.status_code == 200
    saved = response.json()
    assert saved["decision"] == "UYGUN_DEGIL"
    assert saved["decision_detail"] == "Eski kayıtlı karar"
    assert saved["parameter_snapshot_available"] is False
    assert saved["applied_limits"] is None
    assert saved["source_checks"] is None
    assert saved["sources"][0]["data_time"] is None
    with patch.object(weather, "generate_pdf", return_value=b"%PDF") as generate:
        client.get(f'/api/v1/weather/report/{original["report_id"]}/pdf')
    assert generate.call_args.kwargs["parameter_snapshot_available"] is False


def test_missing_report_returns_not_found(client):
    assert client.get("/api/v1/weather/report/99999999").status_code == 404


@pytest.mark.parametrize("metadata", ["not-json", "[]", '{"decision_snapshot": {}}'])
def test_corrupt_metadata_is_explicit_not_recomputed(client, monkeypatch, metadata, database):
    original = create(client, monkeypatch)
    with database() as db:
        report = db.get(WeatherReport, original["report_id"])
        report.ai_analysis_data = metadata
        db.commit()
    response = client.get(f'/api/v1/weather/report/{original["report_id"]}')
    assert response.status_code == 500
    assert "Kayıtlı" in response.json()["detail"]


def test_source_diagnostics_saved_as_observed_not_inferred_after_settings_change(client, monkeypatch):
    monkeypatch.setattr(weather, "settings", replace(
        weather.settings, windy_api_key="test-key", openweathermap_api_key="",
        weatherapi_api_key="", checkwx_api_key="",
    ))
    original = create(client, monkeypatch)
    checks = {(check["provider"], check["name"]): check for check in original["source_checks"]}
    assert checks[("Open-Meteo", "Anlık hava durumu")]["status"] == "received"
    assert checks[("Windy", "Model tahmini")]["status"] == "no_data"
    assert "kesin hata nedeni" in checks[("Windy", "Model tahmini")]["message"]
    assert checks[("OpenWeatherMap", "Anlık hava durumu")]["status"] == "not_configured"
    assert checks[("NOAA Aviation Weather", "METAR")]["status"] == "no_data"
    assert "test-key" not in json.dumps(original["source_checks"])
    monkeypatch.setattr(weather, "settings", replace(weather.settings, windy_api_key=""))
    saved = client.get(f'/api/v1/weather/report/{original["report_id"]}').json()
    assert saved["source_checks"] == original["source_checks"]
    assert saved["decision"] == original["decision"]

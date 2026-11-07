from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from decideflight.api import weather
from decideflight.main import app
from decideflight.services.flight_limits import DroneLimit


@pytest.fixture
def fixed_clock(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 7, 8, 30, tzinfo=timezone.utc)

    monkeypatch.setattr(weather, "datetime", Clock)


def hourly_payload(hours, *, day="2026-11-07", missing=None):
    count = len(hours)
    return {"hourly": {
        "time": [f"{day}T{hour:02}:00" for hour in hours],
        "temperature_2m": [None if hour == missing else 20.0 for hour in hours],
        "relative_humidity_2m": [50.0] * count,
        "dew_point_2m": [10.0] * count,
        "wind_speed_10m": [20.0] * count,
        "precipitation": [0.0] * count,
        "visibility": [10000.0] * count,
        "cloud_cover": [20.0] * count,
    }}


def plan(monkeypatch, payload, **changes):
    monkeypatch.setattr(weather, "_fetch_open_meteo_hourly", AsyncMock(return_value=payload))
    request = {
        "lat": 37, "lon": 37, "date_offset": 0, "start_hour": 9, "end_hour": 12,
        **changes,
    }
    return TestClient(app).post("/api/v1/weather/flight-plan", json=request)


def test_missing_hour_breaks_window_and_is_reported(monkeypatch, fixed_clock):
    response = plan(monkeypatch, hourly_payload([9, 10, 11, 12], missing=10))
    assert response.status_code == 200
    body = response.json()
    assert body["missing_hours"] == [10]
    assert body["best_window"] == {"start_hour": 11, "end_hour": 12, "length": 2}
    assert body["date"] == "2026-11-07"
    assert body["timezone"] == "UTC"
    assert body["hours"][0]["time"].endswith("Z")
    assert body["hours"][0]["cloud_cover_pct"] == 20
    assert body["hours"][0]["detail"]


def test_other_date_never_substitutes_requested_date(monkeypatch, fixed_clock):
    response = plan(monkeypatch, hourly_payload([9, 10], day="2026-11-08"))
    assert response.status_code == 503
    assert "veri bulunamadı" in response.json()["detail"]


def test_custom_limits_affect_plan_and_explanations(monkeypatch, fixed_clock):
    response = plan(
        monkeypatch, hourly_payload([9, 10]),
        limits=DroneLimit(wind_ok_max_knots=8, wind_risky_max_knots=12).model_dump(),
    )
    body = response.json()
    assert all(hour["decision"] == "RISKLI" for hour in body["hours"])
    assert "8–12" in body["hours"][0]["detail"]
    assert body["best_window"] is None


def test_past_hours_are_not_recommended(monkeypatch, fixed_clock):
    response = plan(monkeypatch, hourly_payload([6, 7]), start_hour=6, end_hour=7)
    body = response.json()
    assert all(hour["is_past"] for hour in body["hours"])
    assert body["best_window"] is None


@pytest.mark.parametrize(
    "slots, expected",
    [
        ([(12, "UYGUN"), (9, "UYGUN")], (9, 9, 1)),
        ([(9, "UYGUN"), (10, "RISKLI"), (11, "UYGUN")], (9, 9, 1)),
        ([(22, "UYGUN"), (23, "UYGUN")], (22, 23, 2)),
        ([(9, "RISKLI")], None),
    ],
)
def test_window_requires_contiguous_suitable_hours(slots, expected):
    hours = [weather.FlightPlanHourSchema(
        hour=hour, decision=decision, wind_kt=10, temperature_c=20,
        humidity_pct=50, visibility_km=10, precipitation_level=0,
    ) for hour, decision in slots]
    result = weather._best_window(hours)
    if expected is None:
        assert result is None
    else:
        assert (result.start_hour, result.end_hour, result.length) == expected


@pytest.mark.asyncio
async def test_forecast_is_next_hour_not_current_and_uses_saved_limits(monkeypatch, fixed_clock):
    fetch = AsyncMock(return_value=hourly_payload(range(8, 15), missing=10))
    monkeypatch.setattr(weather, "_fetch_open_meteo_hourly", fetch)
    monkeypatch.setattr(weather, "_load_air_quality", lambda report: None)
    monkeypatch.setattr(weather, "_load_ai_payload", lambda report: {
        "applied_limits": DroneLimit(wind_ok_max_knots=8, wind_risky_max_knots=12).model_dump(),
    })
    db = SimpleNamespace(get=lambda *args: SimpleNamespace(lat=37, lon=37))
    result = await weather.forecast_change(1, db=db)
    assert result.missing_offsets == [2]
    assert [hour.time.hour for hour in result.hours] == [9, 11, 12, 13, 14]
    assert all(hour.decision == "RISKLI" for hour in result.hours)
    assert "8–12" in result.hours[0].detail
    fetch.assert_awaited_once_with(37, 37)


def test_malformed_hourly_timestamp_is_explicit(monkeypatch, fixed_clock):
    payload = hourly_payload([9])
    payload["hourly"]["time"] = ["not-a-date"]
    response = plan(monkeypatch, payload)
    assert response.status_code == 503
    assert "zaman bilgisi geçersiz" in response.json()["detail"]


@pytest.mark.asyncio
async def test_forecast_unavailable_is_explicit(monkeypatch, fixed_clock):
    monkeypatch.setattr(weather, "_fetch_open_meteo_hourly", AsyncMock(return_value=hourly_payload([8])))
    monkeypatch.setattr(weather, "_load_air_quality", lambda report: None)
    monkeypatch.setattr(weather, "_load_ai_payload", lambda report: {})
    db = SimpleNamespace(get=lambda *args: SimpleNamespace(lat=37, lon=37))
    with pytest.raises(HTTPException) as error:
        await weather.forecast_change(1, db=db)
    assert error.value.status_code == 503

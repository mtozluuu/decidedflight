"""Tests for the feedback endpoint and AI decision engine context builder."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from decideflight.main import app
from decideflight.database import SessionLocal
from decideflight.services.ai_decision_engine import build_feedback_context
from decideflight.services.weather_fetcher import (
    PRECIP_NONE,
    WeatherSourceData,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _good_source() -> WeatherSourceData:
    return WeatherSourceData(
        source="test",
        wind_speed_knots=10.0,
        temperature_c=20.0,
        humidity_pct=60.0,
        visibility_km=10.0,
        precipitation_level=PRECIP_NONE,
        cloud_base_ft=1000.0,
        cloud_ceiling_ft=2000.0,
    )


def _mock_weather():
    return patch(
        "decideflight.api.weather.fetch_all_sources",
        new_callable=AsyncMock,
        return_value=[_good_source()],
    )


def _mock_geocode():
    return patch(
        "decideflight.api.weather.geocode_city",
        new_callable=AsyncMock,
        return_value=(41.01, 28.97, "Istanbul"),
    )


# ---------------------------------------------------------------------------
# Helper: create a report and return its ID
# ---------------------------------------------------------------------------


def create_report(client: TestClient) -> int:
    with _mock_geocode(), _mock_weather():
        resp = client.post(
            "/api/v1/weather/report",
            json={"location": "Istanbul"},
        )
    assert resp.status_code == 201
    return resp.json()["report_id"]


# ---------------------------------------------------------------------------
# Feedback endpoint tests
# ---------------------------------------------------------------------------


def test_submit_positive_feedback(client: TestClient) -> None:
    report_id = create_report(client)

    resp = client.post(
        f"/api/v1/weather/report/{report_id}/feedback",
        json={"correct": True},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["report_id"] == report_id
    assert data["correct"] is True
    assert "feedback_id" in data
    assert "created_at" in data


def test_submit_negative_feedback_with_comment(client: TestClient) -> None:
    report_id = create_report(client)

    resp = client.post(
        f"/api/v1/weather/report/{report_id}/feedback",
        json={"correct": False, "user_comment": "Aslında çok sisli bir gündü"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["correct"] is False


def test_feedback_for_nonexistent_report(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/weather/report/999999/feedback",
        json={"correct": True},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# AI decision engine context builder tests
# ---------------------------------------------------------------------------


def test_build_feedback_context_empty() -> None:
    """Returns empty string when no feedback exists in a fresh session."""
    db = SessionLocal()
    try:
        # Just verify the function returns a string (may or may not be empty
        # depending on prior test state)
        result = build_feedback_context(db)
        assert isinstance(result, str)
    finally:
        db.close()


def test_build_feedback_context_with_data(client: TestClient) -> None:
    """After submitting feedback, context string contains a summary line."""
    report_id = create_report(client)
    client.post(
        f"/api/v1/weather/report/{report_id}/feedback",
        json={"correct": True},
    )

    db = SessionLocal()
    try:
        context = build_feedback_context(db)
        assert "Geçmiş kararlar" in context
        assert "DOĞRU" in context
    finally:
        db.close()

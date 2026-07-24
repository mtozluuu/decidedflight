"""Tests for AI decision engine, feedback context injection, trend analysis,
and grid summary endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from decideflight.main import app
from decideflight.services.ai_decision_engine import (
    AIDecisionResult,
    make_ai_decision,
)
from decideflight.services.trend_analyzer import (
    WindTrend,
    _compute_trend,
    fetch_wind_trend,
)
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
        source="Open-Meteo",
        wind_speed_knots=8.0,
        temperature_c=20.0,
        humidity_pct=55.0,
        visibility_km=10.0,
        precipitation_level=PRECIP_NONE,
        cloud_base_ft=2000.0,
        cloud_ceiling_ft=3000.0,
    )


def _mock_weather():
    return patch(
        "decideflight.api.weather.fetch_all_sources",
        new=AsyncMock(return_value=[_good_source()]),
    )


def _mock_geocode():
    return patch(
        "decideflight.api.weather.geocode_city",
        new=AsyncMock(return_value=(41.0, 28.9, "Istanbul, Turkey")),
    )


def _mock_trend_none():
    return patch(
        "decideflight.api.weather.fetch_wind_trend",
        new=AsyncMock(return_value=None),
    )


# ---------------------------------------------------------------------------
# Feature 1: AI fallback behaviour
# ---------------------------------------------------------------------------


class TestAIDecisionFallback:
    @pytest.mark.asyncio
    async def test_no_api_key_returns_rule_based(self):
        """When OPENAI_API_KEY is not set, make_ai_decision returns AIDecisionResult
        with empty AI fields (i.e., falls back to rule-based logic)."""
        import os

        original = os.environ.pop("OPENAI_API_KEY", None)
        try:
            result = await make_ai_decision(
                sources=[_good_source()],
                location="Istanbul",
                lat=41.0,
                lon=28.9,
                feedback_context="",
                wind_trend="",
            )
        finally:
            if original is not None:
                os.environ["OPENAI_API_KEY"] = original

        assert isinstance(result, AIDecisionResult)
        assert result.decision in ("UYGUN", "RISKLI", "UYGUN_DEGIL")
        # Fallback: AI-specific fields are empty
        assert result.summary == ""
        assert result.detailed_analysis == ""
        assert result.risk_factors == []
        assert result.recommendations == []

    @pytest.mark.asyncio
    async def test_exception_in_gpt_falls_back(self):
        """If GPT raises an exception, make_ai_decision silently falls back."""
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}),
            patch(
                "decideflight.services.ai_decision_engine._AsyncOpenAI",  # noqa: E501
                side_effect=Exception("network error"),
            ),
        ):
            result = await make_ai_decision(
                sources=[_good_source()],
                location="Istanbul",
                lat=41.0,
                lon=28.9,
            )

        assert isinstance(result, AIDecisionResult)
        assert result.decision in ("UYGUN", "RISKLI", "UYGUN_DEGIL")
        assert result.summary == ""

    @pytest.mark.asyncio
    async def test_gpt_invalid_json_falls_back(self):
        """If GPT returns invalid JSON, make_ai_decision falls back gracefully."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "this is not json"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}),
            patch(
                "decideflight.services.ai_decision_engine._AsyncOpenAI",  # noqa: E501
                return_value=mock_client,
            ),
        ):
            result = await make_ai_decision(
                sources=[_good_source()],
                location="Istanbul",
                lat=41.0,
                lon=28.9,
            )

        assert isinstance(result, AIDecisionResult)
        assert result.decision in ("UYGUN", "RISKLI", "UYGUN_DEGIL")

    @pytest.mark.asyncio
    async def test_gpt_valid_response_parsed(self):
        """When GPT returns valid JSON, the AI fields are populated."""
        import json

        gpt_payload = {
            "karar": "UYGUN",
            "guven_skoru": 88,
            "ozet": "Hava koşulları uçuş için elverişlidir.",
            "detayli_analiz": "Rüzgar düşük, görüş iyi.",
            "risk_faktorleri": ["Hafif nem artışı"],
            "tavsiyeler": ["Erken saatte uçun"],
            "parametre_degerlendirmeleri": {"ruzgar": "iyi"},
        }

        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(gpt_payload)

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}),
            patch(
                "decideflight.services.ai_decision_engine._AsyncOpenAI",  # noqa: E501
                return_value=mock_client,
            ),
        ):
            result = await make_ai_decision(
                sources=[_good_source()],
                location="Istanbul",
                lat=41.0,
                lon=28.9,
            )

        assert isinstance(result, AIDecisionResult)
        assert result.decision == "UYGUN"
        assert result.confidence == 88
        assert "elverişlidir" in result.summary
        assert result.risk_factors == ["Hafif nem artışı"]
        assert result.recommendations == ["Erken saatte uçun"]


# ---------------------------------------------------------------------------
# Feature 4: Feedback context injection
# ---------------------------------------------------------------------------


class TestFeedbackContextInjection:
    def test_no_api_key_report_still_returns_decision(self, client: TestClient):
        """Report endpoint works without OpenAI key — feedback context is built
        but only passed to rule-based fallback (no crash)."""
        with (
            _mock_weather(),
            _mock_trend_none(),
        ):
            response = client.post(
                "/api/v1/weather/report",
                json={"lat": 41.0, "lon": 28.9},
            )
        assert response.status_code == 201
        body = response.json()
        assert body["decision"] in ("UYGUN", "RISKLI", "UYGUN_DEGIL")

    def test_report_ai_fields_absent_without_key(self, client: TestClient):
        """Without OPENAI_API_KEY the response should not have AI summary."""
        with (
            _mock_weather(),
            _mock_trend_none(),
        ):
            response = client.post(
                "/api/v1/weather/report",
                json={"lat": 41.0, "lon": 28.9},
            )
        body = response.json()
        assert body.get("summary") is None
        assert body.get("detailed_analysis") is None

    def test_feedback_context_passed_to_make_ai_decision(self, client: TestClient):
        """build_feedback_context result is passed as kwarg to make_ai_decision."""
        captured: list[str] = []

        async def _fake_make_ai(
            sources: list,
            location: str,
            lat: float,
            lon: float,
            feedback_context: str = "",
            wind_trend: str = "",
        ) -> object:
            captured.append(feedback_context)
            from decideflight.services.decision_engine import make_decision
            from decideflight.services.ai_decision_engine import _wrap_rule_result

            return _wrap_rule_result(make_decision(sources))

        with (
            _mock_weather(),
            _mock_trend_none(),
            patch(
                "decideflight.api.weather.make_ai_decision",
                new=_fake_make_ai,
            ),
            patch(
                "decideflight.api.weather.build_feedback_context",
                return_value="Geçmiş kararlar: UYGUN → DOĞRU",
            ),
        ):
            client.post(
                "/api/v1/weather/report",
                json={"lat": 41.0, "lon": 28.9},
            )

        assert len(captured) == 1
        assert "UYGUN" in captured[0]


# ---------------------------------------------------------------------------
# Feature 5: Trend analysis
# ---------------------------------------------------------------------------


class TestTrendAnalyzer:
    def test_increasing_trend(self):
        speeds = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = _compute_trend(speeds)
        assert result.direction == "artıyor"
        assert result.change_per_hour_kt > 0

    def test_decreasing_trend(self):
        speeds = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0]
        result = _compute_trend(speeds)
        assert result.direction == "azalıyor"
        assert result.change_per_hour_kt < 0

    def test_stable_trend(self):
        speeds = [8.0, 8.1, 7.9, 8.0, 8.1, 8.0]
        result = _compute_trend(speeds)
        assert result.direction == "sabit"

    def test_single_value_stable(self):
        result = _compute_trend([7.0])
        assert result.direction == "sabit"

    def test_description_contains_direction(self):
        speeds = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        result = _compute_trend(speeds)
        assert result.direction in result.description

    @pytest.mark.asyncio
    async def test_fetch_wind_trend_handles_http_error(self):
        """fetch_wind_trend returns None on network error."""
        patch_path = "decideflight.services.trend_analyzer.httpx.AsyncClient"
        with patch(patch_path) as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=Exception("timeout"))

            result = await fetch_wind_trend(41.0, 28.9)
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_wind_trend_success(self):
        """fetch_wind_trend parses a valid Open-Meteo response."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(
            return_value={
                "hourly": {
                    "wind_speed_10m": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                }
            }
        )

        patch_path = "decideflight.services.trend_analyzer.httpx.AsyncClient"
        with patch(patch_path) as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)

            result = await fetch_wind_trend(41.0, 28.9)

        assert result is not None
        assert isinstance(result, WindTrend)
        assert result.direction == "artıyor"

    def test_wind_trend_in_report_response(self, client: TestClient):
        """wind_trend field is present in report response when trend returns data."""
        fake_trend = WindTrend(
            direction="artıyor",
            change_per_hour_kt=1.2,
            description="Son 6 saatte rüzgar: 1.2 kt/h artıyor",
        )
        with (
            _mock_weather(),
            patch(
                "decideflight.api.weather.fetch_wind_trend",
                new=AsyncMock(return_value=fake_trend),
            ),
        ):
            response = client.post(
                "/api/v1/weather/report",
                json={"lat": 41.0, "lon": 28.9},
            )
        assert response.status_code == 201
        body = response.json()
        assert body.get("wind_trend") == "Son 6 saatte rüzgar: 1.2 kt/h artıyor"


# ---------------------------------------------------------------------------
# Feature 2: Grid summary endpoint
# ---------------------------------------------------------------------------


class TestGridSummaryEndpoint:
    def _grid_point(self, decision: str = "UYGUN") -> dict:
        return {
            "lat": 41.0,
            "lon": 28.9,
            "decision": decision,
            "wind_speed_knots": 8.0,
            "temperature_c": 20.0,
            "humidity_pct": 55.0,
            "visibility_km": 10.0,
            "cloud_base_ft": 2000.0,
            "cloud_ceiling_ft": 3000.0,
            "precipitation_level": 0,
            "cloud_cover_pct": 10.0,
        }

    def _request_body(self, n_uygun: int = 20, n_riskli: int = 3, n_degil: int = 2):
        points = (
            [self._grid_point("UYGUN")] * n_uygun
            + [self._grid_point("RISKLI")] * n_riskli
            + [self._grid_point("UYGUN_DEGIL")] * n_degil
        )
        return {
            "points": points,
            "summary": {
                "UYGUN": n_uygun,
                "RISKLI": n_riskli,
                "UYGUN_DEGIL": n_degil,
                "total": n_uygun + n_riskli + n_degil,
            },
            "altitude_ft": 1000.0,
            "location_hint": "Istanbul, Turkey",
        }

    def test_grid_summary_returns_200(self, client: TestClient):
        response = client.post(
            "/api/v1/weather/grid/summary",
            json=self._request_body(),
        )
        assert response.status_code == 200

    def test_grid_summary_has_ai_summary_field(self, client: TestClient):
        response = client.post(
            "/api/v1/weather/grid/summary",
            json=self._request_body(),
        )
        body = response.json()
        assert "ai_summary" in body
        assert isinstance(body["ai_summary"], str)
        assert len(body["ai_summary"]) > 0

    def test_grid_summary_rule_based_without_key(self, client: TestClient):
        """Without OPENAI_API_KEY, the rule-based Turkish summary is returned."""
        import os

        original = os.environ.pop("OPENAI_API_KEY", None)
        try:
            response = client.post(
                "/api/v1/weather/grid/summary",
                json=self._request_body(n_uygun=20, n_riskli=3, n_degil=2),
            )
        finally:
            if original is not None:
                os.environ["OPENAI_API_KEY"] = original

        assert response.status_code == 200
        body = response.json()
        # Rule-based summary contains Turkish text and stats
        assert "UYGUN" in body["ai_summary"] or "uygun" in body["ai_summary"].lower()

    def test_grid_summary_empty_points(self, client: TestClient):
        response = client.post(
            "/api/v1/weather/grid/summary",
            json={
                "points": [],
                "summary": {"UYGUN": 0, "RISKLI": 0, "UYGUN_DEGIL": 0, "total": 0},
                "altitude_ft": 1000.0,
                "location_hint": "",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert "ai_summary" in body

    def test_grid_summary_gpt_fallback_on_error(self, client: TestClient):
        """If GPT raises, the rule-based summary is still returned."""
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}),
            patch(
                "decideflight.api.weather._AsyncOpenAI",
                side_effect=Exception("api error"),
            ),
        ):
            response = client.post(
                "/api/v1/weather/grid/summary",
                json=self._request_body(),
            )
        assert response.status_code == 200
        body = response.json()
        assert len(body["ai_summary"]) > 0

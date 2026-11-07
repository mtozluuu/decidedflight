import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException

from decideflight.api import weather
from decideflight.services import weather_fetcher
from decideflight.services.ai_decision_engine import (
    _parse_gpt_response,
    make_ai_decision,
)
from decideflight.services.decision_engine import (
    make_decision,
    usable_sources,
    wind_consensus,
)
from decideflight.services.flight_limits import DroneLimit
from decideflight.services.report_generator import generate_pdf
from decideflight.services.weather_cache import WeatherCache
from decideflight.services.weather_fetcher import (
    WeatherSourceData,
    _metar_hazards,
)


def source(**changes):
    defaults = dict(
        source="Test",
        wind_speed_knots=10,
        temperature_c=20,
        humidity_pct=50,
        visibility_km=10,
        precipitation_level=0,
        cloud_base_ft=3000,
        cloud_ceiling_ft=4000,
        wind_direction_deg=180,
    )
    return WeatherSourceData(**(defaults | changes))


@pytest.mark.parametrize(
    "field",
    [
        "wind_speed_knots",
        "visibility_km",
        "temperature_c",
        "humidity_pct",
        "precipitation_level",
    ],
)
def test_missing_critical_field_is_not_suitable(field):
    result = make_decision([source(**{field: None})])
    assert result.decision == "VERI_YETERSIZ"
    assert len(result.missing_parameters) == 1
    assert result.known_decision == "UYGUN"
    assert "uçuş onayı değildir" in result.detail
    assert generate_pdf(
        "Test", 39, 35, [source(**{field: None})], result
    ).startswith(b"%PDF")


def test_missing_data_keeps_known_risk_visible():
    result = make_decision([source(wind_speed_knots=20, visibility_km=None)])
    assert result.decision == "VERI_YETERSIZ"
    assert result.known_decision == "RISKLI"
    assert "Rüzgar hızı" in result.detail
    dangerous = make_decision(
        [source(wind_speed_knots=30, visibility_km=None)]
    )
    assert dangerous.decision == "UYGUN_DEGIL"
    assert dangerous.missing_parameters == ["Görüş mesafesi"]


def test_zero_visibility_is_not_missing_or_good_weather():
    payload = {
        "hourly": {
            "time": ["2026-11-08T10:00"],
            "temperature_2m": [20],
            "wind_speed_10m": [0],
            "visibility": [0],
            "relative_humidity_2m": [0],
            "precipitation": [0],
            "dew_point_2m": [10],
            "cloud_cover": [0],
        }
    }
    row = weather._hourly_row_from_payload(payload, 0)
    assert row["wind_speed_knots"] == 0
    assert row["visibility_km"] == 0
    assert row["humidity_pct"] == 0
    assert row["precipitation_level"] == 0
    assert (
        make_decision([weather._build_point_source(row)]).decision
        == "UYGUN_DEGIL"
    )
    del payload["hourly"]["visibility"]
    row = weather._hourly_row_from_payload(payload, 0)
    assert row["visibility_km"] is None
    assert (
        make_decision([weather._build_point_source(row)]).decision
        == "VERI_YETERSIZ"
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_wind_is_missing(value):
    result = make_decision([source(wind_speed_knots=value)])
    assert result.avg_wind_knots is None
    assert result.decision == "VERI_YETERSIZ"


def test_average_cannot_hide_bad_wind_or_visibility():
    result = make_decision(
        [
            source(source="A", wind_speed_knots=30, visibility_km=0.5),
            source(source="B", wind_speed_knots=0, visibility_km=20),
        ]
    )
    assert result.avg_wind_knots == 15
    assert result.avg_visibility_km == 10.25
    assert result.decision == "UYGUN_DEGIL"
    assert "Ortalama riski azaltmadı" in result.detail


def test_identical_metar_is_not_two_independent_observations():
    first = source(
        source="NOAA",
        observation_id="LTXX 081000Z 18010KT 9999 NSC 20/10 Q1013",
    )
    second = replace(
        first, source="CheckWX", observation_id="METAR " + first.observation_id
    )
    selected, excluded = usable_sources([first, second])
    assert selected == [first]
    assert "tekrarı" in excluded[0]


def test_stale_metar_is_excluded_not_refreshed_by_retrieval():
    old_time = datetime.now(timezone.utc) - timedelta(hours=2)
    old = source(
        source="METAR (NOAA/aviationweather.gov)",
        raw={"obsTime": old_time.timestamp()},
    )
    result = make_decision([old])
    assert result.decision == "VERI_YETERSIZ"
    assert len(result.excluded_sources) == 1
    assert result.avg_wind_knots is None


def test_far_airport_is_not_local_observation():
    selected, excluded = usable_sources([source(station_distance_km=101)])
    assert selected == []
    assert "100 km" in excluded[0]


def test_opposing_winds_have_no_consensus_direction():
    assert (
        wind_consensus(
            [source(wind_direction_deg=0), source(wind_direction_deg=180)]
        )
        is None
    )
    result = wind_consensus(
        [source(wind_direction_deg=350), source(wind_direction_deg=10)]
    )
    assert result[0] == 10
    assert min(result[1], 360 - result[1]) == pytest.approx(0)


@pytest.mark.parametrize(
    "body, precip, base, ceiling, unlimited",
    [
        ("18010KT 9999 -RA BKN003 20/10", 1, 300, 300, False),
        ("18010KT 9999 +TSRA OVC010 20/10", 2, 1000, 1000, False),
        ("18010KT CAVOK 20/10", 0, None, None, True),
        ("18010KT 9999 FEW030 SCT050 20/10", 0, 3000, None, True),
        ("18010KT 0100 VV/// 20/10", 0, None, None, False),
        ("18010KT 9999 NSC 20/10 TEMPO TSRA BKN001", 0, None, None, True),
    ],
)
def test_metar_weather_and_ceiling(body, precip, base, ceiling, unlimited):
    parsed = _metar_hazards("LTXX 081000Z " + body + " Q1013")
    assert parsed == dict(
        precipitation_level=precip,
        cloud_base_ft=base,
        cloud_ceiling_ft=ceiling,
        ceiling_unlimited=unlimited,
        clouds_clear="CAVOK" in body or "NSC" in body,
    )


@pytest.mark.asyncio
async def test_noaa_uses_gust_and_raw_metar_hazards():
    client = SimpleNamespace(
        get=AsyncMock(
            return_value=SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: [
                    {
                        "wspd": 10,
                        "wgst": 30,
                        "rawOb": (
                            "LTXX 081000Z 18010G30KT "
                            "9999 -RA BKN003 20/10 Q1013"
                        ),
                    }
                ],
            )
        )
    )
    parsed = await weather_fetcher._fetch_avwx_gov_metar("LTXX", client)
    assert parsed.wind_speed_knots == 30
    assert parsed.precipitation_level == 1
    assert parsed.cloud_ceiling_ft == 300
    assert parsed.visibility_km is None


@pytest.mark.asyncio
async def test_open_meteo_partial_response_preserves_missing_and_zero():
    client = SimpleNamespace(
        get=AsyncMock(
            return_value=SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "current": {
                        "wind_speed_10m": 0,
                        "temperature_2m": 0,
                        "visibility": 0,
                    }
                },
            )
        )
    )
    parsed = await weather_fetcher._fetch_open_meteo(39, 35, client)
    assert (
        parsed.wind_speed_knots
        == parsed.temperature_c
        == parsed.visibility_km
        == 0
    )
    assert parsed.humidity_pct is None
    assert parsed.precipitation_level is None


@pytest.mark.asyncio
async def test_ai_does_not_fill_missing_or_weaken_floor(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = AsyncMock()
    monkeypatch.setattr(
        "decideflight.services.ai_decision_engine._AsyncOpenAI", client
    )
    result = await make_ai_decision(
        [source(visibility_km=None)], "Test", 39, 35
    )
    assert result.decision == "VERI_YETERSIZ"
    assert result.missing_parameters == ["Görüş mesafesi"]
    client.assert_not_called()
    parsed = _parse_gpt_response(
        {"karar": "UYGUN", "guven_skoru": 100}, result
    )
    assert parsed.decision == "VERI_YETERSIZ"
    assert parsed.missing_parameters == result.missing_parameters


def test_grid_and_point_use_same_temperature_overrides_and_missing_policy():
    limits = DroneLimit(temp_risky_min_override_c=-20)
    values = source(temperature_c=-10)
    result = weather._eval_grid_decision(
        wind_speed_knots=values.wind_speed_knots,
        visibility_km=values.visibility_km,
        temperature_c=values.temperature_c,
        humidity_pct=values.humidity_pct,
        cloud_base_ft=values.cloud_base_ft,
        cloud_ceiling_ft=values.cloud_ceiling_ft,
        precipitation_level=values.precipitation_level,
        limits=limits,
    )
    assert (
        result == make_decision([values], limits=limits).decision == "RISKLI"
    )


@pytest.mark.asyncio
async def test_grid_retains_failed_points_as_unknown(monkeypatch):
    calls = 0

    async def fetch(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            return None
        return dict(
            wind_speed_knots=10,
            temperature_c=20,
            humidity_pct=50,
            visibility_km=10,
            precipitation_level=0,
            cloud_base_ft=3000,
            cloud_ceiling_ft=4000,
            cloud_cover_pct=20,
        )

    monkeypatch.setattr(weather, "_fetch_grid_point_weather", fetch)
    monkeypatch.setattr(weather, "_GRID_BATCH_DELAY", 0)
    monkeypatch.setattr(
        weather, "_build_grid_ai_summary", AsyncMock(return_value=None)
    )
    result = await weather.analyse_weather_grid(
        weather.GridAnalysisRequest(
            lat_min=39,
            lat_max=40,
            lon_min=35,
            lon_max=36,
        )
    )
    assert len(result.points) == result.summary.total == 25
    assert result.summary.VERI_YETERSIZ == 1
    assert result.summary.UYGUN == 24
    assert result.points[0].visibility_km is None


@pytest.mark.asyncio
async def test_cache_shares_requests_and_returns_independent_values():
    cache = WeatherCache(ttl=60, max_entries=2)
    loader = AsyncMock(return_value={"values": [1]})
    results = await asyncio.gather(*(cache.get("a", loader) for _ in range(4)))
    loader.assert_awaited_once()
    results[0]["values"].append(2)
    assert await cache.get("a", loader) == {"values": [1]}
    await cache.get("b", loader)
    await cache.get("c", loader)
    assert len(cache.entries) == 2
    assert not cache.pending


@pytest.mark.asyncio
async def test_expired_cache_does_not_hide_provider_error():
    cache = WeatherCache(ttl=60)
    await cache.get("a", AsyncMock(return_value={"old": True}))
    timestamp, data = cache.entries["a"]
    cache.entries["a"] = (timestamp - 61, data)
    with pytest.raises(RuntimeError):
        await cache.get(
            "a", AsyncMock(side_effect=RuntimeError("provider failed"))
        )
    loader = AsyncMock(return_value={"new": True})
    assert await cache.get("a", loader) == {"new": True}
    assert not cache.pending


@pytest.mark.asyncio
async def test_seasonal_provider_error_never_produces_fake_months(monkeypatch):
    monkeypatch.setattr(
        weather,
        "_fetch_with_retry",
        AsyncMock(side_effect=httpx.ConnectError("offline")),
    )
    with pytest.raises(HTTPException) as error:
        await weather.seasonal_analysis(
            weather.SeasonalRequest(lat=39, lon=35)
        )
    assert error.value.status_code == 503
    assert "varsayımsal" in error.value.detail


@pytest.mark.asyncio
async def test_grid_selects_timestamp_not_array_offset(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 8, 10, 30, tzinfo=timezone.utc)

    monkeypatch.setattr(weather, "datetime", Clock)
    payload = {
        "hourly": {
            "time": ["2026-11-08T09:00", "2026-11-08T12:00"],
            "temperature_2m": [5, 20],
            "wind_speed_10m": [100, 10],
            "visibility": [0, 10000],
            "precipitation": [5, 0],
            "relative_humidity_2m": [90, 50],
            "dew_point_2m": [5, 10],
            "cloud_cover": [100, 20],
        }
    }
    response = SimpleNamespace(json=lambda: payload)
    monkeypatch.setattr(
        weather, "_fetch_with_retry", AsyncMock(return_value=response)
    )
    result = await weather._fetch_grid_point_weather(39, 35, 10, 2, None)
    assert result["temperature_c"] == 20
    assert result["wind_speed_knots"] == 5.4
    assert result["visibility_km"] == 10
    assert result["time"].hour == 12
    pressure = await weather._fetch_grid_point_weather(39, 35, 3000, 2, None)
    assert pressure["wind_speed_knots"] is None
    payload["hourly"]["temperature_2m"][1] = None
    payload["hourly"]["wind_speed_10m"][1] = 100
    partial = await weather._fetch_grid_point_weather(39, 35, 10, 2, None)
    assert partial["temperature_c"] is None
    assert (
        make_decision([weather._build_point_source(partial)]).decision
        == "UYGUN_DEGIL"
    )
    payload["hourly"]["visibility"][1] = float("nan")
    payload["hourly"]["wind_speed_10m"][1] = float("inf")
    invalid = await weather._fetch_grid_point_weather(39, 35, 10, 2, None)
    assert invalid["visibility_km"] is None
    assert invalid["wind_speed_knots"] is None
    payload["hourly"]["time"] = ["2026-11-08T09:00", "2026-11-08T11:00"]
    assert await weather._fetch_grid_point_weather(39, 35, 10, 2, None) is None


@pytest.mark.asyncio
async def test_expired_map_cache_never_hides_provider_error(monkeypatch):
    from decideflight.services import map_layers

    monkeypatch.setattr(
        map_layers,
        "_cache",
        {
            "surface": (0, {"points": [{"wind_speed_knots": 10}]}),
        },
    )
    monkeypatch.setattr(
        map_layers,
        "_live_cache",
        {
            "sigmet": (0, []),
        },
    )
    client = SimpleNamespace(
        get=AsyncMock(side_effect=httpx.ConnectError("offline")),
    )

    class Client:
        async def __aenter__(self):
            return client

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(
        map_layers.httpx, "AsyncClient", lambda **kwargs: Client()
    )
    with pytest.raises(map_layers.MapLayerUnavailableError):
        await map_layers.load_layer("surface")
    with pytest.raises(map_layers.MapLayerUnavailableError):
        await map_layers._cached_get("sigmet", map_layers.AWC_SIGMET_URL, {})

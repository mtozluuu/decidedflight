from datetime import datetime, timezone

import pytest

from decideflight.services.data_freshness import observation_timing, source_timing
from decideflight.services.weather_fetcher import WeatherSourceData


def utc(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "now, status",
    [("2026-11-07T17:50", "current"), ("2026-11-07T17:50:01", "stale")],
)
def test_metar_sixty_minute_boundary(now, status):
    result = observation_timing("METAR LTAJ 071650Z CAVOK", utc(now))
    assert result["status"] == status
    assert result["issued_at"] == "2026-11-07T16:50:00+00:00"


@pytest.mark.parametrize(
    "now, status",
    [("2026-11-07T17:00", "upcoming"), ("2026-11-07T18:00", "valid"),
     ("2026-11-08T18:00", "expired")],
)
def test_taf_actual_validity(now, status):
    result = observation_timing(
        "TAF AMD LTAJ 071640Z 0718/0818 CAVOK", utc(now), taf=True
    )
    assert result["status"] == status
    assert result["valid_until"] == "2026-11-08T18:00:00+00:00"


def test_year_rollover_and_hour_24():
    result = observation_timing(
        "TAF LTAJ 311700Z 3118/0124 CAVOK", utc("2027-01-01T01:00"), taf=True
    )
    assert result["issued_at"] == "2026-12-31T17:00:00+00:00"
    assert result["valid_until"] == "2027-01-02T00:00:00+00:00"
    assert result["status"] == "valid"


@pytest.mark.parametrize(
    "text", [None, "METAR LTAJ invalid", "METAR LTAJ 079999Z CAVOK",
             "METAR LTAJ 072200Z CAVOK"]
)
def test_invalid_or_future_metar_not_marked_current(text):
    assert observation_timing(text, utc("2026-11-07T18:00"))["status"] == "unknown"


def test_cancelled_taf_not_marked_valid():
    assert observation_timing(
        "TAF LTAJ 071640Z 0718/0818 CNL", utc("2026-11-07T19:00"), taf=True
    )["status"] == "unknown"


def test_source_forecast_time_is_not_model_freshness():
    source = WeatherSourceData(
        "Open-Meteo", 5, 20, 60, 10, 0, None, None,
        raw={"time": "2026-11-07T18:00"},
    )
    result = source_timing(source, utc("2026-11-07T18:10"))
    assert result["data_time"] == "2026-11-07T18:00:00+00:00"
    assert result["freshness"] == "unknown"
    assert result["time_kind"] == "forecast"


def test_metar_source_uses_observation_not_fetch_time():
    source = WeatherSourceData(
        "METAR (NOAA/aviationweather.gov)", 5, 20, 60, 10, 0, None, None,
        raw={"obsTime": utc("2026-11-07T16:00").timestamp()},
    )
    assert source_timing(source, utc("2026-11-07T18:00"))["freshness"] == "stale"

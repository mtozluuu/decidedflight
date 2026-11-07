import pytest
from pydantic import ValidationError

from decideflight.api.weather import DroneLimit, _eval_grid_decision
from decideflight.services.decision_engine import make_decision
from decideflight.services.weather_fetcher import WeatherSourceData


@pytest.mark.parametrize(
    "threshold, cover, expected",
    [
        (50, 50, "UYGUN"),
        (50, 50.1, "RISKLI"),
        (50, 100, "RISKLI"),
        (80, 75, "UYGUN"),
        (20, 25, "RISKLI"),
        (0, 0, "UYGUN"),
        (100, 100, "UYGUN"),
        (50, None, "UYGUN"),
    ],
)
def test_grid_cloud_cover_uses_custom_limit(threshold, cover, expected):
    assert _eval_grid_decision(
        wind_speed_knots=5,
        visibility_km=10,
        temperature_c=20,
        humidity_pct=60,
        cloud_base_ft=2000,
        cloud_ceiling_ft=3000,
        precipitation_level=0,
        limits=DroneLimit(cloud_cover_ok_max_pct=threshold),
        cloud_cover_pct=cover,
    ) == expected


@pytest.mark.parametrize("threshold", [-1, 101, float("nan")])
def test_grid_cloud_cover_rejects_invalid_limit(threshold):
    with pytest.raises(ValidationError):
        DroneLimit(cloud_cover_ok_max_pct=threshold)


def test_older_grid_settings_default_to_fifty_percent():
    assert DroneLimit().cloud_cover_ok_max_pct == 50


@pytest.mark.parametrize(
    "changes", [
        {"wind_ok_max_knots": 4}, {"visibility_ok_min_km": 12},
        {"temp_ok_max_c": 19}, {"humidity_ok_max_pct": 50},
        {"cloud_base_ok_min_ft": 2500}, {"cloud_ceiling_ok_min_ft": 3500},
        {"cloud_cover_ok_max_pct": 40},
    ],
)
def test_point_and_region_use_identical_custom_thresholds(changes):
    limits = DroneLimit(**changes)
    source = WeatherSourceData("Test", 5, 20, 60, 10, 0, 2000, 3000, cloud_cover_pct=50)
    point = make_decision([source], limits=limits)
    regional = _eval_grid_decision(
        wind_speed_knots=5, visibility_km=10, temperature_c=20, humidity_pct=60,
        cloud_base_ft=2000, cloud_ceiling_ft=3000, precipitation_level=0,
        limits=limits, cloud_cover_pct=50,
    )
    assert point.decision == regional == "RISKLI"

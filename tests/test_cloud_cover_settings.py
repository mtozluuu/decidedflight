import pytest
from pydantic import ValidationError

from decideflight.api.weather import DroneLimit, _eval_grid_decision


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

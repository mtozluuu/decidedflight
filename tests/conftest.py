"""Shared pytest fixtures and setup for the test suite."""

import pytest

from decideflight.database import init_db

# Import app so all models are registered with Base.metadata before init_db runs
import decideflight.models  # noqa: F401


@pytest.fixture(autouse=True, scope="session")
def initialize_test_database():
    """Create all DB tables once for the test session."""
    init_db()


@pytest.fixture(autouse=True)
def clear_weather_caches():
    from decideflight.api.weather import _hourly_cache, _seasonal_cache
    from decideflight.services.weather_fetcher import _air_quality_cache, _weather_cache

    for cache in (_hourly_cache, _seasonal_cache, _air_quality_cache, _weather_cache):
        cache.clear()

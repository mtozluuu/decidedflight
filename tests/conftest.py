"""Shared pytest fixtures and setup for the test suite."""

import pytest

from decideflight.database import init_db

# Import app so all models are registered with Base.metadata before init_db runs
import decideflight.models  # noqa: F401


@pytest.fixture(autouse=True, scope="session")
def initialize_test_database():
    """Create all DB tables once for the test session."""
    init_db()

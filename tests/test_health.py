import pytest
from fastapi.testclient import TestClient

from decideflight.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_root_endpoint(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    # Root now serves the web UI (index.html) when the static directory exists
    assert "text/html" in response.headers["content-type"]
    assert "DRONE UÇUŞ HAVA ANALİZİ" in response.text
    assert "id=\"countrySelect\"" in response.text
    assert "/static/data/cities.json" in response.text


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

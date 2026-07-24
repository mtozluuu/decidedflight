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


@pytest.mark.parametrize(
    "icon_path",
    [
        "/favicon.ico",
        "/apple-touch-icon.png",
        "/apple-touch-icon-precomposed.png",
    ],
)
def test_icon_endpoints(client: TestClient, icon_path: str) -> None:
    response = client.get(icon_path)

    assert response.status_code == 200
    assert "image/svg+xml" in response.headers["content-type"]
    assert "<svg" in response.text

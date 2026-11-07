from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from decideflight.main import app
from decideflight.services import airspace

STATIC = Path(airspace.__file__).resolve().parents[1] / "static"

RAW = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "_id": "a",
                "name": "GAZIANTEP TMA",
                "type": 7,
                "upperLimit": {"value": 240, "unit": 6, "referenceDatum": 2},
                "lowerLimit": {"value": 3500, "unit": 1, "referenceDatum": 1},
            },
            "geometry": {"type": "Polygon", "coordinates": [[[37.123456789, 36.9], [37.5, 36.9], [37.5, 37.2], [37.123456789, 36.9]]]},
        },
        {
            "type": "Feature",
            "properties": {
                "_id": "b",
                "name": "SPORT",
                "type": 28,
                "upperLimit": {"value": 1, "unit": 1, "referenceDatum": 1},
                "lowerLimit": {"value": 0, "unit": 1, "referenceDatum": 0},
            },
            "geometry": {"type": "Polygon", "coordinates": [[[1, 1], [2, 2], [3, 3], [1, 1]]]},
        },
        {
            "type": "Feature",
            "properties": {
                "_id": "c",
                "name": "ISTANBUL CTR",
                "type": 4,
                "upperLimit": {"value": 2500, "unit": 1, "referenceDatum": 1},
                "lowerLimit": {"value": 0, "unit": 1, "referenceDatum": 0},
            },
            "geometry": {"type": "Polygon", "coordinates": [[[28, 41], [29, 41], [29, 42], [28, 41]]]},
        },
    ],
}


@pytest.fixture(autouse=True)
def reset_cache():
    airspace._cache.update({"fetched_at": 0.0, "features": None})
    yield
    airspace._cache.update({"fetched_at": 0.0, "features": None})


@pytest.fixture
def client(monkeypatch) -> TestClient:
    async def fake_load(force: bool = False):
        return airspace.normalize_collection(RAW), 1_700_000_000.0

    monkeypatch.setattr("decideflight.api.airspace.load_airspaces", fake_load)
    return TestClient(app)


def test_format_limit() -> None:
    assert airspace.format_limit({"value": 240, "unit": 6, "referenceDatum": 2}) == "FL240"
    assert airspace.format_limit({"value": 0, "unit": 1, "referenceDatum": 0}) == "GND"
    assert airspace.format_limit({"value": 3500, "unit": 1, "referenceDatum": 1}) == "3500 ft AMSL"
    assert airspace.format_limit({"value": 300, "unit": 0, "referenceDatum": 0}) == "300 m AGL"
    assert airspace.format_limit(None) == "—"


def test_normalize_drops_unknown_types_and_rounds() -> None:
    features = airspace.normalize_collection(RAW)

    assert [f["properties"]["name"] for f in features] == ["GAZIANTEP TMA", "ISTANBUL CTR"]
    tma = features[0]
    assert tma["properties"]["lower"] == "3500 ft AMSL"
    assert tma["properties"]["upper"] == "FL240"
    assert tma["geometry"]["coordinates"][0][0] == [37.12346, 36.9]


def test_endpoint_defaults_to_tma_ctr_fir(client: TestClient) -> None:
    data = client.get("/api/v1/airspace").json()

    assert {f["properties"]["category"] for f in data["features"]} == {"TMA", "CTR"}
    assert "AIP" in data["disclaimer"]


def test_endpoint_filters_categories(client: TestClient) -> None:
    data = client.get("/api/v1/airspace?categories=ctr").json()

    assert [f["properties"]["name"] for f in data["features"]] == ["ISTANBUL CTR"]


def test_endpoint_rejects_unknown_category(client: TestClient) -> None:
    assert client.get("/api/v1/airspace?categories=XYZ").status_code == 422


def test_endpoint_reports_unavailable(monkeypatch) -> None:
    async def failing(force: bool = False):
        raise airspace.AirspaceUnavailableError("down")

    monkeypatch.setattr("decideflight.api.airspace.load_airspaces", failing)

    assert TestClient(app).get("/api/v1/airspace").status_code == 503


def test_ui_wires_airspace_layer() -> None:
    index = (STATIC / "index.html").read_text()
    assert "/static/airspace.js" in index
    assert "AirspaceLayer.attach(map)" in index
    assert "'/static/airspace.js'" in (STATIC / "sw.js").read_text()
    assert "/static/radar.js" in index
    assert "'/static/radar.js'" in (STATIC / "sw.js").read_text()

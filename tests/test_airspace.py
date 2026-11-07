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
    assert "/static/wind.js" in index
    assert "/static/airports.js" in index
    assert "'/static/radar.js'" in (STATIC / "sw.js").read_text()


def test_map_layer_grid_and_parsing() -> None:
    from decideflight.services import map_layers

    points = map_layers.grid_points()
    assert len(points) == 5 * 13
    payload = [
        {"hourly": {"time": ["2026-11-07T21:00"], "wind_speed_850hPa": [16.0], "wind_direction_850hPa": [314], "cloud_cover_850hPa": [14]}}
        for _ in points
    ]
    result = map_layers.parse_response("850", points, payload)

    assert result["valid_time_utc"] == "2026-11-07T21:00"
    assert result["points"][0] == {
        "lat": 36.0, "lon": 26.0, "wind_speed_knots": 16.0, "wind_direction_deg": 314, "cloud_cover_pct": 14,
    }
    with pytest.raises(map_layers.MapLayerUnavailableError):
        map_layers.parse_response("850", points, payload[:2])


def test_map_layer_endpoint(monkeypatch) -> None:
    async def fake_layer(level: str):
        return {"level": level, "label": "x", "valid_time_utc": "t", "points": []}

    monkeypatch.setattr("decideflight.api.map_layers.load_layer", fake_layer)
    client = TestClient(app)

    ok = client.get("/api/v1/map-layers/wind-cloud?level=700").json()
    assert ok["level"] == "700" and "Open-Meteo" in ok["source"]
    assert client.get("/api/v1/map-layers/wind-cloud?level=999").status_code == 422


def test_airports_endpoint_lists_turkish_icao_airports() -> None:
    data = TestClient(app).get("/api/v1/map-layers/airports").json()["airports"]
    icaos = {a["icao"] for a in data}

    assert {"LTAC", "LTFM", "LTAI"} <= icaos
    assert all(i.startswith("LT") and len(i) == 4 for i in icaos)
    assert all(-90 <= a["lat"] <= 90 for a in data)


def test_ui_wires_route_and_sigmet_plugins() -> None:
    index = (STATIC / "index.html").read_text()
    sw = (STATIC / "sw.js").read_text()
    for name in ("sigmet.js", "route.js"):
        assert f"/static/{name}" in index
        assert f"'/static/{name}'" in sw


def test_limit_feet_conversion() -> None:
    assert airspace.limit_feet({"value": 240, "unit": 6}) == 24000
    assert airspace.limit_feet({"value": 3500, "unit": 1}) == 3500
    assert airspace.limit_feet({"value": 1000, "unit": 0}) == 3281
    assert airspace.limit_feet(None) is None


def test_flight_categories_and_sigmet_parsing() -> None:
    from decideflight.services import map_layers

    cats = map_layers.parse_flight_categories(
        [{"icaoId": "LTAC", "fltCat": "VFR", "reportTime": "t"}, {"icaoId": "LTXX"}, {"icaoId": "LTBA", "fltCat": "??"}]
    )
    assert cats == {"LTAC": {"category": "VFR", "observed": "t"}}

    near = {"hazard": "TS", "firName": "LTAA", "coords": [{"lat": 39.0, "lon": 32.0}], "rawSigmet": "x"}
    far = {"hazard": "VA", "coords": [{"lat": 14.0, "lon": -90.0}]}
    parsed = map_layers.parse_sigmets([near, far])
    assert len(parsed) == 1 and parsed[0]["hazard"] == "TS" and parsed[0]["coords"] == [[39.0, 32.0]]


def test_status_and_sigmet_endpoints(monkeypatch) -> None:
    async def fake_status():
        return {"LTAC": {"category": "IFR", "observed": None}}

    async def fake_sigmets():
        return []

    monkeypatch.setattr("decideflight.api.map_layers.flight_categories", fake_status)
    monkeypatch.setattr("decideflight.api.map_layers.sigmets", fake_sigmets)
    client = TestClient(app)

    assert client.get("/api/v1/map-layers/airports/status").json()["stations"]["LTAC"]["category"] == "IFR"
    assert client.get("/api/v1/map-layers/sigmet").json()["sigmets"] == []

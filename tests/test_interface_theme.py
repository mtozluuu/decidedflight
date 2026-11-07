import json
from pathlib import Path
from xml.etree import ElementTree

from fastapi.testclient import TestClient

from decideflight.main import app


STATIC = Path(__file__).parents[1] / "src/decideflight/static"


def test_shared_theme_is_served_and_cached_for_all_tabs():
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200
    assert '<link rel="stylesheet" href="/static/theme.css"' in page.text
    theme = client.get("/static/theme.css")
    assert theme.status_code == 200
    assert "text/css" in theme.headers["content-type"]
    assert "color-scheme: dark" in theme.text
    assert ".workspace-nav" in theme.text
    assert "position: fixed" in theme.text
    assert 'id="weatherSnapshot"' in page.text
    assert "'/static/theme.css'" in (STATIC / "sw.js").read_text()
    for name in ("point", "region", "wind", "sources", "settings"):
        assert f'id="panel-{name}"' in page.text
        assert f'id="tab-btn-{name}"' in page.text
    manifest = json.loads((STATIC / "manifest.json").read_text())
    assert manifest["theme_color"] == "#14181d"
    assert manifest["background_color"] == "#14181d"
    ElementTree.fromstring((STATIC / "decideflight-icon.svg").read_text())


def test_theme_retains_accessible_status_colors():
    def luminance(color):
        channels = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [
            value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
            for value in channels
        ]
        return sum(value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722)))

    for foreground, background in (
        ("#edf0f3", "#1b2026"),
        ("#a0aab6", "#1b2026"),
        ("#a6d7bd", "#22372c"),
        ("#e8c587", "#352e22"),
        ("#e5a5a5", "#362629"),
        ("#17202a", "#d4dce5"),
    ):
        light, dark = sorted((luminance(foreground), luminance(background)), reverse=True)
        assert (light + 0.05) / (dark + 0.05) >= 4.5
    theme = (STATIC / "theme.css").read_text()
    assert "prefers-reduced-motion" in theme
    assert "summary:focus-visible" in theme
    assert ".leaflet-tile { filter: brightness(0.75) saturate(0.35); }" in theme

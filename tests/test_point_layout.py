from html.parser import HTMLParser
from pathlib import Path


def test_point_details_are_valid_accessible_and_keep_core_output_visible():
    class Layout(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack = []
            self.elements = {}

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if attrs.get("id"):
                self.elements[attrs["id"]] = (tag, attrs, self.stack.copy())
            if tag not in {
                "area", "base", "br", "col", "embed", "hr", "img",
                "input", "link", "meta", "param", "source", "track", "wbr",
            }:
                self.stack.append((tag, attrs.get("id")))

        def handle_startendtag(self, tag, attrs):
            self.handle_starttag(tag, attrs)
            if self.stack and self.stack[-1] == (tag, dict(attrs).get("id")):
                self.handle_endtag(tag)

        def handle_endtag(self, tag):
            assert self.stack and self.stack[-1][0] == tag, (tag, self.getpos(), self.stack[-1:])
            self.stack.pop()

    layout = Layout()
    path = Path(__file__).parents[1] / "src/decideflight/static/index.html"
    layout.feed(path.read_text())
    assert layout.stack == []
    for name in (
        "airportTools", "parameterDetails", "sourceDetails", "planningTools",
        "aiSummaryCard", "riskCard", "chatSection", "historySection", "feedbackSection",
    ):
        tag, attrs, _ = layout.elements[name]
        assert tag == "details"
        assert "open" not in attrs
    for name in ("decisionBanner", "decisionReasons", "recCard", "pdfBtn", "notamSection"):
        assert not any(tag == "details" for tag, _ in layout.elements[name][2])
    for name in ("forecastTimeline", "flightPlanTimeline"):
        assert ("details", "planningTools") in layout.elements[name][2]
    for name in ("feedbackButtons", "feedbackComment", "feedbackThanks"):
        assert ("details", "feedbackSection") in layout.elements[name][2]
    assert ("details", "sourceDetails") in layout.elements["sourceTimingOverview"][2]
    assert any(tag == "summary" for tag, _ in layout.elements["sourceTimingOverview"][2])
    assert layout.elements["weatherSnapshot"][0] == "aside"
    assert layout.elements["weatherSnapshot"][2][-1] == ("section", "resultSection")
    assert ("aside", "weatherSnapshot") in layout.elements["windTrendBadge"][2]
    for name in ("metricWind", "metricVisibility", "metricTemperature", "metricCloudCover"):
        assert not any(tag == "details" for tag, _ in layout.elements[name][2])

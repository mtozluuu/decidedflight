"""Read latest airport observations from the public MGM Rasat result page."""

from html.parser import HTMLParser
import re

import httpx


class _ObservationParser(HTMLParser):
    def __init__(self, icao: str):
        super().__init__()
        self.icao = icao
        self.in_pre = False
        self.parts: list[str] = []
        self.observations: dict[str, str | None] = {"metar": None, "taf": None}

    def handle_starttag(self, tag, attrs):
        if tag == "pre":
            self.in_pre = True
            self.parts = []

    def handle_data(self, data):
        if self.in_pre:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag != "pre" or not self.in_pre:
            return
        self.in_pre = False
        text = " ".join("".join(self.parts).split())
        for kind in self.observations:
            pattern = rf"^{kind.upper()}(?: (?:AMD|COR))? {self.icao} \d{{6}}Z\b"
            if self.observations[kind] is None and re.match(pattern, text):
                self.observations[kind] = text


async def fetch_mgm_observations(icao: str) -> dict[str, str | None]:
    """MGM's hours=0 page lists the latest observation of each selected type."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            "https://rasat.mgm.gov.tr/result",
            params=[("stations", icao), ("obsType", "1"), ("obsType", "2"), ("hours", "0")],
        )
        response.raise_for_status()
    parser = _ObservationParser(icao)
    parser.feed(response.text)
    return parser.observations

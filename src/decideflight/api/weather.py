"""Weather report API endpoints.

POST /api/v1/weather/report
    Collect multi-source weather data, evaluate drone flight suitability,
    persist the report and return the full result.

GET /api/v1/weather/report/{report_id}/pdf
    Return the stored report as a downloadable PDF.

POST /api/v1/weather/report/{report_id}/feedback
    Store user feedback about whether the decision was correct.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, model_validator
from sqlalchemy.orm import Session

from decideflight.database import SessionLocal
from decideflight.models.feedback import Feedback
from decideflight.models.weather_report import WeatherReport
from decideflight.services.decision_engine import make_decision
from decideflight.services.geocoding import geocode_city
from decideflight.services.report_generator import generate_pdf
from decideflight.services.weather_fetcher import (
    WeatherSourceData,
    fetch_all_sources,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/weather", tags=["weather"])


# ---------------------------------------------------------------------------
# Database dependency
# ---------------------------------------------------------------------------


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class WeatherReportRequest(BaseModel):
    location: str | None = None
    lat: float | None = None
    lon: float | None = None

    @model_validator(mode="after")
    def _check_input(self) -> "WeatherReportRequest":
        if self.location is None and (self.lat is None or self.lon is None):
            raise ValueError(
                "Provide either 'location' (city name) "
                "or both 'lat' and 'lon' coordinates."
            )
        return self


class ParameterResultSchema(BaseModel):
    name: str
    value: str
    decision: str


class WeatherSourceSchema(BaseModel):
    source: str
    wind_speed_knots: float
    temperature_c: float
    humidity_pct: float
    visibility_km: float
    precipitation_level: int
    cloud_base_ft: float | None
    cloud_ceiling_ft: float | None


class WeatherReportResponse(BaseModel):
    report_id: int
    location: str
    lat: float
    lon: float
    sources: list[WeatherSourceSchema]
    decision: str
    decision_detail: str
    parameters: list[ParameterResultSchema]
    created_at: datetime


# ---------------------------------------------------------------------------
# POST /api/v1/weather/report
# ---------------------------------------------------------------------------


@router.post(
    "/report",
    response_model=WeatherReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a drone flight weather report",
)
async def create_weather_report(
    body: WeatherReportRequest,
    db: Session = Depends(get_db),
) -> WeatherReportResponse:
    """Collect weather from all configured sources, evaluate drone flight
    suitability and persist the report.  Returns the full report including
    source data and the UYGUN / RİSKLİ / UYGUN_DEĞİL decision.
    """
    # 1. Resolve coordinates
    if body.lat is not None and body.lon is not None:
        lat, lon = body.lat, body.lon
        location_name = body.location or f"{lat:.4f}, {lon:.4f}"
    else:
        # body.location is guaranteed non-None by the validator
        assert body.location is not None
        try:
            lat, lon, location_name = await geocode_city(body.location)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.error("Geocoding error: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Geocoding service unavailable.",
            ) from exc

    # 2. Fetch weather data
    try:
        sources: list[WeatherSourceData] = await fetch_all_sources(lat, lon)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    # 3. Evaluate decision
    decision_result = make_decision(sources)

    # 4. Persist report
    sources_json = json.dumps(
        [{k: v for k, v in asdict(s).items() if k != "raw"} for s in sources]
    )
    report = WeatherReport(
        location=location_name,
        lat=lat,
        lon=lon,
        decision=decision_result.decision,
        decision_detail=decision_result.detail,
        sources_data=sources_json,
        created_at=datetime.now(timezone.utc),
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    return WeatherReportResponse(
        report_id=report.id,
        location=report.location,
        lat=lat,
        lon=lon,
        sources=[
            WeatherSourceSchema(
                source=s.source,
                wind_speed_knots=s.wind_speed_knots,
                temperature_c=s.temperature_c,
                humidity_pct=s.humidity_pct,
                visibility_km=s.visibility_km,
                precipitation_level=s.precipitation_level,
                cloud_base_ft=s.cloud_base_ft,
                cloud_ceiling_ft=s.cloud_ceiling_ft,
            )
            for s in sources
        ],
        decision=decision_result.decision,
        decision_detail=decision_result.detail,
        parameters=[
            ParameterResultSchema(
                name=p.name,
                value=p.value,
                decision=p.decision,
            )
            for p in decision_result.parameters
        ],
        created_at=report.created_at,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/weather/report/{report_id}/pdf
# ---------------------------------------------------------------------------


@router.get(
    "/report/{report_id}/pdf",
    summary="Download a weather report as PDF",
    response_class=StreamingResponse,
)
async def download_report_pdf(
    report_id: int,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Generate and return a PDF for the stored weather report."""
    report: WeatherReport | None = db.get(WeatherReport, report_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report {report_id} not found.",
        )

    # Reconstruct source data from JSON
    raw_sources: list[dict[str, Any]] = json.loads(report.sources_data)
    sources = [
        WeatherSourceData(
            source=s["source"],
            wind_speed_knots=s["wind_speed_knots"],
            temperature_c=s["temperature_c"],
            humidity_pct=s["humidity_pct"],
            visibility_km=s["visibility_km"],
            precipitation_level=s["precipitation_level"],
            cloud_base_ft=s.get("cloud_base_ft"),
            cloud_ceiling_ft=s.get("cloud_ceiling_ft"),
        )
        for s in raw_sources
    ]

    decision_result = make_decision(sources)

    pdf_bytes = generate_pdf(
        location=report.location,
        lat=report.lat,
        lon=report.lon,
        sources=sources,
        decision_result=decision_result,
        created_at=report.created_at,
    )

    filename = f"decidedflight_report_{report_id}.pdf"
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Feedback schemas
# ---------------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    correct: bool
    user_comment: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: int
    report_id: int
    correct: bool
    created_at: datetime


# ---------------------------------------------------------------------------
# POST /api/v1/weather/report/{report_id}/feedback
# ---------------------------------------------------------------------------


@router.post(
    "/report/{report_id}/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit feedback for a weather report decision",
)
async def submit_feedback(
    report_id: int,
    body: FeedbackRequest,
    db: Session = Depends(get_db),
) -> FeedbackResponse:
    """Store user feedback indicating whether the decision was correct.

    The feedback is persisted and later used by the AI decision engine to
    improve future recommendations.
    """
    report: WeatherReport | None = db.get(WeatherReport, report_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report {report_id} not found.",
        )

    fb = Feedback(
        report_id=report_id,
        correct=body.correct,
        user_comment=body.user_comment,
    )
    db.add(fb)
    db.commit()
    db.refresh(fb)

    return FeedbackResponse(
        feedback_id=fb.id,
        report_id=fb.report_id,
        correct=fb.correct,
        created_at=fb.created_at,
    )

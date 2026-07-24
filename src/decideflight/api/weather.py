"""Weather report API endpoints.

POST /api/v1/weather/report
    Collect multi-source weather data, evaluate drone flight suitability,
    persist the report and return the full result.

GET /api/v1/weather/report/{report_id}/pdf
    Return the stored report as a downloadable PDF.
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
from decideflight.models.weather_report import WeatherReport
from decideflight.services.ai_decision_engine import (
    AIDecisionResult,
    make_ai_decision,
    reconstruct_ai_decision,
)
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
    # AI-specific fields (None when falling back to rule-based engine)
    confidence: int | None = None
    summary: str | None = None
    detailed_analysis: str | None = None
    risk_factors: list[str] | None = None
    recommendations: list[str] | None = None


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

    # 3. Evaluate decision (AI engine with rule-based fallback)
    decision_result = await make_ai_decision(sources)

    # 4. Persist report
    ai_analysis_json: str | None = None
    if isinstance(decision_result, AIDecisionResult):
        import json as _json

        ai_extra = decision_result.ai_extra_as_dict()
        ai_extra["decision"] = decision_result.decision
        ai_analysis_json = _json.dumps(ai_extra, ensure_ascii=False)

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
        ai_analysis_data=ai_analysis_json,
        created_at=datetime.now(timezone.utc),
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    # Build AI fields for response
    ai_confidence: int | None = None
    ai_summary: str | None = None
    ai_detailed: str | None = None
    ai_risks: list[str] | None = None
    ai_recs: list[str] | None = None
    if isinstance(decision_result, AIDecisionResult):
        ai_confidence = decision_result.confidence
        ai_summary = decision_result.summary
        ai_detailed = decision_result.detailed_analysis
        ai_risks = decision_result.risk_factors
        ai_recs = decision_result.recommendations

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
        confidence=ai_confidence,
        summary=ai_summary,
        detailed_analysis=ai_detailed,
        risk_factors=ai_risks,
        recommendations=ai_recs,
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

    decision_result = reconstruct_ai_decision(sources, report.ai_analysis_data)

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

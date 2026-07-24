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

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, model_validator
from sqlalchemy.orm import Session

from decideflight.database import SessionLocal
from decideflight.models.feedback import Feedback
from decideflight.models.weather_report import WeatherReport
from decideflight.services.ai_decision_engine import (
    AIDecisionResult,
    build_feedback_context,
    make_ai_decision,
)
from decideflight.services.decision_engine import make_decision
from decideflight.services.geocoding import geocode_city
from decideflight.services.report_generator import generate_pdf
from decideflight.services.trend_analyzer import fetch_wind_trend
from decideflight.services.weather_fetcher import (
    WeatherSourceData,
    fetch_all_sources,
)

try:
    from openai import AsyncOpenAI as _AsyncOpenAI  # type: ignore[import]
except ImportError:  # pragma: no cover
    _AsyncOpenAI = None  # type: ignore[assignment,misc]

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
    confidence_score: int
    parameters: list[ParameterResultSchema]
    created_at: datetime
    # AI-enhanced fields (present when OPENAI_API_KEY is set)
    confidence: int | None = None
    summary: str | None = None
    detailed_analysis: str | None = None
    risk_factors: list[str] | None = None
    recommendations: list[str] | None = None
    wind_trend: str | None = None


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

    # 3. Fetch wind trend (non-blocking — failure just omits trend)
    wind_trend_obj = await fetch_wind_trend(lat, lon)
    wind_trend_str = wind_trend_obj.description if wind_trend_obj else ""

    # 4. Get feedback context from DB (Feature 4)
    feedback_ctx = build_feedback_context(db)

    # 5. Evaluate decision (AI if key present, rule-based fallback)
    decision_result = await make_ai_decision(
        sources=sources,
        location=location_name,
        lat=lat,
        lon=lon,
        feedback_context=feedback_ctx,
        wind_trend=wind_trend_str,
    )

    # 6. Persist report
    ai_data: str | None = None
    if isinstance(decision_result, AIDecisionResult) and (
        decision_result.summary or decision_result.detailed_analysis
    ):
        ai_data = json.dumps(
            {
                "confidence": decision_result.confidence,
                "summary": decision_result.summary,
                "detailed_analysis": decision_result.detailed_analysis,
                "risk_factors": decision_result.risk_factors,
                "recommendations": decision_result.recommendations,
                "parameter_assessments": decision_result.parameter_assessments,
            },
            ensure_ascii=False,
        )

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
        ai_analysis_data=ai_data,
        created_at=datetime.now(timezone.utc),
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    # AI fields for response
    ai_confidence: int | None = None
    ai_summary: str | None = None
    ai_detailed: str | None = None
    ai_risks: list[str] | None = None
    ai_recs: list[str] | None = None
    if isinstance(decision_result, AIDecisionResult) and decision_result.summary:
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
        confidence_score=decision_result.confidence_score,
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
        wind_trend=wind_trend_str if wind_trend_str else None,
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


# ---------------------------------------------------------------------------
# Grid analysis schemas
# ---------------------------------------------------------------------------


class DroneLimit(BaseModel):
    """Configurable drone flight limits used by the grid decision engine."""

    wind_ok_max_knots: float = 15.0
    wind_risky_max_knots: float = 25.0
    visibility_ok_min_km: float = 5.0
    visibility_risky_min_km: float = 1.0
    cloud_base_ok_min_ft: float = 500.0
    cloud_base_risky_min_ft: float = 200.0
    cloud_ceiling_ok_min_ft: float = 1000.0
    cloud_ceiling_risky_min_ft: float = 500.0
    humidity_ok_max_pct: float = 85.0
    humidity_risky_max_pct: float = 95.0
    temp_ok_min_c: float = 0.0
    temp_ok_max_c: float = 40.0


class GridAnalysisRequest(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    grid_size: int = 25  # 25 (5x5) or 100 (10x10)
    altitude_ft: float = 1000.0
    hours: int = 1  # 1 = instant, 24 = next 24 h
    limits: DroneLimit = DroneLimit()

    @model_validator(mode="after")
    def _validate(self) -> "GridAnalysisRequest":
        if self.grid_size not in (25, 100):
            raise ValueError("grid_size must be 25 or 100")
        if not (0 <= self.hours <= 24):
            raise ValueError("hours must be between 0 and 24")
        return self


class GridPointWeather(BaseModel):
    lat: float
    lon: float
    decision: str
    wind_speed_knots: float
    temperature_c: float
    humidity_pct: float
    visibility_km: float
    cloud_base_ft: float | None
    cloud_ceiling_ft: float | None
    precipitation_level: int
    cloud_cover_pct: float = 0.0


class GridSummary(BaseModel):
    UYGUN: int
    RISKLI: int
    UYGUN_DEGIL: int
    total: int


class GridAnalysisResponse(BaseModel):
    points: list[GridPointWeather]
    summary: GridSummary
    grid_size: int
    altitude_ft: float


# ---------------------------------------------------------------------------
# Grid analysis helpers
# ---------------------------------------------------------------------------

_UYGUN = "UYGUN"
_RISKLI = "RISKLI"
_UYGUN_DEGIL = "UYGUN_DEGIL"


def _altitude_to_pressure_hpa(altitude_ft: float) -> int | None:
    """Map altitude (ft) to the nearest Open-Meteo pressure level (hPa).

    Returns None for surface altitudes (<= ~1500 ft) so the caller uses
    the standard 10 m wind measurement instead.
    """
    altitude_m = altitude_ft * 0.3048
    if altitude_m <= 457:  # <= ~1500 ft
        return None
    # Open-Meteo levels with approximate median altitudes (m)
    levels = [
        (925, 762),
        (850, 1457),
        (700, 3012),
        (600, 4206),
        (500, 5574),
    ]
    return min(levels, key=lambda lv: abs(lv[1] - altitude_m))[0]


async def _fetch_grid_point_weather(
    lat: float,
    lon: float,
    altitude_ft: float,
    hours: int,
    client: httpx.AsyncClient,
) -> dict[str, Any] | None:
    """Fetch Open-Meteo weather for one grid point.

    When ``hours`` is 0 or 1 the current conditions are returned.
    For ``hours > 1`` the hourly forecast is fetched and data at the
    requested hour offset (used as array index) is returned.

    For altitudes above ~1500 ft the nearest pressure-level wind speed is
    requested and used in place of the 10 m wind.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    pressure_hpa = _altitude_to_pressure_hpa(altitude_ft)
    use_hourly = hours > 1

    if not use_hourly:
        # Current-conditions path (existing behaviour)
        params: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "current": (
                "temperature_2m,relative_humidity_2m,dew_point_2m,"
                "wind_speed_10m,precipitation,visibility,cloud_cover"
            ),
            "wind_speed_unit": "kmh",
            "timezone": "UTC",
        }
        if pressure_hpa is not None:
            params["hourly"] = f"wind_speed_{pressure_hpa}hPa"
            params["forecast_hours"] = 1
    else:
        # Hourly forecast path — request all needed variables at once
        hourly_vars = (
            "temperature_2m,relative_humidity_2m,dew_point_2m,"
            "wind_speed_10m,precipitation,visibility,cloud_cover"
        )
        if pressure_hpa is not None:
            hourly_vars += f",wind_speed_{pressure_hpa}hPa"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": hourly_vars,
            "wind_speed_unit": "kmh",
            "timezone": "UTC",
            "forecast_hours": hours + 1,
        }

    try:
        resp = await client.get(url, params=params, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning(
            "Grid Open-Meteo fetch failed at (%.4f, %.4f): %s",
            lat,
            lon,
            exc,
        )
        return None

    if not use_hourly:
        cur = data.get("current", {})
        temp_c: float | None = cur.get("temperature_2m")
        if temp_c is None:
            return None

        dew_c: float | None = cur.get("dew_point_2m")
        wind_kmh: float = cur.get("wind_speed_10m", 0.0)
        rh: float = cur.get("relative_humidity_2m", 0.0)
        precip_mm: float = cur.get("precipitation", 0.0)
        vis_m: float = cur.get("visibility", 10000.0)
        cloud_pct: float = cur.get("cloud_cover", 0.0)

        wind_knots = wind_kmh * 0.539957

        if pressure_hpa is not None:
            hourly = data.get("hourly", {})
            pl_wind_list = hourly.get(f"wind_speed_{pressure_hpa}hPa", [])
            if pl_wind_list:
                wind_knots = float(pl_wind_list[0]) * 0.539957
        else:
            # Power-law wind profile: V(z) = V_ref * (z/z_ref)^(1/7)
            # Exponent 1/7 (~0.143) is the standard value for open terrain
            # (Hellmann exponent); z_ref = 10 m (Open-Meteo measurement height).
            alt_m = altitude_ft * 0.3048
            ref_m = 10.0
            if alt_m > ref_m:
                wind_knots *= (alt_m / ref_m) ** (1.0 / 7.0)
    else:
        # Parse hourly arrays at the requested index
        hourly = data.get("hourly", {})
        temp_list: list[Any] = hourly.get("temperature_2m", [])
        if not temp_list or len(temp_list) <= hours:
            return None
        idx = hours

        temp_c = temp_list[idx]
        if temp_c is None:
            return None

        def _h(key: str, default: float) -> float:
            lst = hourly.get(key, [])
            val = lst[idx] if len(lst) > idx else None
            return float(val) if val is not None else default

        dew_c_val = hourly.get("dew_point_2m", [])
        dew_c = (
            float(dew_c_val[idx])
            if len(dew_c_val) > idx and dew_c_val[idx] is not None
            else None
        )
        wind_kmh = _h("wind_speed_10m", 0.0)
        rh = _h("relative_humidity_2m", 0.0)
        precip_mm = _h("precipitation", 0.0)
        vis_m = _h("visibility", 10000.0)
        cloud_pct = _h("cloud_cover", 0.0)

        wind_knots = wind_kmh * 0.539957

        if pressure_hpa is not None:
            pl_wind_list = hourly.get(f"wind_speed_{pressure_hpa}hPa", [])
            if len(pl_wind_list) > idx and pl_wind_list[idx] is not None:
                wind_knots = float(pl_wind_list[idx]) * 0.539957
        else:
            alt_m = altitude_ft * 0.3048
            ref_m = 10.0
            if alt_m > ref_m:
                wind_knots *= (alt_m / ref_m) ** (1.0 / 7.0)

    base_ft: float | None = None
    if dew_c is not None:
        spread = max(temp_c - dew_c, 0.0)
        # Standard approximation: ~122.5 m per degC of T/Td spread
        base_ft = spread * 122.5 * 3.28084

    ceiling_ft: float | None = None
    if base_ft is not None and cloud_pct > 50:
        # Heuristic: when cloud cover > 50 % the ceiling is just above the
        # estimated base (200 ft offset mirrors the existing weather_fetcher).
        ceiling_ft = base_ft + 200.0

    if precip_mm <= 0:
        precip_level = 0
    elif precip_mm < 2.5:
        precip_level = 1
    else:
        precip_level = 2

    return {
        "wind_speed_knots": round(wind_knots, 2),
        "temperature_c": temp_c,
        "humidity_pct": rh,
        "visibility_km": vis_m / 1000.0,
        "cloud_base_ft": base_ft,
        "cloud_ceiling_ft": ceiling_ft,
        "precipitation_level": precip_level,
        "cloud_cover_pct": cloud_pct,
    }


def _eval_grid_decision(
    wind_speed_knots: float,
    visibility_km: float,
    temperature_c: float,
    humidity_pct: float,
    cloud_base_ft: float | None,
    cloud_ceiling_ft: float | None,
    precipitation_level: int,
    limits: DroneLimit,
) -> str:
    """Return UYGUN / RISKLI / UYGUN_DEGIL for one grid point."""
    scores: list[int] = []

    # Wind
    if wind_speed_knots < limits.wind_ok_max_knots:
        scores.append(0)
    elif wind_speed_knots <= limits.wind_risky_max_knots:
        scores.append(1)
    else:
        scores.append(2)

    # Visibility
    if visibility_km > limits.visibility_ok_min_km:
        scores.append(0)
    elif visibility_km >= limits.visibility_risky_min_km:
        scores.append(1)
    else:
        scores.append(2)

    # Precipitation
    scores.append(min(precipitation_level, 2))

    # Temperature
    if limits.temp_ok_min_c <= temperature_c <= limits.temp_ok_max_c:
        scores.append(0)
    elif (limits.temp_ok_min_c - 5) <= temperature_c <= (limits.temp_ok_max_c + 5):
        scores.append(1)
    else:
        scores.append(2)

    # Humidity
    if humidity_pct < limits.humidity_ok_max_pct:
        scores.append(0)
    elif humidity_pct <= limits.humidity_risky_max_pct:
        scores.append(1)
    else:
        scores.append(2)

    # Cloud base
    if cloud_base_ft is not None:
        if cloud_base_ft > limits.cloud_base_ok_min_ft:
            scores.append(0)
        elif cloud_base_ft >= limits.cloud_base_risky_min_ft:
            scores.append(1)
        else:
            scores.append(2)

    # Cloud ceiling
    if cloud_ceiling_ft is not None:
        if cloud_ceiling_ft > limits.cloud_ceiling_ok_min_ft:
            scores.append(0)
        elif cloud_ceiling_ft >= limits.cloud_ceiling_risky_min_ft:
            scores.append(1)
        else:
            scores.append(2)

    worst = max(scores) if scores else 0
    return [_UYGUN, _RISKLI, _UYGUN_DEGIL][worst]


# ---------------------------------------------------------------------------
# POST /api/v1/weather/grid
# ---------------------------------------------------------------------------


@router.post(
    "/grid",
    response_model=GridAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse weather for a geographic grid",
)
async def analyse_weather_grid(
    body: GridAnalysisRequest,
) -> GridAnalysisResponse:
    """Fetch Open-Meteo weather for a uniform grid of points within the
    specified bounding box and evaluate drone flight suitability at each
    point using the provided (or default) drone limits.

    Grid sizes: 25 points (5x5) or 100 points (10x10).
    All grid-point fetches are issued concurrently via ``asyncio.gather``.
    """
    n = {25: 5, 100: 10}[body.grid_size]

    if n > 1:
        lats = [
            body.lat_min + (body.lat_max - body.lat_min) * i / (n - 1) for i in range(n)
        ]
        lons = [
            body.lon_min + (body.lon_max - body.lon_min) * j / (n - 1) for j in range(n)
        ]
    else:
        lats = [body.lat_min]
        lons = [body.lon_min]

    grid_coords = [(lat, lon) for lat in lats for lon in lons]

    async with httpx.AsyncClient() as client:
        tasks = [
            _fetch_grid_point_weather(lat, lon, body.altitude_ft, body.hours, client)
            for lat, lon in grid_coords
        ]
        raw_results: list[Any] = await asyncio.gather(*tasks, return_exceptions=True)

    points: list[GridPointWeather] = []
    for (lat, lon), result in zip(grid_coords, raw_results):
        if isinstance(result, Exception):
            logger.warning("Grid point (%.4f, %.4f) raised: %s", lat, lon, result)
            continue
        if result is None:
            continue

        decision = _eval_grid_decision(
            wind_speed_knots=result["wind_speed_knots"],
            visibility_km=result["visibility_km"],
            temperature_c=result["temperature_c"],
            humidity_pct=result["humidity_pct"],
            cloud_base_ft=result["cloud_base_ft"],
            cloud_ceiling_ft=result["cloud_ceiling_ft"],
            precipitation_level=result["precipitation_level"],
            limits=body.limits,
        )

        points.append(
            GridPointWeather(
                lat=lat,
                lon=lon,
                decision=decision,
                wind_speed_knots=result["wind_speed_knots"],
                temperature_c=result["temperature_c"],
                humidity_pct=result["humidity_pct"],
                visibility_km=result["visibility_km"],
                cloud_base_ft=result["cloud_base_ft"],
                cloud_ceiling_ft=result["cloud_ceiling_ft"],
                precipitation_level=result["precipitation_level"],
                cloud_cover_pct=result.get("cloud_cover_pct", 0.0),
            )
        )

    if not points:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Belirtilen bölge için hava verisi alınamadı.",
        )

    n_uygun = sum(1 for p in points if p.decision == _UYGUN)
    n_riskli = sum(1 for p in points if p.decision == _RISKLI)
    n_uygun_degil = sum(1 for p in points if p.decision == _UYGUN_DEGIL)

    return GridAnalysisResponse(
        points=points,
        summary=GridSummary(
            UYGUN=n_uygun,
            RISKLI=n_riskli,
            UYGUN_DEGIL=n_uygun_degil,
            total=len(points),
        ),
        grid_size=body.grid_size,
        altitude_ft=body.altitude_ft,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/weather/grid/summary
# ---------------------------------------------------------------------------


class GridSummaryRequest(BaseModel):
    points: list[GridPointWeather]
    summary: GridSummary
    altitude_ft: float = 1000.0
    location_hint: str = ""


class GridSummaryResponse(BaseModel):
    ai_summary: str


@router.post(
    "/grid/summary",
    response_model=GridSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get an AI-generated Turkish summary for a grid analysis",
)
async def grid_summary(body: GridSummaryRequest) -> GridSummaryResponse:
    """Call GPT-4o (or fall back to rule-based text) with aggregate grid stats
    and return a single Turkish paragraph summarising the region for drone flight.
    """
    import os

    api_key = os.environ.get("OPENAI_API_KEY", "")
    summary_text = await _build_grid_ai_summary(body, api_key)
    return GridSummaryResponse(ai_summary=summary_text)


async def _build_grid_ai_summary(body: GridSummaryRequest, api_key: str) -> str:
    """Return AI or rule-based Turkish grid summary."""
    points = body.points
    summary = body.summary

    if not points:
        return "Bölge için yeterli veri bulunamadı."

    # Compute aggregate stats
    avg_wind = sum(p.wind_speed_knots for p in points) / len(points)
    avg_temp = sum(p.temperature_c for p in points) / len(points)
    avg_humidity = sum(p.humidity_pct for p in points) / len(points)
    avg_vis = sum(p.visibility_km for p in points) / len(points)
    pct_uygun = round(summary.UYGUN / summary.total * 100) if summary.total else 0

    if not api_key or _AsyncOpenAI is None:
        return _rule_based_grid_summary(
            summary, avg_wind, avg_temp, avg_humidity, avg_vis, pct_uygun
        )

    try:
        client = _AsyncOpenAI(api_key=api_key)
        pct_r = round(summary.RISKLI / summary.total * 100) if summary.total else 0
        pct_d = round(summary.UYGUN_DEGIL / summary.total * 100) if summary.total else 0
        prompt = (
            "Bir drone uçuş güvenlik uzmanı olarak aşağıdaki bölge analizini "
            "değerlendir.\n\n"
            f"Konum ipucu: {body.location_hint or 'Belirtilmedi'}\n"
            f"İrtifa: {body.altitude_ft:.0f} ft\n"
            f"Toplam nokta: {summary.total}\n"
            f"  UYGUN: {summary.UYGUN} (%{pct_uygun})\n"
            f"  RISKLI: {summary.RISKLI} (%{pct_r})\n"
            f"  UYGUN_DEGIL: {summary.UYGUN_DEGIL} (%{pct_d})\n"
            f"Ortalama rüzgar: {avg_wind:.1f} knot\n"
            f"Ortalama sıcaklık: {avg_temp:.1f} °C\n"
            f"Ortalama nem: {avg_humidity:.0f}%\n"
            f"Ortalama görüş: {avg_vis:.1f} km\n\n"
            "Bu bölge için drone uçuşunu Türkçe 2-3 cümleyle değerlendir. "
            "Sadece düz metin döndür, JSON değil."
        )
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        return (response.choices[0].message.content or "").strip()

    except Exception as exc:
        logger.warning("GPT grid summary failed, using rule-based: %s", exc)
        return _rule_based_grid_summary(
            summary, avg_wind, avg_temp, avg_humidity, avg_vis, pct_uygun
        )


def _rule_based_grid_summary(
    summary: GridSummary,
    avg_wind: float,
    avg_temp: float,
    avg_humidity: float,
    avg_vis: float,
    pct_uygun: int,
) -> str:
    """Generate a Turkish rule-based grid summary paragraph."""
    total = summary.total
    if pct_uygun >= 70:
        overall = (
            "Bölgenin büyük çoğunluğu drone uçuşu için UYGUN koşullar göstermektedir."
        )
    elif pct_uygun >= 40:
        overall = (
            "Bölgede karma koşullar gözlemlenmektedir;"
            " uçuş planlaması dikkat gerektirir."
        )
    else:
        overall = (
            "Bölgenin önemli bir kısmı drone uçuşu için UYGUN DEĞİL koşullar "
            "içermektedir."
        )

    stats = (
        f"{total} noktanın {summary.UYGUN}'i uygun (%{pct_uygun}), "
        f"{summary.RISKLI}'si riskli, {summary.UYGUN_DEGIL}'i uygun değil "
        f"olarak değerlendirildi. "
        f"Ortalama rüzgar {avg_wind:.1f} knot, görüş {avg_vis:.1f} km, "
        f"nem %{avg_humidity:.0f}, sıcaklık {avg_temp:.1f} °C."
    )

    return f"{overall} {stats}"

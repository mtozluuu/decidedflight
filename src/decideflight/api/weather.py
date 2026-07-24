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
    confidence_score: int
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
        if self.hours not in (1, 24):
            raise ValueError("hours must be 1 or 24")
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
    hours: int,  # noqa: ARG001 -- future: aggregate over next N hours
    client: httpx.AsyncClient,
) -> dict[str, Any] | None:
    """Fetch Open-Meteo current weather for one grid point.

    For altitudes above ~1500 ft the nearest pressure-level wind speed is
    requested via the hourly forecast.  Returns a normalised dict or None.

    ``hours`` is accepted for API compatibility but not yet used; a future
    update will aggregate forecasts over the next 1 or 24 hours.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    pressure_hpa = _altitude_to_pressure_hpa(altitude_ft)

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
        # Use pressure-level wind when altitude warrants it
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

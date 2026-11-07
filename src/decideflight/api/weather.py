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
import math
import os
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator
from decideflight.services.mgm_observations import fetch_mgm_observations
from decideflight.services.airgram import fetch_airgram
from decideflight.services.weather_cache import WeatherCache
from decideflight.services.data_freshness import observation_timing, source_timing
from decideflight.services.flight_limits import DroneLimit
from decideflight.services.weather_fetcher import airport_location
from fastapi.responses import Response
from sqlalchemy.orm import Session

from decideflight.config import settings
from decideflight.database import SessionLocal
from decideflight.models.feedback import Feedback
from decideflight.models.weather_report import WeatherReport
from decideflight.services.ai_decision_engine import (
    AIDecisionResult,
    _AsyncOpenAI,
    build_feedback_context,
    chat_about_report,
    make_ai_decision,
)
from decideflight.services.decision_engine import DecisionResult, ParameterResult, make_decision
from decideflight.services.decision_engine import compute_wind_components, usable_sources, wind_consensus
from decideflight.services.geocoding import geocode_city
from decideflight.services.report_generator import generate_pdf
from decideflight.services.trend_analyzer import fetch_wind_trend
from decideflight.services.weather_fetcher import (
    AirQualityData,
    WeatherSourceData,
    _fetch_with_retry,
    _precip_level,
    fetch_air_quality,
    fetch_all_sources,
    fetch_nearby_runways,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/weather", tags=["weather"])
_hourly_cache: WeatherCache[dict[str, Any]] = WeatherCache(ttl=120)
_seasonal_cache: WeatherCache[SeasonalResponse] = WeatherCache(ttl=86400)


def _round_optional(value: float | None, digits: int = 1) -> float | None:
    return round(value, digits) if value is not None else None

# Grid analysis: batch size and inter-batch delay to avoid rate limits
_GRID_BATCH_SIZE = 5
_GRID_BATCH_DELAY = 0.5
_GRID_NO_DATA_SUMMARY = "Bölge için yeterli veri bulunamadı."

# Seasonal analysis: climate zone latitude thresholds
_TROPICAL_LAT_THRESHOLD = 23.5
_POLAR_LAT_THRESHOLD = 60.0

# Seasonal analysis: precipitation-to-visibility thresholds (mm/day)
_PRECIP_LIGHT_MM = 2.0
_PRECIP_MODERATE_MM = 8.0

# Seasonal analysis: fallback temperature when no data available
_DEFAULT_TEMP_C = 20.0


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
    limits: DroneLimit | None = None
    location: str | None = None
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)

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
    explanation: str = ""


class WeatherSourceSchema(BaseModel):
    source: str
    data_time: datetime | None = None
    time_kind: str = "observation"
    freshness: str = "unknown"
    freshness_message: str = "Veri zamanı doğrulanamadı."
    cloud_cover_pct: float | None = None
    wind_speed_knots: float | None
    temperature_c: float | None
    humidity_pct: float | None
    visibility_km: float | None
    precipitation_level: int | None
    cloud_base_ft: float | None
    cloud_ceiling_ft: float | None
    wind_direction_deg: float | None = None
    data_kind: str = "unknown"
    station_distance_km: float | None = None
    ceiling_unlimited: bool = False
    clouds_clear: bool = False


class AirportObservationsResponse(BaseModel):
    icao: str
    metar: str | None
    taf: str | None
    checked_at: datetime
    metar_timing: dict
    taf_timing: dict


@router.get("/airgram")
async def point_airgram(
    lat: float = Query(ge=-90, le=90),
    lon: float = Query(ge=-180, le=180),
) -> dict:
    try:
        return await fetch_airgram(lat, lon)
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Airgram fetch failed: %s", exc)
        raise HTTPException(
            status_code=503, detail="Open-Meteo Airgram verisi alınamadı."
        ) from exc


@router.get("/airport-location/{icao}")
async def selected_airport_location(icao: str) -> dict:
    icao = icao.strip().upper()
    if len(icao) != 4 or not icao.isascii() or not icao.isalpha():
        raise HTTPException(status_code=422, detail="Dört harfli ICAO kodu girin.")
    location = airport_location(icao)
    if location is None:
        raise HTTPException(status_code=404, detail="Meydan konumu yerel katalogda bulunamadı.")
    return location


@router.get("/airport-airgram/{icao}")
async def mgm_airgram(icao: str) -> Response:
    icao = icao.upper()
    if len(icao) != 4 or not icao.isascii() or not icao.isalpha():
        raise HTTPException(status_code=422, detail="Dört harfli ICAO kodu girin.")
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                "https://hezarfen.mgm.gov.tr/Genel/pngGetir.ashx",
                params={"syol": f"{icao}.png"},
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("MGM Airgram failed for %s: %s", icao, exc)
        raise HTTPException(status_code=503, detail="MGM Airgram alınamadı.") from exc
    if not response.content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise HTTPException(status_code=404, detail="Meydan için MGM Airgram bulunamadı.")
    return Response(response.content, media_type="image/png")


@router.get("/airport-observations/{icao}", response_model=AirportObservationsResponse)
async def airport_observations(icao: str) -> AirportObservationsResponse:
    icao = icao.strip().upper()
    if len(icao) != 4 or not icao.isascii() or not icao.isalpha():
        raise HTTPException(status_code=422, detail="Dört harfli ICAO kodu girin.")
    try:
        observations = await fetch_mgm_observations(icao)
    except httpx.HTTPError as exc:
        logger.warning("MGM Rasat request failed for %s: %s", icao, exc)
        raise HTTPException(
            status_code=503, detail="MGM Rasat verisi alınamadı. Tekrar deneyin."
        ) from exc
    if not any(observations.values()):
        raise HTTPException(
            status_code=404,
            detail=f"MGM Rasat sayfasında {icao} için METAR/TAF bulunamadı.",
        )
    now = datetime.now(timezone.utc)
    return AirportObservationsResponse(
        icao=icao, checked_at=now, **observations,
        metar_timing=observation_timing(observations["metar"], now),
        taf_timing=observation_timing(observations["taf"], now, taf=True),
    )


class WeatherReportResponse(BaseModel):
    applied_limits: dict | None = None
    is_saved_report: bool = False
    parameter_snapshot_available: bool = True
    source_checks: list[DataSourceCheckSchema] | None = None
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
    parameter_weights: dict[str, int] | None = None
    aqi_score: int | None = None
    pm25: float | None = None
    pm10: float | None = None
    missing_parameters: list[str] = Field(default_factory=list)
    known_decision: str | None = None
    excluded_sources: list[str] = Field(default_factory=list)
    data_quality: dict[str, Any] | None = None
    wind_consensus: dict[str, float | int] | None = None


class DataSourceCheckSchema(BaseModel):
    name: str
    provider: str
    data: str
    status: str
    message: str


class DataSourceStatusResponse(BaseModel):
    lat: float
    lon: float
    checked_at: datetime
    checks: list[DataSourceCheckSchema]


# ---------------------------------------------------------------------------
# GET /api/v1/weather/sources/status
# ---------------------------------------------------------------------------


@router.get(
    "/sources/status",
    response_model=DataSourceStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Check weather data source availability",
)
async def data_source_status(
    lat: float = 39.0,
    lon: float = 35.0,
) -> DataSourceStatusResponse:
    """Probe configured weather sources and report which returned data."""
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Coordinates are outside the valid range.",
        )

    try:
        sources, air_quality, wind_trend = await asyncio.gather(
            fetch_all_sources(lat, lon),
            fetch_air_quality(lat, lon),
            fetch_wind_trend(lat, lon),
            return_exceptions=True,
        )
    except Exception as exc:  # pragma: no cover - gather returns exceptions
        logger.exception("Unexpected data source status check failure")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data source status check failed.",
        ) from exc

    return DataSourceStatusResponse(
        lat=lat,
        lon=lon,
        checked_at=datetime.now(timezone.utc),
        checks=_build_source_checks(sources, air_quality, wind_trend),
    )


def _build_source_checks(
    sources: list[WeatherSourceData] | Exception,
    air_quality: AirQualityData | Exception | None,
    wind_trend: Any,
) -> list[DataSourceCheckSchema]:
    if isinstance(sources, Exception):
        logger.warning("Weather source status check failed: %s", sources)
        received_sources: set[str] = set()
    else:
        received_sources = {source.source for source in sources}

    def _check(
        name: str,
        provider: str,
        data: str,
        *,
        source_name: str | None = None,
        configured: bool = True,
        in_coverage: bool = True,
        received: bool | None = None,
    ) -> DataSourceCheckSchema:
        if not configured:
            return DataSourceCheckSchema(
                name=name,
                provider=provider,
                data=data,
                status="not_configured",
                message="API anahtarı tanımlı değil.",
            )
        if not in_coverage:
            return DataSourceCheckSchema(
                name=name,
                provider=provider,
                data=data,
                status="outside_coverage",
                message="Bu kaynak yalnızca Türkiye koordinatlarında sorgulanır.",
            )
        did_receive = (
            source_name in received_sources if received is None else received
        )
        if did_receive:
            timing_message = "Veri alındı; veri zamanı ve güncelliği doğrulanamadı."
            if not isinstance(sources, Exception):
                match = next((s for s in sources if s.source == source_name), None)
                if match is not None:
                    timing = source_timing(match, datetime.now(timezone.utc))
                    timing_message = timing["freshness_message"]
                    if timing["data_time"]:
                        timing_message = f'Veri zamanı: {timing["data_time"]}. {timing_message}'
            return DataSourceCheckSchema(
                name=name,
                provider=provider,
                data=data,
                status="received",
                message=timing_message,
            )
        return DataSourceCheckSchema(
            name=name,
            provider=provider,
            data=data,
            status="no_data",
            message=(
                "Veri alınamadı; bu kaynak değerlendirmeye katılmadı. "
                "Bağlantı, API anahtarı veya kaynak kapsaması kontrol edilmeli; "
                "kesin hata nedeni bu sonuçtan belirlenemiyor."
            ),
        )

    checks = [
        _check(
            "Anlık hava durumu",
            "Open-Meteo",
            "Sıcaklık, rüzgar, nem, görüş, yağış ve bulutluluk",
            source_name="Open-Meteo",
        ),
        _check(
            "Hava kalitesi",
            "Open-Meteo",
            "Avrupa AQI, PM2.5 ve PM10",
            received=not isinstance(air_quality, Exception) and air_quality is not None,
        ),
        _check(
            "Rüzgar ve tahmin",
            "Open-Meteo",
            "Saatlik rüzgar verisi ve son 6 saatlik trend",
            received=not isinstance(wind_trend, Exception) and wind_trend is not None,
        ),
        _check(
            "Anlık hava durumu",
            "OpenWeatherMap",
            "Sıcaklık, rüzgar, nem, görüş, yağış ve bulutluluk",
            source_name="OpenWeatherMap",
            configured=bool(settings.openweathermap_api_key),
        ),
        _check(
            "Anlık hava durumu",
            "WeatherAPI",
            "Sıcaklık, rüzgar, nem, görüş, yağış ve bulutluluk",
            source_name="WeatherAPI",
            configured=bool(settings.weatherapi_api_key),
        ),
        _check(
            "Model tahmini",
            "Windy",
            "GFS rüzgar, sıcaklık ve nem tahmini",
            source_name="Windy",
            configured=bool(settings.windy_api_key),
        ),
        _check(
            "METAR",
            "NOAA Aviation Weather",
            "En yakın havalimanının güncel METAR gözlemi",
            source_name="METAR (NOAA/aviationweather.gov)",
        ),
        _check(
            "METAR",
            "CheckWX",
            "En yakın havalimanının güncel METAR gözlemi",
            source_name="METAR (CheckWX)",
            configured=bool(settings.checkwx_api_key),
        ),
    ]

    return checks


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

    # 2. Fetch weather data and side-context
    try:
        sources_result, wind_trend_obj, air_quality = await asyncio.gather(
            fetch_all_sources(lat, lon),
            fetch_wind_trend(lat, lon),
            fetch_air_quality(lat, lon),
        )
        sources = list(sources_result)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    # 3. Wind trend / AQI are best-effort side context
    wind_trend_str = wind_trend_obj.description if wind_trend_obj else ""

    # 4. Get feedback context from DB (Feature 4)
    feedback_ctx = build_feedback_context(db)

    # 5. Evaluate decision (AI if key present, rule-based fallback)
    ai_kwargs: dict[str, Any] = dict(
        sources=sources,
        location=location_name,
        lat=lat,
        lon=lon,
        feedback_context=feedback_ctx,
        wind_trend=wind_trend_str,
    )
    if air_quality is not None:
        ai_kwargs["air_quality"] = air_quality
    if body.limits is not None:
        ai_kwargs["limits"] = body.limits
    decision_result = await make_ai_decision(**ai_kwargs)

    # 6. Persist report
    applied_limits = body.limits if body.limits is not None else _default_report_limits()
    source_checks = _build_source_checks(sources, air_quality, wind_trend_obj)
    decision_snapshot = asdict(decision_result)
    ai_payload: dict[str, Any] = {
        "applied_limits": applied_limits.model_dump(),
        "decision_snapshot": {
            field.name: decision_snapshot[field.name]
            for field in fields(DecisionResult)
        },
        "wind_trend": wind_trend_str,
        "source_checks": [check.model_dump() for check in source_checks],
        "source_timings": [
            {"source": source.source, **source_timing(source, datetime.now(timezone.utc))}
            for source in sources
        ],
    }
    selected, _ = usable_sources(sources)
    ai_payload["data_quality"] = {
        "available_parameters": 7 - len(decision_result.missing_parameters),
        "required_parameters": 7,
        "usable_sources": len(selected),
        "received_sources": len(sources),
        "warnings": [
            f"{s.source}: {source_timing(s, datetime.now(timezone.utc))['freshness_message']}"
            for s in selected
            if source_timing(s, datetime.now(timezone.utc))["freshness"] == "unknown"
        ] + decision_result.excluded_sources,
    }
    wind = wind_consensus(sources)
    ai_payload["wind_consensus"] = (
        {"wind_speed_knots": wind[0], "wind_direction_deg": wind[1], "source_count": wind[2]}
        if wind else None
    )
    if isinstance(decision_result, AIDecisionResult):
        ai_payload.update(
            {
                "confidence": decision_result.confidence,
                "summary": decision_result.summary,
                "detailed_analysis": decision_result.detailed_analysis,
                "risk_factors": decision_result.risk_factors,
                "recommendations": decision_result.recommendations,
                "parameter_assessments": decision_result.parameter_assessments,
                "parameter_weights": decision_result.parameter_weights,
            }
        )
    if air_quality is not None:
        ai_payload.update(
            {
                "aqi_score": air_quality.aqi_score,
                "pm25": air_quality.pm25,
                "pm10": air_quality.pm10,
            }
        )
    ai_data = json.dumps(ai_payload, ensure_ascii=False) if ai_payload else None

    sources_json = json.dumps(
        [asdict(s) for s in sources]
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
    ai_parameter_weights: dict[str, int] | None = None
    if isinstance(decision_result, AIDecisionResult):
        ai_parameter_weights = decision_result.parameter_weights or None
        if decision_result.summary:
            ai_confidence = decision_result.confidence
            ai_summary = decision_result.summary
            ai_detailed = decision_result.detailed_analysis
            ai_risks = decision_result.risk_factors
            ai_recs = decision_result.recommendations

    return WeatherReportResponse(
        applied_limits=applied_limits.model_dump(),
        source_checks=source_checks,
        report_id=report.id,
        location=report.location,
        lat=lat,
        lon=lon,
        sources=[
            WeatherSourceSchema(
                source=s.source,
                **source_timing(s, datetime.now(timezone.utc)),
                wind_speed_knots=s.wind_speed_knots,
                temperature_c=s.temperature_c,
                humidity_pct=s.humidity_pct,
                visibility_km=s.visibility_km,
                precipitation_level=s.precipitation_level,
                cloud_base_ft=s.cloud_base_ft,
                cloud_ceiling_ft=s.cloud_ceiling_ft,
                cloud_cover_pct=s.cloud_cover_pct,
                wind_direction_deg=s.wind_direction_deg,
                data_kind=s.data_kind,
                station_distance_km=s.station_distance_km,
                ceiling_unlimited=s.ceiling_unlimited,
                clouds_clear=s.clouds_clear,
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
                explanation=p.explanation,
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
        parameter_weights=ai_parameter_weights,
        aqi_score=decision_result.aqi_score,
        pm25=decision_result.pm25,
        pm10=decision_result.pm10,
        missing_parameters=decision_result.missing_parameters,
        known_decision=decision_result.known_decision,
        excluded_sources=decision_result.excluded_sources,
        data_quality=ai_payload["data_quality"],
        wind_consensus=ai_payload["wind_consensus"],
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

    sources = _load_report_sources(report)
    decision_result, payload = _build_report_decision(report)
    stored_limits = payload.get("applied_limits")
    limits = DroneLimit.model_validate(stored_limits) if stored_limits else None

    pdf_bytes = generate_pdf(
        location=report.location,
        lat=report.lat,
        lon=report.lon,
        sources=sources,
        decision_result=decision_result,
        created_at=report.created_at,
        limits=limits,
        source_checks=payload.get("source_checks"),
        parameter_snapshot_available="decision_snapshot" in payload,
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


def _load_report_sources(report: WeatherReport) -> list[WeatherSourceData]:
    raw_sources: list[dict[str, Any]] = json.loads(report.sources_data)
    return [
        WeatherSourceData(
            source=source["source"],
            wind_speed_knots=source["wind_speed_knots"],
            temperature_c=source["temperature_c"],
            humidity_pct=source["humidity_pct"],
            visibility_km=source["visibility_km"],
            precipitation_level=source["precipitation_level"],
            cloud_base_ft=source.get("cloud_base_ft"),
            cloud_ceiling_ft=source.get("cloud_ceiling_ft"),
            cloud_cover_pct=source.get("cloud_cover_pct"),
            wind_direction_deg=source.get("wind_direction_deg"),
            reliability_weight=source.get("reliability_weight", 1.0),
            raw=source.get("raw", {}),
            data_kind=source.get("data_kind", "unknown"),
            ceiling_unlimited=source.get("ceiling_unlimited", False),
            clouds_clear=source.get("clouds_clear", False),
            station_distance_km=source.get("station_distance_km"),
            observation_id=source.get("observation_id"),
        )
        for source in raw_sources
    ]


def _load_ai_payload(report: WeatherReport) -> dict[str, Any]:
    if not report.ai_analysis_data:
        return {}
    try:
        payload = json.loads(report.ai_analysis_data)
    except json.JSONDecodeError as exc:
        logger.error("Invalid saved report metadata for report %s", report.id)
        raise HTTPException(status_code=500, detail="Kayıtlı rapor bilgisi okunamadı.") from exc
    if not isinstance(payload, dict):
        logger.error("Invalid saved report metadata type for report %s", report.id)
        raise HTTPException(status_code=500, detail="Kayıtlı rapor bilgisi geçersiz.")
    return payload


def _default_report_limits() -> DroneLimit:
    values = {
        name: getattr(settings, name)
        for name in DroneLimit.model_fields
        if hasattr(settings, name)
    }
    return DroneLimit(
        **values,
        temp_risky_min_override_c=settings.temp_risky_min_c,
        temp_risky_max_override_c=settings.temp_risky_max_c,
    )


def _load_air_quality(report: WeatherReport) -> AirQualityData | None:
    payload = _load_ai_payload(report)
    aqi_score = payload.get("aqi_score")
    if aqi_score is None:
        return None
    try:
        return AirQualityData(
            aqi_score=int(aqi_score),
            pm25=payload.get("pm25"),
            pm10=payload.get("pm10"),
            raw=payload,
        )
    except (TypeError, ValueError):
        return None


def _build_report_decision(
    report: WeatherReport,
) -> tuple[DecisionResult, dict[str, Any]]:
    sources = _load_report_sources(report)
    ai_payload = _load_ai_payload(report)
    air_quality = _load_air_quality(report)
    stored_limits = ai_payload.get("applied_limits")
    snapshot = ai_payload.get("decision_snapshot")
    if snapshot is not None:
        try:
            values = dict(snapshot)
            values["parameters"] = [
                ParameterResult(**parameter) for parameter in values["parameters"]
            ]
            decision = DecisionResult(**values)
        except (KeyError, TypeError, ValueError) as exc:
            logger.error("Invalid decision snapshot for report %s", report.id)
            raise HTTPException(
                status_code=500, detail="Kayıtlı karar ayrıntıları okunamadı."
            ) from exc
    else:
        decision = make_decision(
            sources, air_quality=air_quality,
            limits=DroneLimit.model_validate(stored_limits) if stored_limits else None,
        )
        if ai_payload.get("confidence") is not None:
            decision.confidence_score = int(ai_payload["confidence"])
    decision.decision = report.decision
    decision.detail = report.decision_detail
    return decision, ai_payload


@router.get(
    "/report/{report_id}",
    response_model=WeatherReportResponse,
    summary="Kayıtlı hava raporunu yeniden analiz etmeden aç",
)
async def get_weather_report(
    report_id: int, db: Session = Depends(get_db),
) -> WeatherReportResponse:
    report = db.get(WeatherReport, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found.")
    decision, payload = _build_report_decision(report)
    timings = {
        timing["source"]: {
            key: value for key, value in timing.items() if key != "source"
        }
        for timing in payload.get("source_timings", [])
    }
    return WeatherReportResponse(
        report_id=report.id,
        location=report.location,
        lat=report.lat,
        lon=report.lon,
        is_saved_report=True,
        parameter_snapshot_available="decision_snapshot" in payload,
        applied_limits=payload.get("applied_limits"),
        source_checks=payload.get("source_checks"),
        sources=[
            WeatherSourceSchema(
                **{key: value for key, value in asdict(source).items() if key != "raw"},
                **timings.get(source.source, {}),
            )
            for source in _load_report_sources(report)
        ],
        decision=decision.decision,
        decision_detail=decision.detail,
        confidence_score=decision.confidence_score,
        parameters=[ParameterResultSchema(**asdict(p)) for p in decision.parameters],
        created_at=report.created_at,
        confidence=payload.get("confidence") if payload.get("summary") else None,
        summary=payload.get("summary") or None,
        detailed_analysis=payload.get("detailed_analysis") or None,
        risk_factors=payload.get("risk_factors") or None,
        recommendations=payload.get("recommendations") or None,
        parameter_weights=payload.get("parameter_weights") or None,
        wind_trend=payload.get("wind_trend") or None,
        aqi_score=decision.aqi_score,
        pm25=decision.pm25,
        pm10=decision.pm10,
        missing_parameters=decision.missing_parameters,
        known_decision=decision.known_decision,
        excluded_sources=decision.excluded_sources,
        data_quality=payload.get("data_quality"),
        wind_consensus=payload.get("wind_consensus"),
    )


def _build_point_source(hour_data: dict[str, Any]) -> WeatherSourceData:
    return WeatherSourceData(
        source="Open-Meteo Forecast",
        wind_speed_knots=hour_data["wind_speed_knots"],
        temperature_c=hour_data["temperature_c"],
        humidity_pct=hour_data["humidity_pct"],
        visibility_km=hour_data["visibility_km"],
        precipitation_level=hour_data["precipitation_level"],
        cloud_base_ft=hour_data.get("cloud_base_ft"),
        cloud_ceiling_ft=hour_data.get("cloud_ceiling_ft"),
        cloud_cover_pct=hour_data.get("cloud_cover_pct"),
        ceiling_unlimited=hour_data.get("ceiling_unlimited", False),
        clouds_clear=hour_data.get("clouds_clear", False),
        data_kind="forecast",
    )


class ReportChatRequest(BaseModel):
    message: str


class ReportChatResponse(BaseModel):
    reply: str


class ForecastHourSchema(BaseModel):
    offset: int
    decision: str
    wind_kt: float | None
    temperature_c: float | None
    humidity_pct: float | None
    visibility_km: float | None
    cloud_base_ft: float | None = None
    precipitation_level: int | None
    time: datetime
    detail: str
    cloud_cover_pct: float | None = None


class ForecastChangeResponse(BaseModel):
    hours: list[ForecastHourSchema]
    missing_offsets: list[int] = Field(default_factory=list)


class NotamResponse(BaseModel):
    notams: list[dict[str, Any]]
    disclaimer: str


class FlightPlanRequest(BaseModel):
    limits: DroneLimit | None = None
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    date_offset: int
    start_hour: int
    end_hour: int

    @model_validator(mode="after")
    def _validate(self) -> "FlightPlanRequest":
        if self.date_offset not in (0, 1):
            raise ValueError("date_offset 0 veya 1 olmalıdır.")
        if not (0 <= self.start_hour <= 23 and 0 <= self.end_hour <= 23):
            raise ValueError("Saatler 0 ile 23 arasında olmalıdır.")
        if self.end_hour < self.start_hour:
            raise ValueError("Bitiş saati başlangıç saatinden küçük olamaz.")
        return self


class FlightPlanHourSchema(BaseModel):
    hour: int
    decision: str
    wind_kt: float | None
    temperature_c: float | None
    humidity_pct: float | None
    visibility_km: float | None
    cloud_base_ft: float | None = None
    precipitation_level: int | None
    time: datetime | None = None
    detail: str = ""
    cloud_cover_pct: float | None = None
    is_past: bool = False


class FlightPlanBestWindowSchema(BaseModel):
    start_hour: int
    end_hour: int
    length: int


class FlightPlanResponse(BaseModel):
    hours: list[FlightPlanHourSchema]
    best_window: FlightPlanBestWindowSchema | None = None
    date: str
    timezone: str = "UTC"
    missing_hours: list[int] = Field(default_factory=list)


class HistoryItemSchema(BaseModel):
    report_id: int
    location: str
    decision: str
    confidence_score: int
    created_at: datetime
    lat: float
    lon: float


class HistoryResponse(BaseModel):
    reports: list[HistoryItemSchema]


class SeasonalRequest(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class SeasonalMonthSchema(BaseModel):
    month: str
    avg_wind_kt: float | None
    avg_visibility_km: float | None = None
    suitability_score: int | None = None
    decision: str
    avg_temperature_c: float | None = None
    avg_precipitation_mm: float | None = None
    sample_count: int = 0


class SeasonalResponse(BaseModel):
    months: list[SeasonalMonthSchema]
    source: str = "Open-Meteo Historical Weather / ERA5"
    period: str = "1991–2020"
    disclaimer: str = "Tarihsel yeniden analiz; ölçüm veya güncel uçuş tahmini değildir. Rüzgâr günlük maksimumların ortalamasıdır. Görüş/tavan olmadığı için uçuş uygunluğu hesaplanmaz."


_MONTHS_TR = [
    "Ocak",
    "Şubat",
    "Mart",
    "Nisan",
    "Mayıs",
    "Haziran",
    "Temmuz",
    "Ağustos",
    "Eylül",
    "Ekim",
    "Kasım",
    "Aralık",
]


def _hourly_timestamp(time_text: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(time_text)
    except (TypeError, ValueError) as exc:
        logger.warning("Invalid hourly forecast timestamp: %r", time_text)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Saatlik tahmin zaman bilgisi geçersiz.",
        ) from exc
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


async def _fetch_open_meteo_hourly(
    lat: float,
    lon: float,
    *,
    timezone_name: str = "UTC",
    forecast_days: int = 3,
) -> dict[str, Any]:
    return await _hourly_cache.get(
        (lat, lon, timezone_name, forecast_days),
        lambda: _load_open_meteo_hourly(lat, lon, timezone_name, forecast_days),
    )


async def _load_open_meteo_hourly(lat: float, lon: float, timezone_name: str, forecast_days: int) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        resp = await _fetch_with_retry(
            client,
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": lat,
                "longitude": lon,
                "hourly": (
                    "temperature_2m,relative_humidity_2m,dew_point_2m,"
                    "wind_speed_10m,precipitation,visibility,cloud_cover"
                ),
                "wind_speed_unit": "kmh",
                "timezone": timezone_name,
                "forecast_days": forecast_days,
            },
        )
        return resp.json()


def _hourly_row_from_payload(
    payload: dict[str, Any], index: int
) -> dict[str, Any] | None:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if len(times) <= index:
        return None

    def _value(key: str, default: float | None = None) -> float | None:
        values = hourly.get(key, [])
        if len(values) <= index or values[index] is None:
            return default
        value = float(values[index])
        return value if math.isfinite(value) else default

    temp_c = _value("temperature_2m")
    if all(_value(key) is None for key in (
        "temperature_2m", "wind_speed_10m", "relative_humidity_2m",
        "visibility", "precipitation", "cloud_cover", "dew_point_2m",
    )):
        return None

    dew_c = _value("dew_point_2m")
    cloud_cover = _value("cloud_cover")
    base_ft: float | None = None
    if dew_c is not None and temp_c is not None:
        spread = max(temp_c - dew_c, 0.0)
        base_ft = spread * 122.5 * 3.28084

    return {
        "time": times[index],
        "wind_speed_knots": _value("wind_speed_10m") * 0.539957 if _value("wind_speed_10m") is not None else None,
        "temperature_c": temp_c,
        "humidity_pct": _value("relative_humidity_2m"),
        "visibility_km": _value("visibility") / 1000.0 if _value("visibility") is not None else None,
        "cloud_base_ft": base_ft,
        "cloud_cover_pct": _value("cloud_cover"),
        "cloud_ceiling_ft": (
            base_ft + 200.0 if base_ft is not None and cloud_cover is not None and cloud_cover > 50 else None
        ),
        "precipitation_level": _precip_level(_value("precipitation")) if _value("precipitation") is not None else None,
        "ceiling_unlimited": cloud_cover is not None and cloud_cover <= 50,
        "clouds_clear": cloud_cover == 0,
    }


def _best_window(
    hours: list[FlightPlanHourSchema],
) -> FlightPlanBestWindowSchema | None:
    best: FlightPlanBestWindowSchema | None = None
    current: FlightPlanBestWindowSchema | None = None

    for hour in sorted(hours, key=lambda item: item.hour):
        if hour.decision != "UYGUN" or hour.is_past:
            current = None
            continue
        if current is None or hour.hour != current.end_hour + 1:
            current = FlightPlanBestWindowSchema(
                start_hour=hour.hour, end_hour=hour.hour, length=1
            )
        else:
            current = FlightPlanBestWindowSchema(
                start_hour=current.start_hour,
                end_hour=hour.hour,
                length=current.length + 1,
            )
        if best is None or current.length > best.length:
            best = current
    return best


@router.post(
    "/report/{report_id}/chat",
    response_model=ReportChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Bir hava raporu hakkında yapay zeka sohbeti yap",
)
async def report_chat(
    report_id: int,
    body: ReportChatRequest,
    db: Session = Depends(get_db),
) -> ReportChatResponse:
    report: WeatherReport | None = db.get(WeatherReport, report_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report {report_id} not found.",
        )
    if not body.message.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Mesaj boş olamaz.",
        )

    sources = _load_report_sources(report)
    decision_result, ai_payload = _build_report_decision(report)
    reply = await chat_about_report(
        report_id=report_id,
        location=report.location,
        report_created_at=report.created_at,
        sources=sources,
        decision_result=decision_result,
        message=body.message.strip(),
        ai_summary=str(ai_payload.get("summary") or ""),
    )
    return ReportChatResponse(reply=reply)


@router.post(
    "/report/{report_id}/forecast-change",
    response_model=ForecastChangeResponse,
    status_code=status.HTTP_200_OK,
    summary="Sonraki 6 saatlik karar değişimini tahmin et",
)
async def forecast_change(
    report_id: int,
    db: Session = Depends(get_db),
) -> ForecastChangeResponse:
    report: WeatherReport | None = db.get(WeatherReport, report_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report {report_id} not found.",
        )

    try:
        payload = await _fetch_open_meteo_hourly(report.lat, report.lon)
    except Exception as exc:
        logger.warning("Hourly forecast fetch failed for report %s: %s", report_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Saatlik tahmin verisi alınamadı.",
        ) from exc
    start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    targets = {start + timedelta(hours=offset): offset for offset in range(1, 7)}
    results = {}
    for index, time_text in enumerate(payload.get("hourly", {}).get("time", [])):
        timestamp = _hourly_timestamp(time_text)
        if timestamp in targets:
            results[targets[timestamp]] = (timestamp, _hourly_row_from_payload(payload, index))

    hours: list[ForecastHourSchema] = []
    air_quality = _load_air_quality(report)
    saved_limits = _load_ai_payload(report).get("applied_limits")
    limits = DroneLimit.model_validate(saved_limits) if saved_limits else None
    for offset in range(1, 7):
        timestamp, result = results.get(offset, (None, None))
        if result is None:
            continue
        decision = make_decision(
            [_build_point_source(result)], air_quality=air_quality, limits=limits
        )
        hours.append(
            ForecastHourSchema(
                offset=offset,
                decision=decision.decision,
                wind_kt=_round_optional(result["wind_speed_knots"]),
                temperature_c=_round_optional(result["temperature_c"]),
                humidity_pct=_round_optional(result["humidity_pct"]),
                visibility_km=_round_optional(result["visibility_km"]),
                cloud_base_ft=result.get("cloud_base_ft"),
                precipitation_level=result["precipitation_level"],
                time=timestamp,
                detail=decision.detail,
                cloud_cover_pct=decision.avg_cloud_cover_pct,
            )
        )

    if not hours:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tahmin verisi alınamadı.",
        )
    available = {hour.offset for hour in hours}
    return ForecastChangeResponse(
        hours=hours, missing_offsets=[i for i in range(1, 7) if i not in available]
    )


@router.get(
    "/notam",
    response_model=NotamResponse,
    status_code=status.HTTP_200_OK,
    summary="NOTAM yer tutucu yanıtı",
)
async def get_notam_placeholder(lat: float, lon: float) -> NotamResponse:
    return NotamResponse(
        notams=[],
        disclaimer=(
            "NOTAM verisi yakında eklenecek. Resmi NOTAM kontrolü için "
            "ops.faa.gov veya HHMB'yi ziyaret edin."
        ),
    )


@router.post(
    "/flight-plan",
    response_model=FlightPlanResponse,
    status_code=status.HTTP_200_OK,
    summary="Saatlik uçuş planı penceresi oluştur",
)
async def flight_plan(body: FlightPlanRequest) -> FlightPlanResponse:
    try:
        payload = await _fetch_open_meteo_hourly(body.lat, body.lon)
    except Exception as exc:
        logger.warning("Flight plan fetch failed at (%s, %s): %s", body.lat, body.lon, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Uçuş planı verisi alınamadı.",
        ) from exc

    # Open-Meteo returns naive ISO strings (e.g. "2026-07-25T14:00") when
    # timezone=UTC, so fromisoformat yields a naive datetime.  Compare dates
    # using UTC today to stay consistent.
    now = datetime.now(timezone.utc)
    base_date = now.date() + timedelta(days=body.date_offset)

    def _build_hours(
        time_texts: list[str],
    ) -> list[FlightPlanHourSchema]:
        result: list[FlightPlanHourSchema] = []
        seen_hours: set[int] = set()
        for idx, time_text in enumerate(time_texts):
            timestamp = _hourly_timestamp(time_text)
            if timestamp.date() != base_date:
                continue
            if not (body.start_hour <= timestamp.hour <= body.end_hour):
                continue
            if timestamp.hour in seen_hours:
                continue
            row = _hourly_row_from_payload(payload, idx)
            if row is None:
                continue
            decision = make_decision([_build_point_source(row)], limits=body.limits)
            result.append(
                FlightPlanHourSchema(
                    hour=timestamp.hour,
                    decision=decision.decision,
                    wind_kt=_round_optional(row["wind_speed_knots"]),
                    temperature_c=_round_optional(row["temperature_c"]),
                    humidity_pct=_round_optional(row["humidity_pct"]),
                    visibility_km=_round_optional(row["visibility_km"]),
                    cloud_base_ft=row.get("cloud_base_ft"),
                    precipitation_level=row["precipitation_level"],
                    time=timestamp,
                    detail=decision.detail,
                    cloud_cover_pct=decision.avg_cloud_cover_pct,
                    is_past=timestamp + timedelta(hours=1) <= now,
                )
            )
            seen_hours.add(timestamp.hour)
        return result

    time_texts = payload.get("hourly", {}).get("time", [])
    hours = _build_hours(time_texts)

    if not hours:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Seçilen saat aralığı için veri bulunamadı.",
        )

    hours.sort(key=lambda item: item.hour)
    available = {hour.hour for hour in hours}
    return FlightPlanResponse(
        hours=hours, best_window=_best_window(hours), date=base_date.isoformat(),
        missing_hours=[
            i for i in range(body.start_hour, body.end_hour + 1) if i not in available
        ],
    )


@router.get(
    "/history",
    response_model=HistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Son hava raporlarını listele",
)
async def weather_history(
    limit: int = 30,
    location: str | None = Query(default=None, max_length=200),
    start_date: date | None = None,
    end_date: date | None = None,
    db: Session = Depends(get_db),
) -> HistoryResponse:
    if start_date and end_date and end_date < start_date:
        raise HTTPException(status_code=422, detail="Son tarih ilk tarihten önce olamaz.")
    limit = max(1, min(limit, 100))
    query = db.query(WeatherReport)
    if location and location.strip():
        query = query.filter(WeatherReport.location.icontains(location.strip(), autoescape=True))
    if start_date:
        query = query.filter(WeatherReport.created_at >= datetime.combine(start_date, datetime.min.time()))
    if end_date:
        query = query.filter(WeatherReport.created_at <= datetime.combine(end_date, datetime.max.time()))
    reports = query.order_by(WeatherReport.created_at.desc(), WeatherReport.id.desc()).limit(limit).all()

    items: list[HistoryItemSchema] = []
    for report in reports:
        decision_result, ai_payload = _build_report_decision(report)
        confidence = ai_payload.get("confidence")
        if confidence is None:
            confidence = decision_result.confidence_score
        items.append(
            HistoryItemSchema(
                report_id=report.id,
                location=report.location,
                decision=report.decision,
                confidence_score=int(confidence),
                created_at=report.created_at,
                lat=report.lat,
                lon=report.lon,
            )
        )
    return HistoryResponse(reports=items)


@router.post(
    "/seasonal",
    response_model=SeasonalResponse,
    status_code=status.HTTP_200_OK,
    summary="Mevsimsel uçuş uygunluğu analizi oluştur",
)
async def seasonal_analysis(body: SeasonalRequest) -> SeasonalResponse:
    return await _seasonal_cache.get(
        (body.lat, body.lon), lambda: _load_seasonal_analysis(body),
    )


async def _load_seasonal_analysis(body: SeasonalRequest) -> SeasonalResponse:
    """Show historical monthly statistics without inventing flight suitability."""
    monthly_wind_kmh: list[float | None] | None = None
    monthly_precip_mm: list[float | None] | None = None
    monthly_temp_c: list[float | None] | None = None

    try:
        async with httpx.AsyncClient() as client:
            resp = await _fetch_with_retry(
                client,
                "https://archive-api.open-meteo.com/v1/archive",
                {
                    "latitude": body.lat,
                    "longitude": body.lon,
                    "daily": "wind_speed_10m_max,precipitation_sum,temperature_2m_mean",
                    "start_date": "1991-01-01",
                    "end_date": "2020-12-31",
                    "models": "era5",
                    "timezone": "UTC",
                },
            )
            payload = resp.json()

        daily = payload.get("daily", {})
        times = daily.get("time", [])
        winds = daily.get("wind_speed_10m_max", [])
        precips = daily.get("precipitation_sum", [])
        temps = daily.get("temperature_2m_mean", [])

        if times and winds:
            wind_sums = [0.0] * 12
            precip_sums = [0.0] * 12
            temp_sums = [0.0] * 12
            counts = [0] * 12

            for i, t in enumerate(times):
                # "1991-01-01" → month index 0
                m = int(t[5:7]) - 1
                if (
                    len(winds) > i and len(precips) > i and len(temps) > i
                    and all(isinstance(v, (int, float)) and math.isfinite(v) for v in (winds[i], precips[i], temps[i]))
                ):
                    wind_sums[m] += float(winds[i])
                    counts[m] += 1
                    precip_sums[m] += float(precips[i])
                    temp_sums[m] += float(temps[i])

            if any(c > 0 for c in counts):
                monthly_wind_kmh = [
                    wind_sums[i] / counts[i] if counts[i] > 0 else None
                    for i in range(12)
                ]
                monthly_precip_mm = [
                    precip_sums[i] / counts[i] if counts[i] > 0 else None
                    for i in range(12)
                ]
                monthly_temp_c = [
                    temp_sums[i] / counts[i] if counts[i] > 0 else None
                    for i in range(12)
                ]
    except (httpx.HTTPError, ValueError, TypeError, IndexError) as exc:
        logger.warning("Historical seasonal data unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Tarihsel iklim verisi alınamadı; varsayımsal tablo gösterilmiyor.") from exc

    if monthly_wind_kmh is None:
        logger.warning("Historical seasonal payload contains no complete days")
        raise HTTPException(status_code=503, detail="Tarihsel iklim verisi eksik; uygunluk hesaplanmadı.")
    assert monthly_precip_mm is not None and monthly_temp_c is not None

    months: list[SeasonalMonthSchema] = []
    for index, month_name in enumerate(_MONTHS_TR):
        wind_kmh = monthly_wind_kmh[index]
        months.append(
            SeasonalMonthSchema(
                month=month_name,
                avg_wind_kt=_round_optional(wind_kmh * 0.539957) if wind_kmh is not None else None,
                avg_temperature_c=_round_optional(monthly_temp_c[index]),
                avg_precipitation_mm=_round_optional(monthly_precip_mm[index]),
                sample_count=counts[index],
                decision="VERI_YETERSIZ",
            )
        )
    return SeasonalResponse(months=months)


# ---------------------------------------------------------------------------
# Grid analysis schemas
# ---------------------------------------------------------------------------


class GridAnalysisRequest(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    grid_size: int = 25  # 25 (5x5) or 100 (10x10)
    altitude_ft: float = 1000.0
    hours: int = 1  # 1 = instant, 24 = next 24 h
    location_name: str = ""
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
    wind_speed_knots: float | None
    temperature_c: float | None
    humidity_pct: float | None
    visibility_km: float | None
    cloud_base_ft: float | None
    cloud_ceiling_ft: float | None
    precipitation_level: int | None
    cloud_cover_pct: float | None = None
    detail: str = ""
    parameters: list[ParameterResultSchema] = Field(default_factory=list)
    time: datetime | None = None
    wind_method: str = ""


class GridSummary(BaseModel):
    UYGUN: int
    RISKLI: int
    UYGUN_DEGIL: int
    total: int
    VERI_YETERSIZ: int = 0


class GridAnalysisResponse(BaseModel):
    points: list[GridPointWeather]
    summary: GridSummary
    grid_size: int
    altitude_ft: float
    ai_regional_summary: str | None = None


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
    requested UTC target hour is returned.

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
        resp = await _fetch_with_retry(client, url, params)
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
        valid_time = _hourly_timestamp(cur["time"]) if cur.get("time") else None

        dew_c: float | None = cur.get("dew_point_2m")
        wind_kmh = cur.get("wind_speed_10m")
        rh = cur.get("relative_humidity_2m")
        precip_mm = cur.get("precipitation")
        vis_m = cur.get("visibility")
        cloud_pct = cur.get("cloud_cover")

        wind_knots = wind_kmh * 0.539957 if wind_kmh is not None else None

        if pressure_hpa is not None:
            hourly = data.get("hourly", {})
            pl_wind_list = hourly.get(f"wind_speed_{pressure_hpa}hPa", [])
            target = valid_time.replace(minute=0, second=0, microsecond=0) if valid_time else None
            idx = next((i for i, text in enumerate(hourly.get("time", [])) if _hourly_timestamp(text) == target), None)
            wind_knots = float(pl_wind_list[idx]) * 0.539957 if idx is not None and len(pl_wind_list) > idx and pl_wind_list[idx] is not None else None
        else:
            # Power-law wind profile: V(z) = V_ref * (z/z_ref)^(1/7)
            # Exponent 1/7 (~0.143) is the standard value for open terrain
            # (Hellmann exponent); z_ref = 10 m (Open-Meteo measurement height).
            alt_m = altitude_ft * 0.3048
            ref_m = 10.0
            if alt_m > ref_m and wind_knots is not None:
                wind_knots *= (alt_m / ref_m) ** (1.0 / 7.0)
    else:
        hourly = data.get("hourly", {})
        temp_list: list[Any] = hourly.get("temperature_2m", [])
        target = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(hours=hours)
        idx = next((i for i, text in enumerate(hourly.get("time", [])) if _hourly_timestamp(text) == target), None)
        if idx is None:
            logger.warning("Grid target hour missing: %s", target.isoformat())
            return None
        valid_time = target

        temp_c = temp_list[idx] if len(temp_list) > idx else None

        def _h(key: str) -> float | None:
            lst = hourly.get(key, [])
            val = lst[idx] if len(lst) > idx else None
            return float(val) if val is not None else None

        dew_c_val = hourly.get("dew_point_2m", [])
        dew_c = (
            float(dew_c_val[idx])
            if len(dew_c_val) > idx and dew_c_val[idx] is not None
            else None
        )
        wind_kmh = _h("wind_speed_10m")
        rh = _h("relative_humidity_2m")
        precip_mm = _h("precipitation")
        vis_m = _h("visibility")
        cloud_pct = _h("cloud_cover")

        wind_knots = wind_kmh * 0.539957 if wind_kmh is not None else None

        if pressure_hpa is not None:
            pl_wind_list = hourly.get(f"wind_speed_{pressure_hpa}hPa", [])
            if len(pl_wind_list) > idx and pl_wind_list[idx] is not None:
                wind_knots = float(pl_wind_list[idx]) * 0.539957
            else:
                wind_knots = None
        else:
            alt_m = altitude_ft * 0.3048
            ref_m = 10.0
            if alt_m > ref_m and wind_knots is not None:
                wind_knots *= (alt_m / ref_m) ** (1.0 / 7.0)

    base_ft: float | None = None
    if dew_c is not None and temp_c is not None:
        spread = max(temp_c - dew_c, 0.0)
        # Standard approximation: ~122.5 m per degC of T/Td spread
        base_ft = spread * 122.5 * 3.28084

    ceiling_ft: float | None = None
    if base_ft is not None and cloud_pct is not None and cloud_pct > 50:
        # Heuristic: when cloud cover > 50 % the ceiling is just above the
        # estimated base (200 ft offset mirrors the existing weather_fetcher).
        ceiling_ft = base_ft + 200.0

    if precip_mm is None:
        precip_level = None
    elif precip_mm <= 0:
        precip_level = 0
    elif precip_mm < 2.5:
        precip_level = 1
    else:
        precip_level = 2

    return {
        "wind_speed_knots": _round_optional(wind_knots, 2),
        "temperature_c": temp_c,
        "humidity_pct": rh,
        "visibility_km": vis_m / 1000.0 if vis_m is not None else None,
        "cloud_base_ft": base_ft,
        "cloud_ceiling_ft": ceiling_ft,
        "precipitation_level": precip_level,
        "cloud_cover_pct": cloud_pct,
        "ceiling_unlimited": cloud_pct is not None and cloud_pct <= 50,
        "clouds_clear": cloud_pct == 0,
        "time": valid_time,
        "wind_method": (
            f"{pressure_hpa} hPa model seviyesi; yükseklik yaklaşık, sıcaklık/nem yüzey değeridir"
            if pressure_hpa else "10 m model rüzgârından açık arazi güç kanunuyla yaklaşık irtifa hesabı"
            if altitude_ft * 0.3048 > 10 else "10 m model rüzgârı"
        ),
    }


def _eval_grid_decision(
    wind_speed_knots: float | None,
    visibility_km: float | None,
    temperature_c: float | None,
    humidity_pct: float | None,
    cloud_base_ft: float | None,
    cloud_ceiling_ft: float | None,
    precipitation_level: int | None,
    limits: DroneLimit,
    cloud_cover_pct: float | None = None,
    ceiling_unlimited: bool = False,
) -> str:
    """Use the same rules and missing-data policy as point/hourly analysis."""
    return make_decision([WeatherSourceData(
        source="Open-Meteo Grid", wind_speed_knots=wind_speed_knots,
        temperature_c=temperature_c, humidity_pct=humidity_pct,
        visibility_km=visibility_km, precipitation_level=precipitation_level,
        cloud_base_ft=cloud_base_ft, cloud_ceiling_ft=cloud_ceiling_ft,
        cloud_cover_pct=cloud_cover_pct, ceiling_unlimited=ceiling_unlimited,
        data_kind="forecast",
    )], limits=limits).decision


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

    raw_results: list[Any] = []
    async with httpx.AsyncClient() as client:
        for batch_start in range(0, len(grid_coords), _GRID_BATCH_SIZE):
            batch = grid_coords[batch_start : batch_start + _GRID_BATCH_SIZE]
            batch_tasks = [
                _fetch_grid_point_weather(
                    lat, lon, body.altitude_ft, body.hours, client
                )
                for lat, lon in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            raw_results.extend(batch_results)
            if batch_start + _GRID_BATCH_SIZE < len(grid_coords):
                await asyncio.sleep(_GRID_BATCH_DELAY)

    points: list[GridPointWeather] = []
    for (lat, lon), result in zip(grid_coords, raw_results):
        if isinstance(result, Exception):
            logger.warning("Grid point (%.4f, %.4f) raised: %s", lat, lon, result)
            result = None
        if result is None:
            points.append(GridPointWeather(
                lat=lat, lon=lon, decision="VERI_YETERSIZ",
                wind_speed_knots=None, temperature_c=None, humidity_pct=None,
                visibility_km=None, precipitation_level=None,
                cloud_base_ft=None, cloud_ceiling_ft=None,
                detail="Bu noktanın hava verisi alınamadı.",
            ))
            continue

        decision = make_decision([_build_point_source(result)], limits=body.limits)

        points.append(
            GridPointWeather(
                lat=lat,
                lon=lon,
                decision=decision.decision,
                wind_speed_knots=result["wind_speed_knots"],
                temperature_c=result["temperature_c"],
                humidity_pct=result["humidity_pct"],
                visibility_km=result["visibility_km"],
                cloud_base_ft=result["cloud_base_ft"],
                cloud_ceiling_ft=result["cloud_ceiling_ft"],
                precipitation_level=result["precipitation_level"],
                cloud_cover_pct=result.get("cloud_cover_pct", 0.0),
                detail=decision.detail,
                parameters=[ParameterResultSchema(**asdict(p)) for p in decision.parameters],
                time=result.get("time"),
                wind_method=result.get("wind_method", ""),
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
    summary = GridSummary(
        UYGUN=n_uygun,
        RISKLI=n_riskli,
        UYGUN_DEGIL=n_uygun_degil,
        total=len(points),
        VERI_YETERSIZ=sum(p.decision == "VERI_YETERSIZ" for p in points),
    )
    ai_regional_summary = await _build_grid_ai_summary(
        GridSummaryRequest(
            points=points,
            summary=summary,
            altitude_ft=body.altitude_ft,
            location_hint=body.location_name,
            lat_min=body.lat_min,
            lat_max=body.lat_max,
            lon_min=body.lon_min,
            lon_max=body.lon_max,
        ),
        os.environ.get("OPENAI_API_KEY", ""),
    )

    return GridAnalysisResponse(
        points=points,
        summary=summary,
        grid_size=body.grid_size,
        altitude_ft=body.altitude_ft,
        ai_regional_summary=ai_regional_summary,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/weather/grid/summary
# ---------------------------------------------------------------------------


class GridSummaryRequest(BaseModel):
    points: list[GridPointWeather]
    summary: GridSummary
    altitude_ft: float = 1000.0
    location_hint: str = ""
    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None


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
    summary_text = await _build_grid_summary_text(body)
    return GridSummaryResponse(ai_summary=summary_text)


@dataclass(frozen=True)
class GridSummaryMetrics:
    avg_wind: float
    avg_temp: float
    avg_humidity: float
    avg_vis: float
    avg_cloud_base: float | None
    pct_uygun: int
    pct_riskli: int
    pct_uygun_degil: int
    location_name: str
    bounds: str


def _calculate_mean_excluding_none(values: list[float | None]) -> float | None:
    """Average trusted numeric values, ignoring missing entries."""
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _format_grid_bounds(body: GridSummaryRequest) -> str:
    if None in (body.lat_min, body.lat_max, body.lon_min, body.lon_max):
        return "Belirtilmedi"
    return (
        f"{body.lat_min:.4f}-{body.lat_max:.4f}, "
        f"{body.lon_min:.4f}-{body.lon_max:.4f}"
    )


def _build_grid_summary_metrics(body: GridSummaryRequest) -> GridSummaryMetrics | None:
    points = body.points
    summary = body.summary

    if not points or summary.VERI_YETERSIZ or any(
        value is None for p in points
        for value in (p.wind_speed_knots, p.temperature_c, p.humidity_pct, p.visibility_km)
    ):
        return None

    avg_wind = sum(p.wind_speed_knots for p in points) / len(points)
    avg_temp = sum(p.temperature_c for p in points) / len(points)
    avg_humidity = sum(p.humidity_pct for p in points) / len(points)
    avg_vis = sum(p.visibility_km for p in points) / len(points)
    avg_cloud_base = _calculate_mean_excluding_none([p.cloud_base_ft for p in points])
    pct_uygun = round(summary.UYGUN / summary.total * 100) if summary.total else 0
    pct_riskli = round(summary.RISKLI / summary.total * 100) if summary.total else 0
    pct_uygun_degil = (
        round(summary.UYGUN_DEGIL / summary.total * 100) if summary.total else 0
    )

    return GridSummaryMetrics(
        avg_wind=avg_wind,
        avg_temp=avg_temp,
        avg_humidity=avg_humidity,
        avg_vis=avg_vis,
        avg_cloud_base=avg_cloud_base,
        pct_uygun=pct_uygun,
        pct_riskli=pct_riskli,
        pct_uygun_degil=pct_uygun_degil,
        location_name=body.location_hint.strip() or "Belirtilmedi",
        bounds=_format_grid_bounds(body),
    )


async def _build_grid_ai_summary(body: GridSummaryRequest, api_key: str) -> str | None:
    """Return a GPT-generated Turkish grid summary when available."""
    metrics = _build_grid_summary_metrics(body)
    if metrics is None:
        return None

    if not api_key or _AsyncOpenAI is None:
        return None

    try:
        client = _AsyncOpenAI(api_key=api_key)
        avg_cloud_base = metrics.avg_cloud_base
        avg_cloud_base_text = (
            f"{avg_cloud_base:.0f} ft" if avg_cloud_base is not None else "Bilinmiyor"
        )
        prompt = (
            "Sen bir drone uçuş güvenlik uzmanısın. Aşağıdaki bölgesel hava "
            "analizi verilerini değerlendirerek kısa ve anlamlı bir Türkçe "
            "yorum yaz.\n\n"
            f"BÖLGE: {metrics.location_name} ({metrics.bounds})\n"
            f"GRID BOYUTU: {body.summary.total} nokta\n"
            f"İRTİFA: {body.altitude_ft:.0f} ft\n\n"
            "KARAR DAĞILIMI:\n"
            f"- UYGUN: {body.summary.UYGUN} nokta (%{metrics.pct_uygun})\n"
            f"- RİSKLİ: {body.summary.RISKLI} nokta (%{metrics.pct_riskli})\n"
            "- UYGUN DEĞİL: "
            f"{body.summary.UYGUN_DEGIL} nokta (%{metrics.pct_uygun_degil})\n\n"
            "ORT. PARAMETRELER:\n"
            f"- Rüzgar: {metrics.avg_wind:.1f} kt\n"
            f"- Nem: {metrics.avg_humidity:.0f}%\n"
            f"- Sıcaklık: {metrics.avg_temp:.1f} °C\n"
            f"- Görüş: {metrics.avg_vis:.1f} km\n"
            f"- Bulut tabanı: {avg_cloud_base_text}\n\n"
            "2-3 cümle ile bölgenin genel uçuş koşullarını değerlendir. "
            "Varsa dikkat edilmesi gereken başlıca risk faktörlerini belirt. "
            "Yanıtın doğrudan değerlendirme metni olsun, JSON değil."
        )
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        content = (response.choices[0].message.content or "").strip()
        return content if content else None

    except Exception as exc:
        logger.warning("GPT grid summary failed: %s", exc)
        return None


async def _build_grid_summary_text(body: GridSummaryRequest) -> str:
    """Return AI summary when available, otherwise the rule-based fallback."""
    metrics = _build_grid_summary_metrics(body)
    if metrics is None:
        return _GRID_NO_DATA_SUMMARY

    ai_summary = await _build_grid_ai_summary(
        body, os.environ.get("OPENAI_API_KEY", "")
    )
    if ai_summary:
        return ai_summary

    return _rule_based_grid_summary(
        body.summary,
        metrics.avg_wind,
        metrics.avg_temp,
        metrics.avg_humidity,
        metrics.avg_vis,
        metrics.pct_uygun,
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


# ---------------------------------------------------------------------------
# Runway wind component schemas and endpoints
# ---------------------------------------------------------------------------


class RunwaySchema(BaseModel):
    airport_icao: str
    airport_name: str
    runway_ident: str
    heading_true: float
    distance_km: float


class WindComponentRequest(BaseModel):
    wind_direction_deg: float
    wind_speed_knots: float
    runway_heading_deg: float
    crosswind_ok_max: float = 15.0
    crosswind_risky_max: float = 20.0
    tailwind_ok_max: float = 5.0
    tailwind_risky_max: float = 10.0
    headwind_ok_max: float = 25.0
    headwind_risky_max: float = 35.0


class WindComponentResponse(BaseModel):
    headwind_kt: float
    tailwind_kt: float
    crosswind_kt: float
    crosswind_side: str
    headwind_decision: str
    crosswind_decision: str
    tailwind_decision: str
    overall_component_decision: str


def _eval_component(value: float, ok_max: float, risky_max: float) -> str:
    """Evaluate a wind component value against ok/risky thresholds."""
    if value <= ok_max:
        return "UYGUN"
    if value <= risky_max:
        return "RISKLI"
    return "UYGUN_DEGIL"


@router.get(
    "/runways",
    response_model=list[RunwaySchema],
    summary="Fetch nearby airport runways",
)
async def get_nearby_runways(
    lat: float,
    lon: float,
) -> list[RunwaySchema]:
    """Return a list of nearby airport runways fetched from the AVWX API.

    If the AVWX key is not configured or the request fails, returns an empty
    list rather than raising an error.
    """
    runways = await fetch_nearby_runways(lat, lon)
    return [RunwaySchema(**r) for r in runways]


@router.post(
    "/wind-components",
    response_model=WindComponentResponse,
    summary="Calculate runway wind components",
)
async def calculate_wind_components(
    body: WindComponentRequest,
) -> WindComponentResponse:
    """Calculate headwind, crosswind, and tailwind components for a runway.

    Evaluates each component against the supplied limits and returns a
    per-component decision (UYGUN / RISKLI / UYGUN_DEGIL) plus an overall
    decision based on the worst component.
    """
    components = compute_wind_components(
        wind_direction_deg=body.wind_direction_deg,
        wind_speed_knots=body.wind_speed_knots,
        runway_heading_deg=body.runway_heading_deg,
    )

    headwind_dec = _eval_component(
        components["headwind_kt"], body.headwind_ok_max, body.headwind_risky_max
    )
    crosswind_dec = _eval_component(
        components["crosswind_kt"], body.crosswind_ok_max, body.crosswind_risky_max
    )
    tailwind_dec = _eval_component(
        components["tailwind_kt"], body.tailwind_ok_max, body.tailwind_risky_max
    )

    _score = {"UYGUN": 0, "RISKLI": 1, "UYGUN_DEGIL": 2}
    _label = {0: "UYGUN", 1: "RISKLI", 2: "UYGUN_DEGIL"}
    worst = max(_score[headwind_dec], _score[crosswind_dec], _score[tailwind_dec])

    return WindComponentResponse(
        headwind_kt=components["headwind_kt"],
        tailwind_kt=components["tailwind_kt"],
        crosswind_kt=components["crosswind_kt"],
        crosswind_side=components["crosswind_side"],
        headwind_decision=headwind_dec,
        crosswind_decision=crosswind_dec,
        tailwind_decision=tailwind_dec,
        overall_component_decision=_label[worst],
    )


# ---------------------------------------------------------------------------
# GET /api/v1/weather/wind
# ---------------------------------------------------------------------------

_SOURCE_NOTES: dict[str, str] = {
    "METAR (AVWX)": "Gerçek havalimanı ölçümü",
    "METAR (CheckWX)": "Gerçek havalimanı ölçümü",
    "METAR (NOAA/aviationweather.gov)": "Havalimanı gözlemi; hamle varsa hızda kullanılır",
    "Windy": "GFS model tahmini",
    "OpenWeatherMap": "OpenWeatherMap güncel hava ürünü",
    "WeatherAPI": "WeatherAPI anlık verisi",
    "Open-Meteo": "Açık kaynak hava modeli",
    "MGM": "Meteoroloji Genel Müdürlüğü",
}


def _source_note(source_name: str) -> str:
    for key, note in _SOURCE_NOTES.items():
        if key.lower() in source_name.lower():
            return note
    return "Hava verisi kaynağı"


def _reliability_label(weight: float) -> str:
    if weight >= 1.7:
        return "high"
    if weight >= 1.0:
        return "medium"
    return "low"


class WindSourceItem(BaseModel):
    source: str
    wind_speed_knots: float | None
    wind_direction_deg: float | None
    reliability: str
    reliability_weight: float
    note: str


class WindConsensus(BaseModel):
    wind_speed_knots: float
    wind_direction_deg: float
    source_count: int


class WindResponse(BaseModel):
    consensus: WindConsensus
    sources: list[WindSourceItem]


@router.get(
    "/wind",
    response_model=WindResponse,
    summary="Live wind data from all sources",
)
async def get_wind(lat: float, lon: float) -> WindResponse:
    """Return wind readings from every configured source, sorted by reliability.

    Sources without a wind direction are excluded from the consensus calculation
    but still included in the response (with ``wind_direction_deg`` set to
    ``null``).  The consensus is a reliability-weight-averaged value computed
    only from sources that have a valid ``wind_direction_deg``.
    """
    try:
        sources = await fetch_all_sources(lat, lon)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    sources, excluded = usable_sources(sources)
    items = sorted(sources, key=lambda s: s.reliability_weight, reverse=True)

    # Weighted average consensus from sources that have wind_direction_deg
    consensus = wind_consensus(items)
    if consensus is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Kullanılabilir rüzgâr hız/yön çifti yok veya kaynak yönleri çelişiyor; ortak rüzgâr hesaplanamaz.",
        )

    avg_speed, avg_dir, count = consensus

    return WindResponse(
        consensus=WindConsensus(
            wind_speed_knots=round(avg_speed, 1),
            wind_direction_deg=round(avg_dir, 1),
            source_count=count,
        ),
        sources=[
            WindSourceItem(
                source=s.source,
                wind_speed_knots=_round_optional(s.wind_speed_knots),
                wind_direction_deg=(
                    round(s.wind_direction_deg, 1)
                    if s.wind_direction_deg is not None
                    else None
                ),
                reliability=_reliability_label(s.reliability_weight),
                reliability_weight=s.reliability_weight,
                note=_source_note(s.source),
            )
            for s in items
        ],
    )

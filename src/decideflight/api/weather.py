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
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
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
    chat_about_report,
    make_ai_decision,
)
from decideflight.services.decision_engine import DecisionResult, make_decision
from decideflight.services.decision_engine import compute_wind_components
from decideflight.services.geocoding import geocode_city
from decideflight.services.report_generator import generate_pdf
from decideflight.services.trend_analyzer import fetch_wind_trend
from decideflight.services.weather_fetcher import (
    AirQualityData,
    WeatherSourceData,
    _fetch_with_retry,
    fetch_air_quality,
    fetch_all_sources,
    fetch_nearby_runways,
)

try:
    from openai import AsyncOpenAI as _AsyncOpenAI  # type: ignore[import]
except ImportError:  # pragma: no cover
    _AsyncOpenAI = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/weather", tags=["weather"])

# Grid analysis: batch size and inter-batch delay to avoid rate limits
_GRID_BATCH_SIZE = 5
_GRID_BATCH_DELAY = 0.5

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
    wind_direction_deg: float | None = None


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
    parameter_weights: dict[str, int] | None = None
    aqi_score: int | None = None
    pm25: float | None = None
    pm10: float | None = None


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
    decision_result = await make_ai_decision(**ai_kwargs)

    # 6. Persist report
    ai_payload: dict[str, Any] = {}
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
                wind_direction_deg=s.wind_direction_deg,
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
        parameter_weights=ai_parameter_weights,
        aqi_score=decision_result.aqi_score,
        pm25=decision_result.pm25,
        pm10=decision_result.pm10,
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
            wind_direction_deg=s.get("wind_direction_deg"),
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
            wind_direction_deg=source.get("wind_direction_deg"),
            reliability_weight=source.get("reliability_weight", 1.0),
            raw=source.get("raw", {}),
        )
        for source in raw_sources
    ]


def _load_ai_payload(report: WeatherReport) -> dict[str, Any]:
    if not report.ai_analysis_data:
        return {}
    try:
        payload = json.loads(report.ai_analysis_data)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


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
    return make_decision(sources, air_quality=air_quality), ai_payload


def _build_point_source(hour_data: dict[str, Any]) -> WeatherSourceData:
    return WeatherSourceData(
        source="Open-Meteo Forecast",
        wind_speed_knots=float(hour_data["wind_speed_knots"]),
        temperature_c=float(hour_data["temperature_c"]),
        humidity_pct=float(hour_data["humidity_pct"]),
        visibility_km=float(hour_data["visibility_km"]),
        precipitation_level=int(hour_data["precipitation_level"]),
        cloud_base_ft=hour_data.get("cloud_base_ft"),
        cloud_ceiling_ft=hour_data.get("cloud_ceiling_ft"),
    )


class ReportChatRequest(BaseModel):
    message: str


class ReportChatResponse(BaseModel):
    reply: str


class ForecastHourSchema(BaseModel):
    offset: int
    decision: str
    wind_kt: float
    temperature_c: float
    humidity_pct: float
    visibility_km: float
    cloud_base_ft: float | None = None
    precipitation_level: int


class ForecastChangeResponse(BaseModel):
    hours: list[ForecastHourSchema]


class NotamResponse(BaseModel):
    notams: list[dict[str, Any]]
    disclaimer: str


class FlightPlanRequest(BaseModel):
    lat: float
    lon: float
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
    wind_kt: float
    temperature_c: float
    humidity_pct: float
    visibility_km: float
    cloud_base_ft: float | None = None
    precipitation_level: int


class FlightPlanBestWindowSchema(BaseModel):
    start_hour: int
    end_hour: int
    length: int


class FlightPlanResponse(BaseModel):
    hours: list[FlightPlanHourSchema]
    best_window: FlightPlanBestWindowSchema | None = None


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
    lat: float
    lon: float


class SeasonalMonthSchema(BaseModel):
    month: str
    avg_wind_kt: float
    avg_visibility_km: float
    suitability_score: int
    decision: str


class SeasonalResponse(BaseModel):
    months: list[SeasonalMonthSchema]


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


async def _fetch_open_meteo_hourly(
    lat: float,
    lon: float,
    *,
    timezone_name: str = "UTC",
    forecast_days: int = 3,
) -> dict[str, Any]:
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
        return float(values[index])

    temp_c = _value("temperature_2m")
    if temp_c is None:
        return None

    dew_c = _value("dew_point_2m")
    cloud_cover = _value("cloud_cover", 0.0) or 0.0
    base_ft: float | None = None
    if dew_c is not None:
        spread = max(temp_c - dew_c, 0.0)
        base_ft = spread * 122.5 * 3.28084

    return {
        "time": times[index],
        "wind_speed_knots": ((_value("wind_speed_10m", 0.0) or 0.0) * 0.539957),
        "temperature_c": temp_c,
        "humidity_pct": _value("relative_humidity_2m", 0.0) or 0.0,
        "visibility_km": ((_value("visibility", 10000.0) or 10000.0) / 1000.0),
        "cloud_base_ft": base_ft,
        "cloud_ceiling_ft": (
            base_ft + 200.0 if base_ft is not None and cloud_cover > 50 else None
        ),
        "precipitation_level": (
            0
            if ((_value("precipitation", 0.0) or 0.0) <= 0)
            else 1 if ((_value("precipitation", 0.0) or 0.0) < 2.5) else 2
        ),
    }


def _best_window(
    hours: list[FlightPlanHourSchema],
) -> FlightPlanBestWindowSchema | None:
    best: FlightPlanBestWindowSchema | None = None
    current_start: int | None = None

    for hour in hours + [
        FlightPlanHourSchema(
            hour=-1,
            decision="BREAK",
            wind_kt=0.0,
            temperature_c=0.0,
            humidity_pct=0.0,
            visibility_km=0.0,
            precipitation_level=0,
        )
    ]:
        if hour.decision == "UYGUN":
            if current_start is None:
                current_start = hour.hour
            continue
        if current_start is None:
            continue
        length = (
            hour.hour - current_start
            if hour.hour >= 0
            else (hours[-1].hour - current_start + 1)
        )
        candidate = FlightPlanBestWindowSchema(
            start_hour=current_start,
            end_hour=(hour.hour - 1) if hour.hour >= 0 else hours[-1].hour,
            length=length,
        )
        if best is None or candidate.length > best.length:
            best = candidate
        current_start = None
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

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[
                _fetch_grid_point_weather(
                    report.lat, report.lon, 1000.0, offset, client
                )
                for offset in range(1, 7)
            ]
        )

    hours: list[ForecastHourSchema] = []
    air_quality = _load_air_quality(report)
    for offset, result in enumerate(results, start=1):
        if result is None:
            continue
        decision = make_decision([_build_point_source(result)], air_quality=air_quality)
        hours.append(
            ForecastHourSchema(
                offset=offset,
                decision=decision.decision,
                wind_kt=round(float(result["wind_speed_knots"]), 1),
                temperature_c=round(float(result["temperature_c"]), 1),
                humidity_pct=round(float(result["humidity_pct"]), 1),
                visibility_km=round(float(result["visibility_km"]), 1),
                cloud_base_ft=result.get("cloud_base_ft"),
                precipitation_level=int(result["precipitation_level"]),
            )
        )

    if not hours:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tahmin verisi alınamadı.",
        )
    return ForecastChangeResponse(hours=hours)


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
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Uçuş planı verisi alınamadı.",
        ) from exc

    # Open-Meteo returns naive ISO strings (e.g. "2026-07-25T14:00") when
    # timezone=UTC, so fromisoformat yields a naive datetime.  Compare dates
    # using UTC today to stay consistent.
    base_date = datetime.now(timezone.utc).date() + timedelta(days=body.date_offset)

    def _build_hours(
        time_texts: list[str], *, require_date: bool
    ) -> list[FlightPlanHourSchema]:
        result: list[FlightPlanHourSchema] = []
        seen_hours: set[int] = set()
        for idx, time_text in enumerate(time_texts):
            timestamp = datetime.fromisoformat(time_text)
            if require_date and timestamp.date() != base_date:
                continue
            if not (body.start_hour <= timestamp.hour <= body.end_hour):
                continue
            if timestamp.hour in seen_hours:
                continue
            row = _hourly_row_from_payload(payload, idx)
            if row is None:
                continue
            decision = make_decision([_build_point_source(row)])
            result.append(
                FlightPlanHourSchema(
                    hour=timestamp.hour,
                    decision=decision.decision,
                    wind_kt=round(float(row["wind_speed_knots"]), 1),
                    temperature_c=round(float(row["temperature_c"]), 1),
                    humidity_pct=round(float(row["humidity_pct"]), 1),
                    visibility_km=round(float(row["visibility_km"]), 1),
                    cloud_base_ft=row.get("cloud_base_ft"),
                    precipitation_level=int(row["precipitation_level"]),
                )
            )
            seen_hours.add(timestamp.hour)
        return result

    time_texts = payload.get("hourly", {}).get("time", [])
    hours = _build_hours(time_texts, require_date=True)

    # Fallback: if no hours matched the strict date filter (e.g. timezone
    # mismatch between the caller's UTC date and available forecast data),
    # relax the date requirement and return the first matching hour range
    # from any available day.
    if not hours:
        hours = _build_hours(time_texts, require_date=False)

    if not hours:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Seçilen saat aralığı için veri bulunamadı.",
        )

    return FlightPlanResponse(hours=hours, best_window=_best_window(hours))


@router.get(
    "/history",
    response_model=HistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Son hava raporlarını listele",
)
async def weather_history(
    limit: int = 30,
    db: Session = Depends(get_db),
) -> HistoryResponse:
    limit = max(1, min(limit, 100))
    reports = (
        db.query(WeatherReport)
        .order_by(WeatherReport.created_at.desc())
        .limit(limit)
        .all()
    )

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
    """Compute seasonal flight suitability using ERA5 30-year climate normals.

    Tries the Open-Meteo Climate API (ERA5 1991-2020 daily data, averaged by
    month).  Falls back to latitude-based climate zone normals if the API is
    unavailable or returns unexpected data.
    """
    monthly_wind_kmh: list[float] | None = None
    monthly_precip_mm: list[float] | None = None
    monthly_temp_c: list[float] | None = None

    try:
        async with httpx.AsyncClient() as client:
            resp = await _fetch_with_retry(
                client,
                "https://climate-api.open-meteo.com/v1/climate",
                {
                    "latitude": body.lat,
                    "longitude": body.lon,
                    "daily": (
                        "wind_speed_10m_mean,precipitation_sum,"
                        "temperature_2m_mean"
                    ),
                    "start_date": "1991-01-01",
                    "end_date": "2020-12-31",
                    "models": "ERA5",
                },
            )
            payload = resp.json()

        daily = payload.get("daily", {})
        times = daily.get("time", [])
        winds = daily.get("wind_speed_10m_mean", [])
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
                if len(winds) > i and winds[i] is not None:
                    wind_sums[m] += float(winds[i])
                    counts[m] += 1
                if len(precips) > i and precips[i] is not None:
                    precip_sums[m] += float(precips[i])
                if len(temps) > i and temps[i] is not None:
                    temp_sums[m] += float(temps[i])

            if any(c > 0 for c in counts):
                monthly_wind_kmh = [
                    wind_sums[i] / counts[i] if counts[i] > 0 else 0.0
                    for i in range(12)
                ]
                monthly_precip_mm = [
                    precip_sums[i] / counts[i] if counts[i] > 0 else 0.0
                    for i in range(12)
                ]
                monthly_temp_c = [
                    temp_sums[i] / counts[i] if counts[i] > 0 else _DEFAULT_TEMP_C
                    for i in range(12)
                ]
    except Exception:
        pass  # Fall through to latitude-based fallback

    if monthly_wind_kmh is None:
        # Latitude-based climate zone fallback
        abs_lat = abs(body.lat)
        if abs_lat < _TROPICAL_LAT_THRESHOLD:  # Tropical
            monthly_wind_kmh = [
                12.0, 12.0, 13.0, 13.0, 14.0, 15.0,
                15.0, 14.0, 13.0, 12.0, 12.0, 12.0,
            ]
            monthly_precip_mm = [
                5.0, 5.0, 6.0, 8.0, 12.0, 18.0,
                20.0, 18.0, 15.0, 10.0, 7.0, 5.0,
            ]
            monthly_temp_c = [28.0] * 12
        elif abs_lat < _POLAR_LAT_THRESHOLD:  # Temperate
            monthly_wind_kmh = [
                18.0, 17.0, 16.0, 14.0, 12.0, 11.0,
                11.0, 12.0, 14.0, 16.0, 18.0, 19.0,
            ]
            monthly_precip_mm = [
                8.0, 7.0, 8.0, 7.0, 7.0, 6.0,
                5.0, 6.0, 7.0, 9.0, 10.0, 9.0,
            ]
            monthly_temp_c = [
                2.0, 3.0, 7.0, 12.0, 17.0, 21.0,
                23.0, 22.0, 18.0, 12.0, 7.0, 3.0,
            ]
        else:  # Polar
            monthly_wind_kmh = [
                25.0, 23.0, 20.0, 18.0, 15.0, 12.0,
                12.0, 14.0, 18.0, 22.0, 25.0, 27.0,
            ]
            monthly_precip_mm = [
                4.0, 3.0, 4.0, 4.0, 5.0, 6.0,
                7.0, 7.0, 6.0, 5.0, 4.0, 4.0,
            ]
            monthly_temp_c = [
                -15.0, -14.0, -9.0, -2.0, 5.0, 12.0,
                14.0, 13.0, 7.0, -1.0, -9.0, -13.0,
            ]

    months: list[SeasonalMonthSchema] = []
    for index, month_name in enumerate(_MONTHS_TR):
        wind_kmh = monthly_wind_kmh[index]
        precip = monthly_precip_mm[index] if monthly_precip_mm else 0.0
        temp = monthly_temp_c[index] if monthly_temp_c else _DEFAULT_TEMP_C

        visibility_km = (
            10.0 if precip < _PRECIP_LIGHT_MM
            else (7.0 if precip < _PRECIP_MODERATE_MM else 4.0)
        )
        precip_level = (
            0 if precip < _PRECIP_LIGHT_MM
            else (1 if precip < _PRECIP_MODERATE_MM else 2)
        )

        source = WeatherSourceData(
            source="Open-Meteo Climate",
            wind_speed_knots=wind_kmh * 0.539957,
            temperature_c=temp,
            humidity_pct=60.0,
            visibility_km=visibility_km,
            precipitation_level=precip_level,
            cloud_base_ft=None,
            cloud_ceiling_ft=None,
        )
        decision = make_decision([source])
        suitability = {
            "UYGUN": 85,
            "RISKLI": 55,
            "UYGUN_DEGIL": 25,
        }[decision.decision]
        months.append(
            SeasonalMonthSchema(
                month=month_name,
                avg_wind_kt=round(source.wind_speed_knots, 1),
                avg_visibility_km=round(visibility_km, 1),
                suitability_score=suitability,
                decision=decision.decision,
            )
        )
    return SeasonalResponse(months=months)


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
    "METAR (AviationWeather)": "Gerçek havalimanı ölçümü",
    "Windy": "Yüksek çözünürlüklü model",
    "OWM": "OpenWeatherMap anlık verisi",
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
    wind_speed_knots: float
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

    items = sorted(sources, key=lambda s: s.reliability_weight, reverse=True)

    # Weighted average consensus from sources that have wind_direction_deg
    valid = [s for s in items if s.wind_direction_deg is not None]
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No source returned a valid wind direction.",
        )

    total_weight = sum(s.reliability_weight for s in valid)
    avg_speed = (
        sum(s.wind_speed_knots * s.reliability_weight for s in valid) / total_weight
    )

    # Circular mean for wind direction
    sin_sum = sum(
        math.sin(math.radians(s.wind_direction_deg)) * s.reliability_weight
        for s in valid
    )
    cos_sum = sum(
        math.cos(math.radians(s.wind_direction_deg)) * s.reliability_weight
        for s in valid
    )
    avg_dir = math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0

    return WindResponse(
        consensus=WindConsensus(
            wind_speed_knots=round(avg_speed, 1),
            wind_direction_deg=round(avg_dir, 1),
            source_count=len(valid),
        ),
        sources=[
            WindSourceItem(
                source=s.source,
                wind_speed_knots=round(s.wind_speed_knots, 1),
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

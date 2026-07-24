"""AI decision engine — GPT-4o powered drone flight assessment.

Provides:
- ``build_feedback_context(db)``  — builds a context string from stored feedback
- ``make_ai_decision(...)``       — calls GPT-4o and returns an AIDecisionResult,
                                    falling back to rule-based make_decision() on
                                    any error or missing API key.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

try:
    from openai import AsyncOpenAI as _AsyncOpenAI  # type: ignore[import]
except ImportError:  # pragma: no cover
    _AsyncOpenAI = None  # type: ignore[assignment,misc]

from decideflight.models.feedback import Feedback
from decideflight.models.weather_report import WeatherReport
from decideflight.services.decision_engine import (
    DecisionResult,
    make_decision,
)
from decideflight.services.weather_fetcher import AirQualityData, WeatherSourceData

logger = logging.getLogger(__name__)

MAX_FEEDBACK_CONTEXT = 20
MAX_CHAT_MESSAGES = 10
_REPORT_CHAT_HISTORY: dict[int, list[dict[str, str]]] = {}


# ---------------------------------------------------------------------------
# Feedback context builder (pre-existing, kept here)
# ---------------------------------------------------------------------------


def build_feedback_context(db: Session) -> str:
    """Return the last *MAX_FEEDBACK_CONTEXT* feedback entries as a prompt.

    The returned string is suitable for prepending to a GPT system prompt so
    that the model is aware of historical decisions and whether they were
    correct according to the user.

    Returns an empty string when no feedback has been recorded yet.
    """
    rows = (
        db.query(Feedback, WeatherReport)
        .join(WeatherReport, Feedback.report_id == WeatherReport.id)
        .order_by(Feedback.created_at.desc())
        .limit(MAX_FEEDBACK_CONTEXT)
        .all()
    )
    if not rows:
        return ""

    lines = ["Geçmiş kararlar ve gerçek sonuçlar:"]
    for fb, report in reversed(rows):
        verdict = "DOĞRU" if fb.correct else "YANLIŞ"
        line = f"- Karar: {report.decision} → Kullanıcı: {verdict}"
        if fb.user_comment:
            line += f" (yorum: {fb.user_comment})"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AI result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AIDecisionResult(DecisionResult):
    """DecisionResult extended with GPT-4o analysis fields."""

    confidence: int = 0  # 0-100, GPT-estimated confidence
    summary: str = ""  # Short Turkish summary
    detailed_analysis: str = ""  # Longer Turkish analysis
    risk_factors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    parameter_assessments: dict[str, Any] = field(default_factory=dict)
    parameter_weights: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Location context helpers (Feature 3)
# ---------------------------------------------------------------------------

_SEASONS_TR = {
    12: "Kış",
    1: "Kış",
    2: "Kış",
    3: "İlkbahar",
    4: "İlkbahar",
    5: "İlkbahar",
    6: "Yaz",
    7: "Yaz",
    8: "Yaz",
    9: "Sonbahar",
    10: "Sonbahar",
    11: "Sonbahar",
}


def _build_location_context(location: str, lat: float, lon: float) -> str:
    """Build a brief location/season context string for the GPT prompt."""
    now = datetime.now(timezone.utc)
    month = now.month
    season = _SEASONS_TR.get(month, "Bilinmiyor")
    month_name = now.strftime("%B")

    # Rough coastal heuristic: within ~150 km of a coast is "coastal"
    # We use a simplified bounding-box approach.
    coastal = _is_coastal(lat, lon)
    coastal_text = "kıyı bölgesi" if coastal else "iç bölge"

    # Climate zone by latitude
    abs_lat = abs(lat)
    if abs_lat < 23.5:
        climate = "tropikal"
    elif abs_lat < 35:
        climate = "subtropikal"
    elif abs_lat < 60:
        climate = "ılıman"
    else:
        climate = "kutupsal"

    return (
        f"Konum: {location} (enlem: {lat:.2f}, boylam: {lon:.2f})\n"
        f"UTC Ay/Mevsim: {month_name} / {season}\n"
        f"Coğrafi bağlam: {coastal_text}, {climate} iklim kuşağı"
    )


def _is_coastal(lat: float, lon: float) -> bool:
    """Very rough coastal heuristic based on known coastal bounding boxes."""
    coastal_regions = [
        # Mediterranean / Turkish coasts
        (35.0, 42.0, 25.0, 37.0),
        # Black Sea coast
        (40.5, 42.5, 28.0, 42.0),
        # Atlantic coasts of Western Europe
        (35.0, 65.0, -10.0, 5.0),
        # US East Coast
        (24.0, 50.0, -82.0, -65.0),
        # US West Coast
        (30.0, 50.0, -125.0, -115.0),
        # Gulf of Mexico
        (18.0, 31.0, -98.0, -80.0),
    ]
    for lat_min, lat_max, lon_min, lon_max in coastal_regions:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return True
    return False


# ---------------------------------------------------------------------------
# GPT-4o call
# ---------------------------------------------------------------------------


async def make_ai_decision(
    sources: list[WeatherSourceData],
    location: str,
    lat: float,
    lon: float,
    feedback_context: str = "",
    wind_trend: str = "",
    air_quality: AirQualityData | None = None,
) -> AIDecisionResult:
    """Call GPT-4o for a drone flight decision analysis.

    Falls back silently to the rule-based ``make_decision()`` result wrapped in
    an ``AIDecisionResult`` if the API key is missing or any exception occurs.

    Parameters
    ----------
    sources:
        Normalised weather data from all configured sources.
    location:
        Human-readable location name.
    lat, lon:
        Coordinates for location context.
    feedback_context:
        Output of ``build_feedback_context()`` — may be empty string.
    wind_trend:
        Human-readable wind trend string, e.g. "Son 6 saatte rüzgar: 2.3 kt/h artıyor"
    """
    # Always compute the rule-based result as the fallback/baseline
    rule_result = make_decision(sources, air_quality=air_quality)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or _AsyncOpenAI is None:
        return _wrap_rule_result(rule_result)

    try:
        client = _AsyncOpenAI(api_key=api_key)
        prompt = _build_system_prompt(
            sources,
            location,
            lat,
            lon,
            feedback_context,
            wind_trend,
            rule_result,
            air_quality,
        )

        response = await client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "Hava verilerini değerlendirip drone uçuş kararı ver."
                        " JSON formatında yanıt ver."
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=1200,
        )

        raw_json = response.choices[0].message.content or "{}"
        parsed = json.loads(raw_json)
        return _parse_gpt_response(parsed, rule_result)

    except Exception as exc:
        logger.warning("GPT-4o decision failed, falling back to rule-based: %s", exc)
        return _wrap_rule_result(rule_result)


def _wrap_rule_result(rule_result: DecisionResult) -> AIDecisionResult:
    """Wrap a plain DecisionResult into an AIDecisionResult with empty AI fields."""
    return AIDecisionResult(
        decision=rule_result.decision,
        detail=rule_result.detail,
        parameters=rule_result.parameters,
        avg_wind_knots=rule_result.avg_wind_knots,
        avg_temp_c=rule_result.avg_temp_c,
        avg_humidity_pct=rule_result.avg_humidity_pct,
        avg_visibility_km=rule_result.avg_visibility_km,
        avg_precip_level=rule_result.avg_precip_level,
        avg_cloud_base_ft=rule_result.avg_cloud_base_ft,
        avg_cloud_ceiling_ft=rule_result.avg_cloud_ceiling_ft,
        confidence_score=rule_result.confidence_score,
        aqi_score=rule_result.aqi_score,
        pm25=rule_result.pm25,
        pm10=rule_result.pm10,
        confidence=rule_result.confidence_score,
        summary="",
        detailed_analysis="",
        risk_factors=[],
        recommendations=[],
        parameter_assessments={},
        parameter_weights={},
    )


def _parse_gpt_response(
    parsed: dict[str, Any], rule_result: DecisionResult
) -> AIDecisionResult:
    """Parse GPT JSON response into an AIDecisionResult."""
    decision_raw = str(parsed.get("karar", rule_result.decision)).upper()
    if decision_raw not in ("UYGUN", "RISKLI", "UYGUN_DEGIL"):
        decision_raw = rule_result.decision

    confidence = int(parsed.get("guven_skoru", rule_result.confidence_score))
    confidence = max(0, min(100, confidence))

    return AIDecisionResult(
        decision=decision_raw,
        detail=str(parsed.get("ozet", rule_result.detail)),
        parameters=rule_result.parameters,
        avg_wind_knots=rule_result.avg_wind_knots,
        avg_temp_c=rule_result.avg_temp_c,
        avg_humidity_pct=rule_result.avg_humidity_pct,
        avg_visibility_km=rule_result.avg_visibility_km,
        avg_precip_level=rule_result.avg_precip_level,
        avg_cloud_base_ft=rule_result.avg_cloud_base_ft,
        avg_cloud_ceiling_ft=rule_result.avg_cloud_ceiling_ft,
        confidence_score=confidence,
        aqi_score=rule_result.aqi_score,
        pm25=rule_result.pm25,
        pm10=rule_result.pm10,
        confidence=confidence,
        summary=str(parsed.get("ozet", "")),
        detailed_analysis=str(parsed.get("detayli_analiz", "")),
        risk_factors=list(parsed.get("risk_faktorleri", [])),
        recommendations=list(parsed.get("tavsiyeler", [])),
        parameter_assessments=dict(parsed.get("parametre_degerlendirmeleri", {})),
        parameter_weights=_parse_parameter_weights(parsed.get("parameter_weights")),
    )


def _parse_parameter_weights(raw: Any) -> dict[str, int]:
    expected = (
        "wind",
        "visibility",
        "humidity",
        "temperature",
        "precipitation",
        "cloud_base",
    )
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, int] = {}
    for key in expected:
        value = raw.get(key)
        try:
            parsed[key] = max(0, min(100, int(float(value))))
        except (TypeError, ValueError):
            continue
    return parsed


def _build_system_prompt(
    sources: list[WeatherSourceData],
    location: str,
    lat: float,
    lon: float,
    feedback_context: str,
    wind_trend: str,
    rule_result: DecisionResult,
    air_quality: AirQualityData | None,
) -> str:
    """Build the full GPT-4o system prompt in Turkish."""
    location_context = _build_location_context(location, lat, lon)

    # Build weather data summary
    weather_lines: list[str] = []
    for s in sources:
        weather_lines.append(
            f"Kaynak: {s.source}\n"
            f"  Rüzgar: {s.wind_speed_knots:.1f} knot\n"
            f"  Sıcaklık: {s.temperature_c:.1f} °C\n"
            f"  Nem: {s.humidity_pct:.0f}%\n"
            f"  Görüş: {s.visibility_km:.1f} km\n"
            f"  Yağış seviyesi: {s.precipitation_level} (0=yok, 1=hafif, 2=ağır)\n"
            f"  Bulut tabanı: "
            f"{f'{s.cloud_base_ft:.0f} ft' if s.cloud_base_ft is not None else 'N/A'}\n"
            "  Bulut tavanı: "
            + (
                f"{s.cloud_ceiling_ft:.0f} ft"
                if s.cloud_ceiling_ft is not None
                else "N/A"
            )
        )

    weather_summary = "\n\n".join(weather_lines)
    air_quality_section = ""
    if air_quality is not None:
        air_quality_section = (
            "\nHAVA KALİTESİ:\n"
            f"  Avrupa AQI: {air_quality.aqi_score}\n"
            f"  PM2.5: {air_quality.pm25 if air_quality.pm25 is not None else 'N/A'}\n"
            f"  PM10: {air_quality.pm10 if air_quality.pm10 is not None else 'N/A'}"
        )

    avg_section = (
        f"Ortalamalar (ağırlıklı):\n"
        f"  Rüzgar: {rule_result.avg_wind_knots:.1f} knot\n"
        f"  Sıcaklık: {rule_result.avg_temp_c:.1f} °C\n"
        f"  Nem: {rule_result.avg_humidity_pct:.0f}%\n"
        f"  Görüş: {rule_result.avg_visibility_km:.1f} km\n"
        f"  Bulut tabanı: "
        f"{f'{rule_result.avg_cloud_base_ft:.0f} ft' if rule_result.avg_cloud_base_ft is not None else 'N/A'}\n"  # noqa: E501
        f"  Bulut tavanı: "
        f"{f'{rule_result.avg_cloud_ceiling_ft:.0f} ft' if rule_result.avg_cloud_ceiling_ft is not None else 'N/A'}"  # noqa: E501
    )

    rule_decision_section = (
        f"Kural tabanlı karar: {rule_result.decision} "
        f"(güven: {rule_result.confidence_score}%)"
    )

    trend_section = f"\nRüzgar trendi: {wind_trend}" if wind_trend else ""
    feedback_section = f"\n\n{feedback_context}" if feedback_context else ""

    response_schema = """{
  "karar": "UYGUN" | "RISKLI" | "UYGUN_DEGIL",
  "guven_skoru": 0-100 arası tam sayı,
  "ozet": "Kısa Türkçe özet cümlesi",
  "detayli_analiz": "Kapsamlı Türkçe analiz paragrafı",
  "risk_faktorleri": ["Risk 1", "Risk 2", ...],
  "tavsiyeler": ["Tavsiye 1", "Tavsiye 2", ...],
  "parameter_weights": {
    "wind": 0-100,
    "visibility": 0-100,
    "humidity": 0-100,
    "temperature": 0-100,
    "precipitation": 0-100,
    "cloud_base": 0-100
  },
  "parametre_degerlendirmeleri": {
    "ruzgar": "değerlendirme",
    "nem": "değerlendirme",
    ...
  }
}"""

    return f"""Sen bir drone uçuş güvenlik uzmanısın. Hava koşullarını analiz ederek \
uçuş kararı vereceksin.

GÖREV: Aşağıdaki hava verilerini değerlendirerek drone uçuşunun UYGUN, RISKLI veya \
UYGUN_DEGIL olduğunu Türkçe olarak açıkla.

KONUM BİLGİSİ:
{location_context}

HAVA VERİLERİ:
{weather_summary}
{air_quality_section}

{avg_section}
{rule_decision_section}{trend_section}

ÖĞRENİLEN PATERNLER (geçmiş geri bildirimlerden öğren, varsa dikkate al):
{feedback_section if feedback_section else "(Henüz geri bildirim yok)"}

YANIT FORMATI (kesinlikle geçerli JSON döndür):
{response_schema}

KURALLAR:
- Tüm metin alanları Türkçe olacak
- karar alanı yalnızca UYGUN, RISKLI veya UYGUN_DEGIL olabilir
- guven_skoru 0-100 arasında tam sayı olacak
- risk_faktorleri en az 1, en fazla 5 madde içerecek
- tavsiyeler en az 1, en fazla 5 madde içerecek
- parameter_weights alanındaki tüm sayılar 0 ile 100 arasında tam sayı olacak
- Konum ve mevsim bağlamını dikkate al (tropikal bölgede muson, kıyıda deniz esintisi)
- Geçmiş geri bildirim varsa, benzer koşullardaki önceki hatalardan öğren"""


def _build_chat_context(
    location: str,
    report_created_at: datetime,
    sources: list[WeatherSourceData],
    decision_result: DecisionResult,
    ai_summary: str,
) -> str:
    source_lines = []
    for source in sources:
        source_lines.append(
            f"- {source.source}: rüzgar {source.wind_speed_knots:.1f} kt, "
            f"görüş {source.visibility_km:.1f} km, "
            f"sıcaklık {source.temperature_c:.1f} °C, "
            f"nem %{source.humidity_pct:.0f}, "
            f"yağış seviyesi {source.precipitation_level}"
        )
    params_text = "\n".join(
        f"- {param.name}: {param.value} ({param.decision})"
        for param in decision_result.parameters
    )
    summary_section = f"\nAI özeti: {ai_summary}" if ai_summary else ""
    return (
        "Sen DecideFlight içinde rapor bazlı sohbet yapan bir uçuş hava uzmanısın.\n"
        "Yanıtların kısa, Türkçe ve bağlama dayalı olmalı.\n"
        f"Rapor zamanı: {report_created_at.isoformat()}\n"
        f"Konum: {location}\n"
        f"Nihai karar: {decision_result.decision}\n"
        f"Karar özeti: {decision_result.detail}\n"
        f"Parametreler:\n{params_text}\n"
        f"Kaynaklar:\n" + "\n".join(source_lines) + summary_section
    )


def _fallback_chat_reply(
    message: str,
    decision_result: DecisionResult,
    ai_summary: str,
) -> str:
    lowered = message.lower()
    if "rüzgar" in lowered:
        return (
            "Rüzgar tarafında mevcut değerlendirme "
            f"{decision_result.avg_wind_knots:.1f} kt. "
            f"Genel karar şu an {decision_result.decision}."
        )
    if "güven" in lowered or "güvenli" in lowered:
        return (
            f"Bu rapora göre genel karar {decision_result.decision}. "
            "Uçuş öncesinde sahadaki anlık koşulları yeniden doğrulayın."
        )
    if "ne zaman" in lowered:
        return (
            "Zaman bazlı ayrıntı için alt bölümdeki tahmin ve "
            "uçuş planlama panellerini kullanabilirsiniz."
        )
    if ai_summary:
        return ai_summary
    return (
        f"Bu raporda genel karar {decision_result.decision}. "
        "Detaylı parametreleri aşağıdaki analiz panellerinden inceleyebilirsiniz."
    )


async def chat_about_report(
    report_id: int,
    location: str,
    report_created_at: datetime,
    sources: list[WeatherSourceData],
    decision_result: DecisionResult,
    message: str,
    ai_summary: str = "",
) -> str:
    history = list(_REPORT_CHAT_HISTORY.get(report_id, []))
    api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key or _AsyncOpenAI is None:
        reply = _fallback_chat_reply(message, decision_result, ai_summary)
    else:
        try:
            client = _AsyncOpenAI(api_key=api_key)
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": _build_chat_context(
                            location,
                            report_created_at,
                            sources,
                            decision_result,
                            ai_summary,
                        ),
                    },
                    *history,
                    {"role": "user", "content": message},
                ],
                temperature=0.4,
                max_tokens=500,
            )
            reply = (response.choices[0].message.content or "").strip()
            if not reply:
                reply = _fallback_chat_reply(message, decision_result, ai_summary)
        except Exception as exc:
            logger.warning("GPT-4o chat failed, falling back: %s", exc)
            reply = _fallback_chat_reply(message, decision_result, ai_summary)

    history.extend(
        [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ]
    )
    _REPORT_CHAT_HISTORY[report_id] = history[-MAX_CHAT_MESSAGES:]
    return reply

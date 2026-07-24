"""AI-powered decision engine using OpenAI GPT-4o.

Sends normalised weather data from all sources to GPT-4o and returns a
structured flight-suitability decision.  If ``OPENAI_API_KEY`` is not
configured or the API call fails, the function silently falls back to the
rule-based :func:`decideflight.services.decision_engine.make_decision`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, field_validator

from decideflight.config import settings
from decideflight.services.decision_engine import (
    DecisionResult,
    make_decision as _rule_based_decision,
)
from decideflight.services.weather_fetcher import WeatherSourceData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt (Turkish) – built from settings to keep limits consistent
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
Sen bir drone uçuş güvenlik uzmanısın. Verilen meteoroloji verilerini analiz ederek \
drone uçuşunun güvenli olup olmadığına karar veriyorsun.

Drone limitleri:
- Rüzgar: <{wind_ok} knot UYGUN, {wind_ok}-{wind_risky} knot RISKLI, \
>{wind_risky} knot UYGUN_DEGIL
- Görüş mesafesi: >{vis_ok}km UYGUN, {vis_risky}-{vis_ok}km RISKLI, \
<{vis_risky}km UYGUN_DEGIL
- Yağış: Yok UYGUN, Hafif RISKLI, Orta/Şiddetli UYGUN_DEGIL
- Sıcaklık: {temp_ok_min}-{temp_ok_max}°C UYGUN, \
{temp_risky_min}~{temp_ok_min} veya {temp_ok_max}-{temp_risky_max}°C RISKLI, \
dışı UYGUN_DEGIL
- Nem: <%{hum_ok} UYGUN, %{hum_ok}-{hum_risky} RISKLI, >%{hum_risky} UYGUN_DEGIL
- Bulut tabanı: >{cb_ok}ft UYGUN, {cb_risky}-{cb_ok}ft RISKLI, \
<{cb_risky}ft UYGUN_DEGIL
- Bulut tavanı: >{cc_ok}ft UYGUN, {cc_risky}-{cc_ok}ft RISKLI, \
<{cc_risky}ft UYGUN_DEGIL

Birden fazla kaynaktan veri geliyorsa hepsini değerlendir, çelişen verilere dikkat et.
Sadece limitlerle değil, genel hava durumu tablosuna bakarak bütüncül bir karar ver.
Türkçe yanıt ver.

ÖNEMLİ: Yanıtını YALNIZCA aşağıdaki JSON formatında ver, başka hiçbir metin ekleme:
{{
  "decision": "UYGUN" veya "RISKLI" veya "UYGUN_DEGIL",
  "confidence": 0-100 arası tam sayı,
  "summary": "Kısa Türkçe özet (1-2 cümle)",
  "detailed_analysis": "Detaylı Türkçe analiz paragrafı",
  "risk_factors": ["risk1", "risk2"],
  "recommendations": ["öneri1", "öneri2"],
  "parameter_assessments": {{
    "wind": {{"value": 12.3, "unit": "knot", "status": "UYGUN", "comment": "..."}},
    "visibility": {{"value": 10.0, "unit": "km", "status": "UYGUN", "comment": "..."}},
    "precipitation": {{"value": 0, "unit": "level", "status": "UYGUN", "comment": "..."}},
    "temperature": {{"value": 22.5, "unit": "°C", "status": "UYGUN", "comment": "..."}},
    "humidity": {{"value": 65.0, "unit": "%", "status": "UYGUN", "comment": "..."}},
    "cloud_base": {{"value": 1500.0, "unit": "ft", "status": "UYGUN", "comment": "..."}},
    "cloud_ceiling": {{"value": 2000.0, "unit": "ft", "status": "UYGUN", "comment": "..."}}
  }}
}}
"""


def _build_system_prompt() -> str:
    """Build the system prompt with limit values sourced from settings."""
    return _SYSTEM_PROMPT_TEMPLATE.format(
        wind_ok=int(settings.wind_ok_max_knots),
        wind_risky=int(settings.wind_risky_max_knots),
        vis_ok=int(settings.visibility_ok_min_km),
        vis_risky=int(settings.visibility_risky_min_km),
        temp_ok_min=int(settings.temp_ok_min_c),
        temp_ok_max=int(settings.temp_ok_max_c),
        temp_risky_min=int(settings.temp_risky_min_c),
        temp_risky_max=int(settings.temp_risky_max_c),
        hum_ok=int(settings.humidity_ok_max_pct),
        hum_risky=int(settings.humidity_risky_max_pct),
        cb_ok=int(settings.cloud_base_ok_min_ft),
        cb_risky=int(settings.cloud_base_risky_min_ft),
        cc_ok=int(settings.cloud_ceiling_ok_min_ft),
        cc_risky=int(settings.cloud_ceiling_risky_min_ft),
    )


# ---------------------------------------------------------------------------
# Pydantic models for GPT-4o structured output
# ---------------------------------------------------------------------------

_VALID_DECISIONS = {"UYGUN", "RISKLI", "UYGUN_DEGIL"}

# Aliases GPT-4o might use for the decision values
_DECISION_ALIASES: dict[str, str] = {
    "UYGUN DEĞİL": "UYGUN_DEGIL",
    "UYGUN_DEĞİL": "UYGUN_DEGIL",
    "RİSKLİ": "RISKLI",
    "RISKLI": "RISKLI",
    "UYGUN": "UYGUN",
    "UYGUN_DEGIL": "UYGUN_DEGIL",
}


class _ParameterAssessment(BaseModel):
    value: float
    unit: str
    status: str
    comment: str


class _AIFlightDecision(BaseModel):
    decision: str
    confidence: int
    summary: str
    detailed_analysis: str
    risk_factors: list[str]
    recommendations: list[str]
    parameter_assessments: dict[str, _ParameterAssessment]

    @field_validator("decision")
    @classmethod
    def _normalise_decision(cls, v: str) -> str:
        normalised = _DECISION_ALIASES.get(v.strip().upper(), v.strip().upper())
        if normalised not in _VALID_DECISIONS:
            raise ValueError(
                f"Invalid decision value '{v}'. " f"Must be one of {_VALID_DECISIONS}."
            )
        return normalised


# ---------------------------------------------------------------------------
# Extended result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AIDecisionResult(DecisionResult):
    """``DecisionResult`` extended with GPT-4o analysis fields."""

    confidence: int = 0
    summary: str = ""
    detailed_analysis: str = ""
    risk_factors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    parameter_assessments: dict[str, Any] = field(default_factory=dict)

    def ai_extra_as_dict(self) -> dict[str, Any]:
        """Return the AI-specific fields as a plain dict (for DB storage)."""
        return {
            "confidence": self.confidence,
            "summary": self.summary,
            "detailed_analysis": self.detailed_analysis,
            "risk_factors": self.risk_factors,
            "recommendations": self.recommendations,
            "parameter_assessments": self.parameter_assessments,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PRECIP_LABELS = {0: "Yok", 1: "Hafif", 2: "Orta/Şiddetli"}


def _fmt_cloud(value: float | None) -> str:
    """Format an optional cloud altitude value for display."""
    return f"{value:.0f} ft" if value is not None else "N/A"


def _build_user_message(sources: list[WeatherSourceData]) -> str:
    lines: list[str] = [
        "Aşağıdaki hava kaynaklarından toplanan verilere göre "
        "drone uçuşu değerlendirmesi yapmanı istiyorum:\n"
    ]
    for src in sources:
        lines.append(
            f"Kaynak: {src.source}\n"
            f"  - Rüzgar: {src.wind_speed_knots:.1f} knot\n"
            f"  - Görüş: {src.visibility_km:.1f} km\n"
            f"  - Sıcaklık: {src.temperature_c:.1f} °C\n"
            f"  - Nem: {src.humidity_pct:.0f}%\n"
            f"  - Yağış: {_PRECIP_LABELS.get(src.precipitation_level, 'Bilinmiyor')}\n"
            f"  - Bulut Tabanı: {_fmt_cloud(src.cloud_base_ft)}\n"
            f"  - Bulut Tavanı: {_fmt_cloud(src.cloud_ceiling_ft)}\n"
        )
    lines.append(
        "\nBu drone için uçuş yapılabilir mi? " "Belirtilen JSON formatında yanıt ver."
    )
    return "\n".join(lines)


def _reconstruct_from_extra(
    rule_result: DecisionResult,
    extra: dict[str, Any],
) -> AIDecisionResult:
    """Rebuild an :class:`AIDecisionResult` from stored extra-field JSON."""
    return AIDecisionResult(
        decision=extra.get("decision", rule_result.decision),
        detail=extra.get("summary", rule_result.detail),
        parameters=rule_result.parameters,
        avg_wind_knots=rule_result.avg_wind_knots,
        avg_temp_c=rule_result.avg_temp_c,
        avg_humidity_pct=rule_result.avg_humidity_pct,
        avg_visibility_km=rule_result.avg_visibility_km,
        avg_precip_level=rule_result.avg_precip_level,
        avg_cloud_base_ft=rule_result.avg_cloud_base_ft,
        avg_cloud_ceiling_ft=rule_result.avg_cloud_ceiling_ft,
        confidence=extra.get("confidence", 0),
        summary=extra.get("summary", ""),
        detailed_analysis=extra.get("detailed_analysis", ""),
        risk_factors=extra.get("risk_factors", []),
        recommendations=extra.get("recommendations", []),
        parameter_assessments=extra.get("parameter_assessments", {}),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def make_ai_decision(sources: list[WeatherSourceData]) -> DecisionResult:
    """Return a flight-suitability decision, preferring GPT-4o.

    Falls back to the rule-based engine when:
    * ``OPENAI_API_KEY`` is not set, or
    * the API call raises any exception.
    """
    if not settings.openai_api_key:
        logger.info("OPENAI_API_KEY not configured; using rule-based decision engine")
        return _rule_based_decision(sources)

    try:
        from openai import AsyncOpenAI  # imported lazily to keep startup fast

        client = AsyncOpenAI(api_key=settings.openai_api_key)
        user_message = _build_user_message(sources)

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _build_system_prompt()},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from GPT-4o")

        ai_result = _AIFlightDecision.model_validate_json(content)

        # Use rule-based engine to obtain the aggregated numeric averages and
        # per-parameter objects (these are needed by report_generator).
        rule_result = _rule_based_decision(sources)

        return AIDecisionResult(
            decision=ai_result.decision,
            detail=ai_result.summary,
            parameters=rule_result.parameters,
            avg_wind_knots=rule_result.avg_wind_knots,
            avg_temp_c=rule_result.avg_temp_c,
            avg_humidity_pct=rule_result.avg_humidity_pct,
            avg_visibility_km=rule_result.avg_visibility_km,
            avg_precip_level=rule_result.avg_precip_level,
            avg_cloud_base_ft=rule_result.avg_cloud_base_ft,
            avg_cloud_ceiling_ft=rule_result.avg_cloud_ceiling_ft,
            confidence=ai_result.confidence,
            summary=ai_result.summary,
            detailed_analysis=ai_result.detailed_analysis,
            risk_factors=ai_result.risk_factors,
            recommendations=ai_result.recommendations,
            parameter_assessments={
                k: v.model_dump() for k, v in ai_result.parameter_assessments.items()
            },
        )

    except Exception as exc:
        logger.warning(
            "AI decision engine failed (%s); falling back to rule-based engine",
            exc,
        )
        return _rule_based_decision(sources)


def reconstruct_ai_decision(
    sources: list[WeatherSourceData],
    ai_analysis_json: str | None,
) -> DecisionResult:
    """Reconstruct a :class:`DecisionResult` (sync) for PDF generation.

    If *ai_analysis_json* is provided it is used to build an
    :class:`AIDecisionResult`; otherwise the plain rule-based result is
    returned.
    """
    rule_result = _rule_based_decision(sources)
    if not ai_analysis_json:
        return rule_result

    try:
        extra: dict[str, Any] = json.loads(ai_analysis_json)
        return _reconstruct_from_extra(rule_result, extra)
    except Exception as exc:
        logger.warning(
            "Failed to reconstruct AI decision from stored JSON (%s); "
            "using rule-based result",
            exc,
        )
        return rule_result

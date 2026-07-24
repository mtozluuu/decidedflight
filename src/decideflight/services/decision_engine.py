"""Decision engine: evaluates normalised weather data against drone limits.

Each parameter is scored as:
  0 – UYGUN       (suitable)
  1 – RISKLI      (risky)
  2 – UYGUN_DEGIL (unsuitable)

The overall decision is the worst score across all averaged parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from decideflight.config import settings
from decideflight.services.weather_fetcher import (
    AirQualityData,
    PRECIP_HEAVY,
    PRECIP_LIGHT,
    PRECIP_NONE,
    WeatherSourceData,
)

# ---------------------------------------------------------------------------
# Decision constants
# ---------------------------------------------------------------------------
UYGUN = "UYGUN"
RISKLI = "RISKLI"
UYGUN_DEGIL = "UYGUN_DEGIL"

_SCORE = {UYGUN: 0, RISKLI: 1, UYGUN_DEGIL: 2}
_LABEL = {0: UYGUN, 1: RISKLI, 2: UYGUN_DEGIL}

_PRECIP_LABELS = {
    PRECIP_NONE: "Yok",
    PRECIP_LIGHT: "Hafif",
    PRECIP_HEAVY: "Orta/Şiddetli",
}


# ---------------------------------------------------------------------------
# Parameter evaluation helpers
# ---------------------------------------------------------------------------


def _eval_wind(knots: float) -> str:
    if knots < settings.wind_ok_max_knots:
        return UYGUN
    if knots <= settings.wind_risky_max_knots:
        return RISKLI
    return UYGUN_DEGIL


def _eval_visibility(km: float) -> str:
    if km > settings.visibility_ok_min_km:
        return UYGUN
    if km >= settings.visibility_risky_min_km:
        return RISKLI
    return UYGUN_DEGIL


def _eval_precipitation(level: int) -> str:
    if level == PRECIP_NONE:
        return UYGUN
    if level == PRECIP_LIGHT:
        return RISKLI
    return UYGUN_DEGIL


def _eval_temperature(temp_c: float) -> str:
    if settings.temp_ok_min_c <= temp_c <= settings.temp_ok_max_c:
        return UYGUN
    if settings.temp_risky_min_c <= temp_c <= settings.temp_risky_max_c:
        return RISKLI
    return UYGUN_DEGIL


def _eval_humidity(pct: float) -> str:
    if pct < settings.humidity_ok_max_pct:
        return UYGUN
    if pct <= settings.humidity_risky_max_pct:
        return RISKLI
    return UYGUN_DEGIL


def _eval_cloud_base(ft: float | None) -> str:
    if ft is None:
        return UYGUN  # unknown → assume OK
    if ft > settings.cloud_base_ok_min_ft:
        return UYGUN
    if ft >= settings.cloud_base_risky_min_ft:
        return RISKLI
    return UYGUN_DEGIL


def _eval_cloud_ceiling(ft: float | None) -> str:
    if ft is None:
        return UYGUN
    if ft > settings.cloud_ceiling_ok_min_ft:
        return UYGUN
    if ft >= settings.cloud_ceiling_risky_min_ft:
        return RISKLI
    return UYGUN_DEGIL


def _eval_aqi(aqi_score: float | None) -> str:
    if aqi_score is None:
        return UYGUN
    if aqi_score > 150:
        return UYGUN_DEGIL
    if aqi_score > 100:
        return RISKLI
    return UYGUN


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _avg_optional(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    return _avg(valid) if valid else None


def _weighted_avg(values: list[tuple[float, float]]) -> float:
    if not values:
        return 0.0
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return _avg([value for value, _ in values])
    return sum(value * weight for value, weight in values) / total_weight


def _weighted_avg_optional(values: list[tuple[float | None, float]]) -> float | None:
    valid = [(value, weight) for value, weight in values if value is not None]
    if not valid:
        return None
    return _weighted_avg([(float(value), weight) for value, weight in valid])


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ParameterResult:
    name: str
    value: str  # human-readable value with unit
    decision: str


@dataclass
class DecisionResult:
    decision: str  # UYGUN / RISKLI / UYGUN_DEGIL
    detail: str  # human-readable summary
    parameters: list[ParameterResult]
    avg_wind_knots: float
    avg_temp_c: float
    avg_humidity_pct: float
    avg_visibility_km: float
    avg_precip_level: float
    avg_cloud_base_ft: float | None
    avg_cloud_ceiling_ft: float | None
    confidence_score: int
    aqi_score: int | None = None
    pm25: float | None = None
    pm10: float | None = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def make_decision(
    sources: list[WeatherSourceData],
    air_quality: AirQualityData | None = None,
) -> DecisionResult:
    """Aggregate *sources* and return a ``DecisionResult``."""

    def _weighted_pairs(
        getter: Callable[[WeatherSourceData], float | None],
    ) -> list[tuple[float | None, float]]:
        return [(getter(source), source.reliability_weight) for source in sources]

    avg_wind = _weighted_avg(_weighted_pairs(lambda s: s.wind_speed_knots))
    avg_temp = _weighted_avg(_weighted_pairs(lambda s: s.temperature_c))
    avg_humidity = _weighted_avg(_weighted_pairs(lambda s: s.humidity_pct))
    avg_visibility = _weighted_avg(_weighted_pairs(lambda s: s.visibility_km))
    # For precipitation use worst (max) across sources
    max_precip = max(s.precipitation_level for s in sources)
    avg_cloud_base = _weighted_avg_optional(_weighted_pairs(lambda s: s.cloud_base_ft))
    avg_cloud_ceiling = _weighted_avg_optional(
        _weighted_pairs(lambda s: s.cloud_ceiling_ft)
    )

    params: list[ParameterResult] = [
        ParameterResult(
            name="Rüzgar hızı",
            value=f"{avg_wind:.1f} knot",
            decision=_eval_wind(avg_wind),
        ),
        ParameterResult(
            name="Görüş mesafesi",
            value=f"{avg_visibility:.1f} km",
            decision=_eval_visibility(avg_visibility),
        ),
        ParameterResult(
            name="Yağış",
            value=_PRECIP_LABELS.get(max_precip, "Bilinmiyor"),
            decision=_eval_precipitation(max_precip),
        ),
        ParameterResult(
            name="Sıcaklık",
            value=f"{avg_temp:.1f} °C",
            decision=_eval_temperature(avg_temp),
        ),
        ParameterResult(
            name="Nem",
            value=f"{avg_humidity:.0f}%",
            decision=_eval_humidity(avg_humidity),
        ),
        ParameterResult(
            name="Bulut tabanı",
            value=(f"{avg_cloud_base:.0f} ft" if avg_cloud_base is not None else "N/A"),
            decision=_eval_cloud_base(avg_cloud_base),
        ),
        ParameterResult(
            name="Bulut tavanı",
            value=(
                f"{avg_cloud_ceiling:.0f} ft"
                if avg_cloud_ceiling is not None
                else "N/A"
            ),
            decision=_eval_cloud_ceiling(avg_cloud_ceiling),
        ),
    ]
    if air_quality is not None:
        params.append(
            ParameterResult(
                name="Hava kalitesi",
                value=f"Avrupa AQI {air_quality.aqi_score}",
                decision=_eval_aqi(air_quality.aqi_score),
            )
        )

    worst_score = max(_SCORE[p.decision] for p in params)
    overall = _LABEL[worst_score]

    # Build detail message
    bad = [p for p in params if p.decision != UYGUN]
    if not bad:
        detail = "Tüm parametreler uygun sınırlar içinde."
    else:
        lines = []
        for p in bad:
            label = "⚠️ Riskli" if p.decision == RISKLI else "❌ Uygun Değil"
            lines.append(f"{p.name}: {p.value} – {label}")
        detail = "\n".join(lines)

    metar_sources = [s for s in sources if s.source == "METAR (AVWX)"]
    metar_conf_bonus = 0
    if metar_sources:
        metar = metar_sources[0]
        raw_metar = str(metar.raw.get("raw_metar") or "").strip() or "N/A"
        taf_summary = str(metar.raw.get("taf_summary_next_6h") or "").strip() or "N/A"
        visibility_text = str(metar.raw.get("visibility_text") or "N/A")
        wind_dir = metar.raw.get("metar", {}).get("wind_direction")
        wind_dir_val: int | None = None
        if isinstance(wind_dir, dict) and wind_dir.get("value") is not None:
            try:
                wind_dir_val = int(wind_dir.get("value"))
            except (TypeError, ValueError):
                wind_dir_val = None
        wind_dir_text = f", yön: {wind_dir_val}°" if wind_dir_val is not None else ""
        temperature_text = (
            f"{metar.temperature_c:.0f}°C" if metar.temperature_c is not None else "N/A"
        )
        cloud_text = (
            f"{metar.cloud_base_ft:.0f}ft" if metar.cloud_base_ft is not None else "N/A"
        )
        detail += (
            "\n\nMETAR (Resmi Havacılık Gözlemi):\n"
            f"Ham veri: {raw_metar}\n"
            f"- Rüzgar: {metar.wind_speed_knots:.0f} knot{wind_dir_text}\n"
            f"- Görüş: {visibility_text}\n"
            f"- Bulut tabanı: {cloud_text}\n"
            f"- Sıcaklık: {temperature_text}\n\n"
            "TAF (Terminal Hava Tahmini - Sonraki 6 saat):\n"
            f"{taf_summary}"
        )
        metar_conf_bonus = 12

    score_map = {UYGUN: 100, RISKLI: 50, UYGUN_DEGIL: 0}
    base_confidence = round(sum(score_map[p.decision] for p in params) / len(params))
    confidence_score = max(0, min(100, base_confidence + metar_conf_bonus))

    return DecisionResult(
        decision=overall,
        detail=detail,
        parameters=params,
        avg_wind_knots=avg_wind,
        avg_temp_c=avg_temp,
        avg_humidity_pct=avg_humidity,
        avg_visibility_km=avg_visibility,
        avg_precip_level=float(max_precip),
        avg_cloud_base_ft=avg_cloud_base,
        avg_cloud_ceiling_ft=avg_cloud_ceiling,
        confidence_score=confidence_score,
        aqi_score=air_quality.aqi_score if air_quality is not None else None,
        pm25=air_quality.pm25 if air_quality is not None else None,
        pm10=air_quality.pm10 if air_quality is not None else None,
    )

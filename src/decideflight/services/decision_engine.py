"""Decision engine: evaluates normalised weather data against drone limits.

Each parameter is scored as:
  0 – UYGUN       (suitable)
  1 – RISKLI      (risky)
  2 – UYGUN_DEGIL (unsuitable)

The overall decision is the worst score across all averaged parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from decideflight.config import Settings, settings
from decideflight.services.flight_limits import DroneLimit
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
VERI_YETERSIZ = "VERI_YETERSIZ"

_SCORE = {UYGUN: 0, RISKLI: 1, UYGUN_DEGIL: 2, VERI_YETERSIZ: 3}
_LABEL = {0: UYGUN, 1: RISKLI, 2: UYGUN_DEGIL}

_PRECIP_LABELS = {
    PRECIP_NONE: "Yok",
    PRECIP_LIGHT: "Hafif",
    PRECIP_HEAVY: "Orta/Şiddetli",
}


# ---------------------------------------------------------------------------
# Parameter evaluation helpers
# ---------------------------------------------------------------------------


def _eval_wind(knots: float | None, limits: DroneLimit | Settings = settings) -> str:
    if knots is None:
        return VERI_YETERSIZ
    if knots < limits.wind_ok_max_knots:
        return UYGUN
    if knots <= limits.wind_risky_max_knots:
        return RISKLI
    return UYGUN_DEGIL


def _eval_visibility(km: float | None, limits: DroneLimit | Settings = settings) -> str:
    if km is None:
        return VERI_YETERSIZ
    if km > limits.visibility_ok_min_km:
        return UYGUN
    if km >= limits.visibility_risky_min_km:
        return RISKLI
    return UYGUN_DEGIL


def _eval_precipitation(level: int | None) -> str:
    if level is None:
        return VERI_YETERSIZ
    if level == PRECIP_NONE:
        return UYGUN
    if level == PRECIP_LIGHT:
        return RISKLI
    return UYGUN_DEGIL


def _eval_temperature(temp_c: float | None, limits: DroneLimit | Settings = settings) -> str:
    if temp_c is None:
        return VERI_YETERSIZ
    if limits.temp_ok_min_c <= temp_c <= limits.temp_ok_max_c:
        return UYGUN
    if limits.temp_risky_min_c <= temp_c <= limits.temp_risky_max_c:
        return RISKLI
    return UYGUN_DEGIL


def _eval_humidity(pct: float | None, limits: DroneLimit | Settings = settings) -> str:
    if pct is None:
        return VERI_YETERSIZ
    if pct < limits.humidity_ok_max_pct:
        return UYGUN
    if pct <= limits.humidity_risky_max_pct:
        return RISKLI
    return UYGUN_DEGIL


def _eval_cloud_base(ft: float | None, limits: DroneLimit | Settings = settings) -> str:
    if ft is None:
        return VERI_YETERSIZ
    if ft > limits.cloud_base_ok_min_ft:
        return UYGUN
    if ft >= limits.cloud_base_risky_min_ft:
        return RISKLI
    return UYGUN_DEGIL


def _eval_cloud_ceiling(ft: float | None, limits: DroneLimit | Settings = settings) -> str:
    if ft is None:
        return VERI_YETERSIZ
    if ft > limits.cloud_ceiling_ok_min_ft:
        return UYGUN
    if ft >= limits.cloud_ceiling_risky_min_ft:
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
    explanation: str = ""


@dataclass
class DecisionResult:
    decision: str  # UYGUN / RISKLI / UYGUN_DEGIL
    detail: str  # human-readable summary
    parameters: list[ParameterResult]
    avg_wind_knots: float | None
    avg_temp_c: float | None
    avg_humidity_pct: float | None
    avg_visibility_km: float | None
    avg_precip_level: float | None
    avg_cloud_base_ft: float | None
    avg_cloud_ceiling_ft: float | None
    confidence_score: int
    aqi_score: int | None = None
    pm25: float | None = None
    pm10: float | None = None
    avg_cloud_cover_pct: float | None = None
    missing_parameters: list[str] = field(default_factory=list)
    known_decision: str = UYGUN
    excluded_sources: list[str] = field(default_factory=list)


def usable_sources(sources: list[WeatherSourceData]) -> tuple[list[WeatherSourceData], list[str]]:
    from decideflight.services.data_freshness import source_timing

    selected = []
    excluded = []
    observations = set()
    now = datetime.now(timezone.utc)
    for source in sources:
        if source_timing(source, now)["freshness"] == "stale":
            excluded.append(f"{source.source}: eski veri")
            continue
        if source.station_distance_km is not None and source.station_distance_km > 100:
            excluded.append(f"{source.source}: istasyon 100 km sınırının dışında")
            continue
        identity = " ".join((source.observation_id or "").split())
        identity = identity.removeprefix("METAR ").removeprefix("SPECI ")
        if identity and identity in observations:
            excluded.append(f"{source.source}: aynı gözlemin tekrarı")
            continue
        if identity:
            observations.add(identity)
        selected.append(source)
    return selected, excluded


def wind_consensus(sources: list[WeatherSourceData]) -> tuple[float, float, int] | None:
    sources, _ = usable_sources(sources)
    valid = [s for s in sources if s.wind_speed_knots is not None and s.wind_direction_deg is not None]
    if not valid:
        return None
    sin_sum = sum(math.sin(math.radians(s.wind_direction_deg)) * s.reliability_weight for s in valid)
    cos_sum = sum(math.cos(math.radians(s.wind_direction_deg)) * s.reliability_weight for s in valid)
    if math.hypot(sin_sum, cos_sum) < 0.1 * sum(s.reliability_weight for s in valid):
        return None
    speed = _weighted_avg([(s.wind_speed_knots, s.reliability_weight) for s in valid])
    return speed, math.degrees(math.atan2(sin_sum, cos_sum)) % 360, len(valid)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def make_decision(
    sources: list[WeatherSourceData],
    air_quality: AirQualityData | None = None,
    limits: DroneLimit | None = None,
) -> DecisionResult:
    """Aggregate *sources* and return a ``DecisionResult``."""

    active_limits = limits if limits is not None else settings
    sources, excluded = usable_sources(sources)

    def _weighted_pairs(
        getter: Callable[[WeatherSourceData], float | None],
    ) -> list[tuple[float | None, float]]:
        return [(getter(source), source.reliability_weight) for source in sources]

    avg_wind = _weighted_avg_optional(_weighted_pairs(lambda s: s.wind_speed_knots))
    avg_temp = _weighted_avg_optional(_weighted_pairs(lambda s: s.temperature_c))
    avg_humidity = _weighted_avg_optional(_weighted_pairs(lambda s: s.humidity_pct))
    avg_visibility = _weighted_avg_optional(_weighted_pairs(lambda s: s.visibility_km))
    # For precipitation use worst (max) across sources
    precipitation = [s.precipitation_level for s in sources if s.precipitation_level is not None]
    max_precip = max(precipitation) if precipitation else None
    avg_cloud_base = _weighted_avg_optional(_weighted_pairs(lambda s: s.cloud_base_ft))
    avg_cloud_cover = _weighted_avg_optional(
        _weighted_pairs(lambda s: s.cloud_cover_pct)
    )
    avg_cloud_ceiling = _weighted_avg_optional(
        _weighted_pairs(lambda s: s.cloud_ceiling_ft)
    )

    params: list[ParameterResult] = [
        ParameterResult(
            name="Rüzgar hızı",
            value=f"{avg_wind:.1f} knot" if avg_wind is not None else "N/A",
            decision=_eval_wind(avg_wind, active_limits),
        ),
        ParameterResult(
            name="Görüş mesafesi",
            value=f"{avg_visibility:.1f} km" if avg_visibility is not None else "N/A",
            decision=_eval_visibility(avg_visibility, active_limits),
        ),
        ParameterResult(
            name="Yağış",
            value=_PRECIP_LABELS.get(max_precip, "N/A"),
            decision=_eval_precipitation(max_precip),
        ),
        ParameterResult(
            name="Sıcaklık",
            value=f"{avg_temp:.1f} °C" if avg_temp is not None else "N/A",
            decision=_eval_temperature(avg_temp, active_limits),
        ),
        ParameterResult(
            name="Nem",
            value=f"{avg_humidity:.0f}%" if avg_humidity is not None else "N/A",
            decision=_eval_humidity(avg_humidity, active_limits),
        ),
        ParameterResult(
            name="Bulut tabanı",
            value=(f"{avg_cloud_base:.0f} ft" if avg_cloud_base is not None else "N/A"),
            decision=_eval_cloud_base(avg_cloud_base, active_limits),
        ),
        ParameterResult(
            name="Bulut tavanı",
            value=(
                f"{avg_cloud_ceiling:.0f} ft"
                if avg_cloud_ceiling is not None
                else "N/A"
            ),
            decision=_eval_cloud_ceiling(avg_cloud_ceiling, active_limits),
        ),
    ]
    if sources and any(s.ceiling_unlimited for s in sources):
        for parameter in params:
            if parameter.value == "N/A" and (
                parameter.name == "Bulut tavanı"
                or (parameter.name == "Bulut tabanı" and any(s.clouds_clear for s in sources))
            ):
                parameter.value = "Tavan sınırlaması bildirilmedi"
                parameter.decision = UYGUN

    evaluators = {
        "Rüzgar hızı": (lambda s: s.wind_speed_knots, lambda v: _eval_wind(v, active_limits)),
        "Görüş mesafesi": (lambda s: s.visibility_km, lambda v: _eval_visibility(v, active_limits)),
        "Sıcaklık": (lambda s: s.temperature_c, lambda v: _eval_temperature(v, active_limits)),
        "Nem": (lambda s: s.humidity_pct, lambda v: _eval_humidity(v, active_limits)),
        "Bulut tabanı": (lambda s: s.cloud_base_ft, lambda v: _eval_cloud_base(v, active_limits)),
        "Bulut tavanı": (lambda s: s.cloud_ceiling_ft, lambda v: _eval_cloud_ceiling(v, active_limits)),
    }
    evidence = {}
    for parameter in params:
        if parameter.name not in evaluators:
            continue
        getter, evaluate = evaluators[parameter.name]
        candidates = [(evaluate(getter(s)), s) for s in sources if getter(s) is not None]
        if candidates:
            worst, source = max(candidates, key=lambda item: _SCORE[item[0]])
            if _SCORE[worst] > _SCORE[parameter.decision]:
                parameter.decision = worst
                evidence[parameter.name] = f"{source.source} ({getter(source):g})"
    if avg_cloud_cover is not None:
        params.insert(
            5,
            ParameterResult(
                name="Bulutluluk",
                value=f"{avg_cloud_cover:.1f}%",
                decision=RISKLI if avg_cloud_cover > (
                    limits.cloud_cover_ok_max_pct if limits is not None else 50
                ) else UYGUN,
            ),
        )
    if air_quality is not None:
        params.append(
            ParameterResult(
                name="Hava kalitesi",
                value=f"Avrupa AQI {air_quality.aqi_score}",
                decision=_eval_aqi(air_quality.aqi_score),
            )
        )

    cover_limit = limits.cloud_cover_ok_max_pct if limits is not None else 50
    ranges = {
        "Rüzgar hızı": (
            f"< {active_limits.wind_ok_max_knots:g} knot",
            f"{active_limits.wind_ok_max_knots:g}–{active_limits.wind_risky_max_knots:g} knot (dahil)",
            f"> {active_limits.wind_risky_max_knots:g} knot",
        ),
        "Görüş mesafesi": (
            f"> {active_limits.visibility_ok_min_km:g} km",
            f"{active_limits.visibility_risky_min_km:g}–{active_limits.visibility_ok_min_km:g} km (dahil)",
            f"< {active_limits.visibility_risky_min_km:g} km",
        ),
        "Yağış": ("Yağış yok", "Hafif yağış", "Orta/şiddetli yağış"),
        "Sıcaklık": (
            f"{active_limits.temp_ok_min_c:g}–{active_limits.temp_ok_max_c:g} °C (dahil)",
            f"Uygun aralık dışında; {active_limits.temp_risky_min_c:g}–{active_limits.temp_risky_max_c:g} °C içinde",
            f"< {active_limits.temp_risky_min_c:g} veya > {active_limits.temp_risky_max_c:g} °C",
        ),
        "Nem": (
            f"< %{active_limits.humidity_ok_max_pct:g}",
            f"%{active_limits.humidity_ok_max_pct:g}–%{active_limits.humidity_risky_max_pct:g} (dahil)",
            f"> %{active_limits.humidity_risky_max_pct:g}",
        ),
        "Bulutluluk": (
            f"≤ %{cover_limit:g}",
            f"> %{cover_limit:g}; tek başına UYGUN DEĞİL yapmaz",
            "",
        ),
        "Hava kalitesi": ("AQI ≤ 100", "100 < AQI ≤ 150", "AQI > 150"),
    }
    for name, ok_min, risky_min in (
        ("Bulut tabanı", active_limits.cloud_base_ok_min_ft, active_limits.cloud_base_risky_min_ft),
        ("Bulut tavanı", active_limits.cloud_ceiling_ok_min_ft, active_limits.cloud_ceiling_risky_min_ft),
    ):
        ranges[name] = (
            f"> {ok_min:g} ft",
            f"{risky_min:g}–{ok_min:g} ft (dahil)",
            f"< {risky_min:g} ft",
        )
    for parameter in params:
        if parameter.value == "N/A":
            parameter.explanation = "Kullanılabilir veri yok; uçuş uygunluğu doğrulanamaz."
        elif parameter.value == "Tavan sınırlaması bildirilmedi":
            parameter.explanation = "Kaynak tavan sınırlaması bildirmiyor; model kaynaklarında bu ölçüm değil, tahmindir."
        else:
            parameter.explanation = ranges[parameter.name][_SCORE[parameter.decision]]
        if parameter.name in evidence:
            parameter.explanation += f" Ortalama riski azaltmadı; {evidence[parameter.name]} daha ağır koşul bildiriyor."

    missing = [p.name for p in params if p.decision == VERI_YETERSIZ]
    known_score = max((_SCORE[p.decision] for p in params if p.decision != VERI_YETERSIZ), default=0)
    known_decision = _LABEL[known_score]
    overall = VERI_YETERSIZ if missing and known_decision != UYGUN_DEGIL else known_decision

    # Build detail message
    bad = [p for p in params if p.decision not in {UYGUN, VERI_YETERSIZ}]
    if not bad:
        detail = "Tüm parametreler uygun sınırlar içinde."
    else:
        lines = []
        for p in bad:
            label = "⚠️ Riskli" if p.decision == RISKLI else "❌ Uygun Değil"
            lines.append(f"{p.name}: {p.value} – {label} ({p.explanation})")
        detail = "\n".join(lines)
    if missing:
        detail = f"Veri yetersiz: {', '.join(missing)}. Uçuş uygunluğu doğrulanamaz.\n" + (
            detail if bad else "Mevcut parametrelerde sınır aşımı görülmedi; bu bir uçuş onayı değildir."
        )
    if excluded:
        detail += "\nDeğerlendirmeye katılmayan kaynaklar: " + "; ".join(excluded)

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

    score_map = {UYGUN: 100, RISKLI: 50, UYGUN_DEGIL: 0, VERI_YETERSIZ: 0}
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
        avg_precip_level=float(max_precip) if max_precip is not None else None,
        avg_cloud_base_ft=avg_cloud_base,
        avg_cloud_ceiling_ft=avg_cloud_ceiling,
        avg_cloud_cover_pct=avg_cloud_cover,
        confidence_score=confidence_score,
        aqi_score=air_quality.aqi_score if air_quality is not None else None,
        pm25=air_quality.pm25 if air_quality is not None else None,
        pm10=air_quality.pm10 if air_quality is not None else None,
        missing_parameters=missing,
        known_decision=known_decision,
        excluded_sources=excluded,
    )


def compute_wind_components(
    wind_direction_deg: float,
    wind_speed_knots: float,
    runway_heading_deg: float,
) -> dict:
    """Calculate headwind, crosswind, and tailwind components.

    Args:
        wind_direction_deg: Meteorological wind direction (where wind comes FROM).
        wind_speed_knots: Wind speed in knots.
        runway_heading_deg: True heading of the runway (direction aircraft points).

    Returns:
        Dict with headwind_kt, tailwind_kt, crosswind_kt, crosswind_side.
        Positive headwind = into the wind (good).
        Positive crosswind absolute value used for limits.
        Positive tailwind = with the wind (bad).
    """
    angle_diff = math.radians(wind_direction_deg - runway_heading_deg)
    headwind = wind_speed_knots * math.cos(angle_diff)
    crosswind = wind_speed_knots * math.sin(angle_diff)

    crosswind_side = "R" if crosswind > 0.05 else ("L" if crosswind < -0.05 else "")

    if headwind >= 0:
        return {
            "headwind_kt": round(headwind, 1),
            "tailwind_kt": 0.0,
            "crosswind_kt": round(abs(crosswind), 1),
            "crosswind_side": crosswind_side,
        }
    else:
        return {
            "headwind_kt": 0.0,
            "tailwind_kt": round(abs(headwind), 1),
            "crosswind_kt": round(abs(crosswind), 1),
            "crosswind_side": crosswind_side,
        }

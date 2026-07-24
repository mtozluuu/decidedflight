"""Decision engine: evaluates normalised weather data against drone limits.

Each parameter is scored as:
  0 – UYGUN       (suitable)
  1 – RISKLI      (risky)
  2 – UYGUN_DEGIL (unsuitable)

The overall decision is the worst score across all averaged parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

from decideflight.config import settings
from decideflight.services.weather_fetcher import (
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


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _avg_optional(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    return _avg(valid) if valid else None


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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def make_decision(sources: list[WeatherSourceData]) -> DecisionResult:
    """Aggregate *sources* and return a ``DecisionResult``."""
    avg_wind = _avg([s.wind_speed_knots for s in sources])
    avg_temp = _avg([s.temperature_c for s in sources])
    avg_humidity = _avg([s.humidity_pct for s in sources])
    avg_visibility = _avg([s.visibility_km for s in sources])
    # For precipitation use worst (max) across sources
    max_precip = max(s.precipitation_level for s in sources)
    avg_cloud_base = _avg_optional([s.cloud_base_ft for s in sources])
    avg_cloud_ceiling = _avg_optional([s.cloud_ceiling_ft for s in sources])

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
    )

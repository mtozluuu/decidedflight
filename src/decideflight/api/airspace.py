"""Airspace map layer endpoint."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from decideflight.services.airspace import (
    AIRSPACE_TYPES,
    DEFAULT_TYPES,
    OPENAIP_AIRSPACE_URL,
    AirspaceUnavailableError,
    filter_features,
    load_airspaces,
)

router = APIRouter(prefix="/api/v1/airspace", tags=["airspace"])

DISCLAIMER = (
    "Hava sahası sınırları openAIP verisinden gelir ve yalnızca bilgilendirme amaçlıdır. "
    "Resmî DHMİ AIP, AIRAC değişiklikleri ve NOTAM ile doğrulanmadan operasyonel "
    "karar için tek kaynak olarak kullanılmamalıdır."
)


@router.get("")
async def get_airspaces(
    categories: str = Query(",".join(DEFAULT_TYPES), description="Virgülle ayrılmış: TMA,CTR,FIR,R,D,P,..."),
) -> dict:
    """Return Turkish airspace polygons as GeoJSON with vertical limits."""
    known = {code for code, _ in AIRSPACE_TYPES.values()}
    requested = {item.strip().upper() for item in categories.split(",") if item.strip()}
    invalid = requested - known
    if invalid:
        raise HTTPException(status_code=422, detail=f"Bilinmeyen kategori: {', '.join(sorted(invalid))}")
    try:
        features, fetched_at = await load_airspaces()
    except AirspaceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "type": "FeatureCollection",
        "features": filter_features(features, requested),
        "source": OPENAIP_AIRSPACE_URL,
        "fetched_at": datetime.fromtimestamp(fetched_at, tz=timezone.utc).isoformat(),
        "disclaimer": DISCLAIMER,
    }

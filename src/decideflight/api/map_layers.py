"""Wind and cloud grid endpoint for the map overlay."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from decideflight.services.map_layers import (
    LEVELS,
    MapLayerUnavailableError,
    flight_categories,
    load_layer,
    sigmets,
    turkish_airports,
)

router = APIRouter(prefix="/api/v1/map-layers", tags=["map-layers"])

DISCLAIMER = (
    "Rüzgâr ve bulut değerleri Open-Meteo model tahminidir, ölçüm değildir. "
    "Resmî havacılık meteoroloji brifingi, METAR, TAF ve SIGMET yerine geçmez."
)


@router.get("/airports")
def airports() -> dict:
    """Turkish airports with ICAO codes for the METAR/TAF map layer."""
    return {"airports": turkish_airports()}


@router.get("/wind-cloud")
async def wind_cloud(level: str = Query("surface", description="surface, 850, 700 veya 500")) -> dict:
    if level not in LEVELS:
        raise HTTPException(status_code=422, detail="Geçersiz seviye. surface, 850, 700 veya 500 kullanın.")
    try:
        data = await load_layer(level)
    except MapLayerUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {**data, "source": "Open-Meteo", "disclaimer": DISCLAIMER}


@router.get("/airports/status")
async def airport_status() -> dict:
    """Latest METAR flight category (VFR/MVFR/IFR/LIFR) per airport."""
    try:
        return {"stations": await flight_categories(), "source": "NOAA Aviation Weather Center"}
    except MapLayerUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/sigmet")
async def sigmet_list() -> dict:
    """Active international SIGMETs near Türkiye."""
    try:
        items = await sigmets()
    except MapLayerUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "sigmets": items,
        "source": "NOAA Aviation Weather Center",
        "disclaimer": "Bilgilendirme amaçlıdır; resmî SIGMET/AIRMET için DHMİ/MGM kaynaklarını kullanın.",
    }

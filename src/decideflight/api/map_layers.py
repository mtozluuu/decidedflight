"""Wind and cloud grid endpoint for the map overlay."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from decideflight.services.map_layers import LEVELS, MapLayerUnavailableError, load_layer, turkish_airports

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

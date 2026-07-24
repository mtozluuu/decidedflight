"""FastAPI application entrypoint."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from decideflight.api.health import router as health_router
from decideflight.api.weather import router as weather_router
from decideflight.database import init_db

# Ensure all models are imported so SQLAlchemy registers their tables
import decideflight.models  # noqa: F401

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="DecideFlight API",
    description="AI-powered flight weather decision system",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(weather_router)

if os.path.isdir(_STATIC_DIR):
    app.mount(
        "/static",
        StaticFiles(directory=_STATIC_DIR),
        name="static",
    )


@app.get("/", include_in_schema=False)
def root():
    """Serve the web UI or fall back to a JSON status message."""
    index_path = os.path.join(_STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"message": "DecideFlight API is running"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Serve favicon for browsers requesting /favicon.ico directly."""
    icon_path = os.path.join(_STATIC_DIR, "decideflight-icon.svg")
    return FileResponse(icon_path, media_type="image/svg+xml")


@app.get("/apple-touch-icon.png", include_in_schema=False)
@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
def apple_touch_icon():
    """Serve Apple touch icon fallback requests."""
    icon_path = os.path.join(_STATIC_DIR, "decideflight-icon.svg")
    return FileResponse(icon_path, media_type="image/svg+xml")

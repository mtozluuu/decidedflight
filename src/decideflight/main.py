"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from decideflight.api.health import router as health_router
from decideflight.api.weather import router as weather_router
from decideflight.database import init_db

# Ensure all models are imported so SQLAlchemy registers their tables
import decideflight.models  # noqa: F401


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


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "DecideFlight API is running"}

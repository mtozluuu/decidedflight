"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from decideflight.api.health import router as health_router
from decideflight.database import init_db


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


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "DecideFlight API is running"}

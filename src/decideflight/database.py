"""Database setup for DecideFlight."""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from decideflight.config import settings


class Base(DeclarativeBase):
    """Base class for ORM models."""


engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)

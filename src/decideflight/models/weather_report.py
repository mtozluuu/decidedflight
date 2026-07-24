"""Weather report database model."""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from decideflight.database import Base


class WeatherReport(Base):
    """Persisted drone flight weather report."""

    __tablename__ = "weather_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    location: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    decision_detail: Mapped[str] = mapped_column(Text, nullable=False)
    sources_data: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="JSON-encoded list of WeatherSourceData dicts",
    )
    ai_analysis_data: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="JSON-encoded AI analysis extra fields (confidence, detailed_analysis, etc.)",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

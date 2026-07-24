"""AI decision engine context builder.

Fetches recent user feedback from the database and builds a context string
that can be injected into a GPT system prompt so the model learns from past
decisions and real-world outcomes.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from decideflight.models.feedback import Feedback
from decideflight.models.weather_report import WeatherReport

MAX_FEEDBACK_CONTEXT = 20


def build_feedback_context(db: Session) -> str:
    """Return the last *MAX_FEEDBACK_CONTEXT* feedback entries as a prompt.

    The returned string is suitable for prepending to a GPT system prompt so
    that the model is aware of historical decisions and whether they were
    correct according to the user.

    Returns an empty string when no feedback has been recorded yet.
    """
    rows = (
        db.query(Feedback, WeatherReport)
        .join(WeatherReport, Feedback.report_id == WeatherReport.id)
        .order_by(Feedback.created_at.desc())
        .limit(MAX_FEEDBACK_CONTEXT)
        .all()
    )
    if not rows:
        return ""

    lines = ["Geçmiş kararlar ve gerçek sonuçlar:"]
    for fb, report in reversed(rows):
        verdict = "DOĞRU" if fb.correct else "YANLIŞ"
        line = f"- Karar: {report.decision} → Kullanıcı: {verdict}"
        if fb.user_comment:
            line += f" (yorum: {fb.user_comment})"
        lines.append(line)
    return "\n".join(lines)

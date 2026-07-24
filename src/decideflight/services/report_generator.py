"""PDF report generator using ReportLab.

Generates a single-page (multi-section) PDF document that includes:
  - Report header (title, date, location)
  - Weather data table (one row per source)
  - Aggregated averages
  - Decision (large, colour-coded)
  - Per-parameter detail
  - Drone limits reference table
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from decideflight.config import settings
from decideflight.services.decision_engine import (
    DecisionResult,
    UYGUN,
    RISKLI,
    UYGUN_DEGIL,
)
from decideflight.services.weather_fetcher import WeatherSourceData

# Imported lazily to avoid circular-import risk at module load time;
# used only for isinstance checks inside generate_pdf.
_AIDecisionResult = None


def _get_ai_result_class():  # type: ignore[return]
    global _AIDecisionResult
    if _AIDecisionResult is None:
        from decideflight.services.ai_decision_engine import (
            AIDecisionResult,
        )

        _AIDecisionResult = AIDecisionResult
    return _AIDecisionResult


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
_DECISION_COLOURS = {
    UYGUN: colors.HexColor("#1a7a1a"),
    RISKLI: colors.HexColor("#c47a00"),
    UYGUN_DEGIL: colors.HexColor("#c0392b"),
}

_PARAM_BG = {
    UYGUN: colors.HexColor("#d4edda"),
    RISKLI: colors.HexColor("#fff3cd"),
    UYGUN_DEGIL: colors.HexColor("#f8d7da"),
}

_HEADER_BG = colors.HexColor("#2c3e50")
_ALT_ROW_BG = colors.HexColor("#f2f2f2")

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------

_STYLES = getSampleStyleSheet()

_TITLE_STYLE = ParagraphStyle(
    "ReportTitle",
    parent=_STYLES["Title"],
    fontSize=22,
    textColor=colors.HexColor("#2c3e50"),
    spaceAfter=6,
)

_SUBTITLE_STYLE = ParagraphStyle(
    "Subtitle",
    parent=_STYLES["Normal"],
    fontSize=11,
    textColor=colors.HexColor("#7f8c8d"),
    spaceAfter=4,
    alignment=TA_CENTER,
)

_SECTION_STYLE = ParagraphStyle(
    "Section",
    parent=_STYLES["Heading2"],
    fontSize=13,
    textColor=colors.HexColor("#2c3e50"),
    spaceBefore=12,
    spaceAfter=6,
)

_BODY_STYLE = ParagraphStyle(
    "Body",
    parent=_STYLES["Normal"],
    fontSize=10,
    leading=14,
    alignment=TA_LEFT,
)

_DECISION_LABEL_STYLE = ParagraphStyle(
    "DecisionLabel",
    parent=_STYLES["Normal"],
    fontSize=28,
    fontName="Helvetica-Bold",
    alignment=TA_CENTER,
    spaceAfter=6,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _precip_label(level: float) -> str:
    mapping = {0: "Yok", 1: "Hafif", 2: "Orta/Şiddetli"}
    return mapping.get(int(level), "Bilinmiyor")


def _optional_str(val: float | None, unit: str = "ft") -> str:
    if val is None:
        return "N/A"
    return f"{val:.0f} {unit}"


def _header_row() -> list[str]:
    return [
        "Kaynak",
        "Rüzgar (kt)",
        "Sıcaklık (°C)",
        "Nem (%)",
        "Görüş (km)",
        "Yağış",
        "Bulut Tabanı (ft)",
        "Bulut Tavanı (ft)",
    ]


def _source_row(s: WeatherSourceData) -> list[str]:
    return [
        s.source,
        f"{s.wind_speed_knots:.1f}",
        f"{s.temperature_c:.1f}",
        f"{s.humidity_pct:.0f}",
        f"{s.visibility_km:.1f}",
        _precip_label(s.precipitation_level),
        _optional_str(s.cloud_base_ft),
        _optional_str(s.cloud_ceiling_ft),
    ]


def _avg_row(dr: DecisionResult) -> list[str]:
    return [
        "ORTALAMA",
        f"{dr.avg_wind_knots:.1f}",
        f"{dr.avg_temp_c:.1f}",
        f"{dr.avg_humidity_pct:.0f}",
        f"{dr.avg_visibility_km:.1f}",
        _precip_label(dr.avg_precip_level),
        _optional_str(dr.avg_cloud_base_ft),
        _optional_str(dr.avg_cloud_ceiling_ft),
    ]


def _limits_rows() -> list[list[str]]:
    return [
        ["Parametre", "✅ Uygun", "⚠️ Riskli", "❌ Uygun Değil"],
        [
            "Rüzgar hızı",
            f"< {settings.wind_ok_max_knots:.0f} knot",
            f"{settings.wind_ok_max_knots:.0f}–"
            f"{settings.wind_risky_max_knots:.0f} knot",
            f"> {settings.wind_risky_max_knots:.0f} knot",
        ],
        [
            "Görüş mesafesi",
            f"> {settings.visibility_ok_min_km:.0f} km",
            f"{settings.visibility_risky_min_km:.0f}–"
            f"{settings.visibility_ok_min_km:.0f} km",
            f"< {settings.visibility_risky_min_km:.0f} km",
        ],
        ["Yağış", "Yok", "Hafif", "Orta/Şiddetli"],
        [
            "Sıcaklık",
            f"{settings.temp_ok_min_c:.0f}–{settings.temp_ok_max_c:.0f} °C",
            f"{settings.temp_risky_min_c:.0f}–{settings.temp_ok_min_c:.0f} °C veya "
            f"{settings.temp_ok_max_c:.0f}–{settings.temp_risky_max_c:.0f} °C",
            f"< {settings.temp_risky_min_c:.0f} °C veya "
            f"> {settings.temp_risky_max_c:.0f} °C",
        ],
        [
            "Nem",
            f"< %{settings.humidity_ok_max_pct:.0f}",
            f"%{settings.humidity_ok_max_pct:.0f}–"
            f"{settings.humidity_risky_max_pct:.0f}",
            f"> %{settings.humidity_risky_max_pct:.0f}",
        ],
        [
            "Bulut tabanı",
            f"> {settings.cloud_base_ok_min_ft:.0f} ft",
            f"{settings.cloud_base_risky_min_ft:.0f}–"
            f"{settings.cloud_base_ok_min_ft:.0f} ft",
            f"< {settings.cloud_base_risky_min_ft:.0f} ft",
        ],
        [
            "Bulut tavanı",
            f"> {settings.cloud_ceiling_ok_min_ft:.0f} ft",
            f"{settings.cloud_ceiling_risky_min_ft:.0f}–"
            f"{settings.cloud_ceiling_ok_min_ft:.0f} ft",
            f"< {settings.cloud_ceiling_risky_min_ft:.0f} ft",
        ],
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_pdf(
    location: str,
    lat: float,
    lon: float,
    sources: list[WeatherSourceData],
    decision_result: DecisionResult,
    created_at: datetime | None = None,
) -> bytes:
    """Return PDF bytes for the given weather report data."""
    if created_at is None:
        created_at = datetime.now(timezone.utc)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    story: list[Any] = []

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    story.append(Paragraph("✈ DecideFlight — Drone Uçuş Raporu", _TITLE_STYLE))
    story.append(
        Paragraph(
            f"Konum: <b>{location}</b> ({lat:.4f}, {lon:.4f})",
            _SUBTITLE_STYLE,
        )
    )
    story.append(
        Paragraph(
            f"Oluşturulma: {created_at.strftime('%Y-%m-%d %H:%M UTC')}",
            _SUBTITLE_STYLE,
        )
    )
    story.append(Spacer(1, 0.4 * cm))

    # ------------------------------------------------------------------
    # Decision box
    # ------------------------------------------------------------------
    dec_colour = _DECISION_COLOURS.get(decision_result.decision, colors.grey)
    decision_text = {
        UYGUN: "✅ UYGUN",
        RISKLI: "⚠️ RİSKLİ",
        UYGUN_DEGIL: "❌ UYGUN DEĞİL",
    }.get(decision_result.decision, decision_result.decision)

    dec_label_style = ParagraphStyle(
        "DecisionLabelColoured",
        parent=_DECISION_LABEL_STYLE,
        textColor=dec_colour,
    )
    dec_label = Paragraph(decision_text, dec_label_style)
    # Use a single-cell coloured table as a "box"
    dec_table = Table([[dec_label]], colWidths=[16 * cm])
    dec_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 2, dec_colour),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(dec_table)
    story.append(Spacer(1, 0.3 * cm))

    # Detail lines
    if decision_result.detail:
        for line in decision_result.detail.split("\n"):
            if line.strip():
                story.append(Paragraph(line, _BODY_STYLE))
    story.append(Spacer(1, 0.4 * cm))

    # ------------------------------------------------------------------
    # Weather data table
    # ------------------------------------------------------------------
    story.append(Paragraph("Kaynak Verileri", _SECTION_STYLE))

    table_data = [_header_row()]
    for src in sources:
        table_data.append(_source_row(src))
    table_data.append(_avg_row(decision_result))

    col_widths = [
        3.2 * cm,
        2 * cm,
        2.2 * cm,
        1.8 * cm,
        2 * cm,
        2 * cm,
        2.5 * cm,
        2.5 * cm,
    ]
    data_table = Table(table_data, colWidths=col_widths, repeatRows=1)

    ts = [
        ("BACKGROUND", (0, 0), (-1, 0), _HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        # Average row highlight
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#d6eaf8")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
    ]
    # Alternate row colours for source rows
    for i in range(1, len(table_data) - 1):
        if i % 2 == 0:
            ts.append(("BACKGROUND", (0, i), (-1, i), _ALT_ROW_BG))

    data_table.setStyle(TableStyle(ts))
    story.append(data_table)
    story.append(Spacer(1, 0.4 * cm))

    # ------------------------------------------------------------------
    # Per-parameter assessment
    # ------------------------------------------------------------------
    story.append(Paragraph("Parametre Değerlendirmesi", _SECTION_STYLE))

    param_data = [["Parametre", "Değer", "Durum"]]
    for p in decision_result.parameters:
        label = {
            UYGUN: "✅ Uygun",
            RISKLI: "⚠️ Riskli",
            UYGUN_DEGIL: "❌ Uygun Değil",
        }.get(p.decision, p.decision)
        param_data.append([p.name, p.value, label])

    param_table = Table(param_data, colWidths=[5 * cm, 4 * cm, 4 * cm])
    pts = [
        ("BACKGROUND", (0, 0), (-1, 0), _HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for i, row in enumerate(param_data[1:], start=1):
        dec = decision_result.parameters[i - 1].decision
        pts.append(("BACKGROUND", (0, i), (-1, i), _PARAM_BG.get(dec, colors.white)))

    param_table.setStyle(TableStyle(pts))
    story.append(param_table)
    story.append(Spacer(1, 0.4 * cm))

    # ------------------------------------------------------------------
    # Drone limits reference
    # ------------------------------------------------------------------
    story.append(Paragraph("Drone Uçuş Limitleri", _SECTION_STYLE))

    limits_data = _limits_rows()
    limits_table = Table(limits_data, colWidths=[4 * cm, 4 * cm, 4 * cm, 4 * cm])
    lts = [
        ("BACKGROUND", (0, 0), (-1, 0), _HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        # Colour limit header cells
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#1a7a1a")),
        ("BACKGROUND", (2, 0), (2, 0), colors.HexColor("#c47a00")),
        ("BACKGROUND", (3, 0), (3, 0), colors.HexColor("#c0392b")),
    ]
    for i in range(1, len(limits_data)):
        if i % 2 == 0:
            lts.append(("BACKGROUND", (0, i), (0, i), _ALT_ROW_BG))

    limits_table.setStyle(TableStyle(lts))
    story.append(limits_table)

    # ------------------------------------------------------------------
    # AI Analysis section (only when GPT-4o result is available)
    # ------------------------------------------------------------------
    AIDecisionResult = _get_ai_result_class()
    if isinstance(decision_result, AIDecisionResult):
        ai = decision_result

        story.append(Paragraph("AI Analizi (GPT-4o)", _SECTION_STYLE))

        # Confidence score
        conf_style = ParagraphStyle(
            "ConfScore",
            parent=_BODY_STYLE,
            fontSize=10,
            textColor=colors.HexColor("#2c3e50"),
        )
        story.append(Paragraph(f"<b>Güven Skoru:</b> %{ai.confidence}", conf_style))
        story.append(Spacer(1, 0.2 * cm))

        # Detailed analysis
        if ai.detailed_analysis:
            story.append(Paragraph("<b>Detaylı Analiz:</b>", _BODY_STYLE))
            story.append(Paragraph(ai.detailed_analysis, _BODY_STYLE))
            story.append(Spacer(1, 0.2 * cm))

        # Risk factors
        if ai.risk_factors:
            story.append(Paragraph("<b>Risk Faktörleri:</b>", _BODY_STYLE))
            for rf in ai.risk_factors:
                story.append(Paragraph(f"• {rf}", _BODY_STYLE))
            story.append(Spacer(1, 0.2 * cm))

        # Recommendations
        if ai.recommendations:
            story.append(Paragraph("<b>Öneriler:</b>", _BODY_STYLE))
            for rec in ai.recommendations:
                story.append(Paragraph(f"• {rec}", _BODY_STYLE))
            story.append(Spacer(1, 0.2 * cm))

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    doc.build(story)
    return buffer.getvalue()

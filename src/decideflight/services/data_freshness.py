"""UTC data times, separate from retrieval times and flight decisions."""

from datetime import datetime, timedelta, timezone
import re

from decideflight.services.weather_fetcher import WeatherSourceData


def _day_time(day: int, hour: int, minute: int, reference: datetime) -> datetime | None:
    candidates = []
    for offset in (-1, 0, 1):
        month_index = reference.year * 12 + reference.month - 1 + offset
        year, month = divmod(month_index, 12)
        try:
            value = datetime(year, month + 1, day, tzinfo=timezone.utc)
            if hour > 24 or minute > 59 or (hour == 24 and minute):
                continue
            candidates.append(value + timedelta(hours=hour, minutes=minute))
        except ValueError:
            continue
    return min(candidates, key=lambda value: abs(value - reference)) if candidates else None


def observation_timing(
    text: str | None, now: datetime, *, taf: bool = False
) -> dict:
    result = {
        "issued_at": None, "valid_from": None, "valid_until": None,
        "status": "unknown", "message": "Veri zamanı doğrulanamadı.",
    }
    if not text:
        result["message"] = "Rasat bulunamadı."
        return result
    issue = re.match(r"^(?:METAR|TAF)(?: (?:AMD|COR))? [A-Z]{4} (\d{2})(\d{2})(\d{2})Z\b", text)
    if not issue:
        return result
    issued = _day_time(*map(int, issue.groups()), now)
    if issued is None:
        return result
    result["issued_at"] = issued.isoformat()
    age = (now - issued).total_seconds() / 60
    if age < -5:
        result["message"] = "Rasat zamanı gelecekte; doğrulanamadı."
        return result
    if not taf:
        result.update(
            status="stale" if age > 60 else "current",
            message=f"METAR gözlem yaşı: {max(0, int(age))} dakika."
            + (" Eski veri: 60 dakikayı aşıyor." if age > 60 else ""),
        )
        return result
    validity = re.search(r"\b(\d{2})(\d{2})/(\d{2})(\d{2})\b", text[issue.end():])
    if not validity or re.search(r"\b(?:CNL|NIL)\b", text):
        result["message"] = "TAF geçerliliği doğrulanamadı veya tahmin iptal edilmiş."
        return result
    start_day, start_hour, end_day, end_hour = map(int, validity.groups())
    start = _day_time(start_day, start_hour, 0, issued)
    end = _day_time(end_day, end_hour, 0, start) if start else None
    if start is None or end is None or not start < end <= start + timedelta(hours=72):
        return result
    state = "expired" if now >= end else "upcoming" if now < start else "valid"
    result.update(
        valid_from=start.isoformat(), valid_until=end.isoformat(), status=state,
        message={"expired": "TAF süresi dolmuş.", "upcoming": "TAF henüz geçerli değil.",
                 "valid": "TAF geçerlilik aralığı içinde."}[state],
    )
    return result


def source_timing(source: WeatherSourceData, now: datetime) -> dict:
    raw = source.raw
    value = None
    kind = "observation"
    if source.source == "Open-Meteo":
        value = raw.get("time")
        kind = "forecast"
    elif source.source == "OpenWeatherMap":
        value = raw.get("dt")
    elif source.source == "WeatherAPI":
        value = raw.get("last_updated_epoch")
    elif source.source == "Windy":
        times = raw.get("ts")
        value = raw.get("selected_time")
        if value is None:
            value = times[0] / 1000 if times and isinstance(times[0], (int, float)) else None
        kind = "forecast"
    elif "NOAA" in source.source:
        value = raw.get("obsTime")
    elif "CheckWX" in source.source:
        value = raw.get("observed")
    result = {"data_time": None, "time_kind": kind, "freshness": "unknown",
              "freshness_message": "Veri zamanı doğrulanamadı; güncellik bilinmiyor."}
    try:
        if isinstance(value, (int, float)):
            timestamp = datetime.fromtimestamp(value, timezone.utc)
        elif isinstance(value, str):
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                if kind != "forecast":
                    return result
                timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            return result
    except (ValueError, OverflowError, OSError):
        return result
    result["data_time"] = timestamp.isoformat()
    age = (now - timestamp).total_seconds() / 60
    if kind == "forecast":
        result["freshness_message"] = "Tahminin hedef zamanı; model üretim zamanı bilinmiyor."
        if age > 120:
            result["freshness"] = "stale"
            result["freshness_message"] += " Hedef zamanı 120 dakikadan eski; değerlendirmeye katılmaz."
        elif age < -120:
            result["freshness"] = "stale"
            result["freshness_message"] += " Hedef zamanı anlık değerlendirmeden 120 dakikadan fazla ileride; değerlendirmeye katılmaz."
    elif age < -5:
        result["freshness_message"] = "Veri zamanı gelecekte; güncellik doğrulanamadı."
    else:
        result["freshness_message"] = f"Gözlem yaşı: {max(0, int(age))} dakika."
        if "METAR" in source.source:
            result["freshness"] = "stale" if age > 60 else "current"
            if age > 60:
                result["freshness_message"] += " Eski veri: 60 dakikayı aşıyor."
        else:
            result["freshness"] = "stale" if age > 120 else "current"
            result["freshness_message"] += " Güncellik sınırı: 120 dakika."
    return result

# DecideFlight

DecideFlight is an AI-powered flight weather decision system. It combines data from multiple weather providers and applies configurable rules/ML-assisted evaluation to determine whether flight conditions are suitable.

## Features

- FastAPI-based REST API foundation
- Data timing is separate from retrieval time: METAR older than 60 minutes is flagged as stale, and TAF is checked against its own UTC validity interval. Provider timestamps appear on source cards/status checks when available. Forecast target times are not model issuance times; missing or unverified timestamps are never labeled current. MGM Airgram image production/validity still requires reading the original image. These warnings are informational and do not automatically change the flight decision.
- Config-driven environment management
- SQLAlchemy database layer ready for PostgreSQL
- Initial weather data model
- Weather reports use Open-Meteo, with optional OpenWeatherMap, WeatherAPI, and Windy sources when configured. Nearby METAR observations from NOAA Aviation Weather are included when available; CheckWX is optional.
- Air quality and wind trends use Open-Meteo. The web UI's **Veri Kaynakları** tab checks provider availability for selected coordinates.
- Reports show cloud cover (%) from Open-Meteo, OpenWeatherMap and WeatherAPI, weighted over available measurements. Cover up to 50% is suitable; above 50% is risky, never unsuitable on its own. Missing total cloud cover (including Windy's separate cloud layers and METAR observations) is excluded rather than treated as clear sky. Cloud base, ceiling and visibility remain separate checks.
- The Settings tab saves weather limits in the browser for new point reports and regional/grid analysis, including cloud cover (0–100%, default 50%). Values above the cloud-cover maximum are risky, not unsuitable. Point reports persist their applied limits and display them; PDF/report reloads reuse that snapshot rather than current browser settings. Requests without custom limits retain server defaults; old reports remain compatible.
- Point analysis shows the selected airport's latest raw METAR/TAF from the public MGM Hezarfen Rasat page. Enter an ICAO code or select a nearby airport/runway, then refresh as needed. Observation/validity times remain in UTC in the raw text; failures and missing reports are displayed explicitly. This display-only source does not participate in weather decisions.
- Point analysis also displays an Open-Meteo Airgram (48-hour pressure-level profile sampled every three hours, UTC) with temperature, humidity, wind in knots and geopotential height above sea level, plus a Windy Airgram link for the analyzed coordinates. MGM Hezarfen's original Airgram image follows the selected airport ICAO code, not a fixed station. These views do not affect decisions; missing data is shown explicitly, and pressure levels may lie below local terrain.
- Point and airport coordinates are labeled separately, with straight-line distance from the local airport catalog when available. New point reports clear the previous airport observations/image; missing airport coordinates are shown explicitly rather than guessed.
- PDF exports retain the saved final decision (including more conservative AI decisions), include stored air-quality checks, and show the report's saved threshold reference rather than current server defaults.
- Each report parameter includes the rule range that produced its status, using the applied limits. The UI separates the deterministic rule result from a more conservative AI final decision and highlights the highest-risk parameters; PDF parameter rows include the same rule explanations. Missing cloud data is explicitly labeled without changing existing decision behavior.
- Hourly forecasts and flight planning show UTC dates/times, weather parameters and rule explanations using the report's saved limits. Planning does not substitute another date when the requested date is unavailable. Missing hours break suitable windows, past hours are not recommended, and the displayed window end is exclusive (09:00–11:00 covers two hourly slots). These are weather-only recommendations, not flight authorization.
- OpenWeatherMap, WeatherAPI, Windy, and CheckWX require API keys configured in `.env`. AVWX remains optional for runway lookup, and OpenAI remains optional for AI analysis.
- Extendable service and API module layout
- Testing and code quality tooling setup

## Installation

### Prerequisites

- Python 3.11+

### Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -e ".[dev]"
```

4. Copy environment template:

```bash
cp .env.example .env
```

## Quick Start

Run the API locally:

```bash
uvicorn decideflight.main:app --reload --app-dir src
```

Then open:

- API root: `http://127.0.0.1:8000/`
- Health endpoint: `http://127.0.0.1:8000/health`
- Swagger UI: `http://127.0.0.1:8000/docs`

## Architecture Overview

- `src/decideflight/main.py`: FastAPI application entrypoint
- `src/decideflight/config.py`: environment-backed application settings
- `src/decideflight/database.py`: SQLAlchemy engine/session/base setup
- `src/decideflight/models/`: ORM models (example weather observation model)
- `src/decideflight/api/`: API route modules
- `src/decideflight/services/`: business logic/services layer
- `tests/`: test suite
- `docs/SETUP.md`: detailed developer setup

## Contributing

1. Create a feature branch.
2. Add/update tests for your changes.
3. Run checks:

```bash
pytest
black --check src tests
flake8 src tests
```

4. Open a pull request with a clear summary.
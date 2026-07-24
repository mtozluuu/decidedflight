# DecideFlight Development Setup

## 1. Environment

- Python 3.11+
- Virtual environment recommended

```bash
python -m venv .venv
source .venv/bin/activate
```

## 2. Install Dependencies

```bash
pip install -e ".[dev]"
```

## 3. Configure Environment Variables

Create local environment file:

```bash
cp .env.example .env
```

Variables:

- `DATABASE_URL`: SQLAlchemy URL (PostgreSQL in production, SQLite for local dev)
- `OPENWEATHERMAP_API_KEY`: OpenWeatherMap API key
- `WEATHERAPI_API_KEY`: WeatherAPI key
- `DEBUG`: `true`/`false`

## 4. Run the Application

```bash
uvicorn decideflight.main:app --reload --app-dir src
```

## 5. Run Quality Checks

```bash
pytest
black --check src tests
flake8 src tests
```

## 6. Project Layout

```text
src/decideflight/
  api/
  models/
  services/
  config.py
  database.py
  main.py
tests/
docs/
```

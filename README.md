# DecideFlight

DecideFlight is an AI-powered flight weather decision system. It combines data from multiple weather providers and applies configurable rules/ML-assisted evaluation to determine whether flight conditions are suitable.

## Features

- FastAPI-based REST API foundation
- Config-driven environment management
- SQLAlchemy database layer ready for PostgreSQL
- Initial weather data model
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
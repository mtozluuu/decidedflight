import importlib

import decideflight.config as config_module


def test_settings_defaults(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("OPENWEATHERMAP_API_KEY", raising=False)
    monkeypatch.delenv("WEATHERAPI_API_KEY", raising=False)
    monkeypatch.delenv("CHECKWX_API_KEY", raising=False)
    monkeypatch.delenv("DEBUG", raising=False)

    config = importlib.reload(config_module)
    settings = config.Settings()

    assert settings.database_url == "sqlite:///./decideflight.db"
    assert settings.openweathermap_api_key == ""
    assert settings.weatherapi_api_key == ""
    assert settings.checkwx_api_key == ""
    assert settings.debug is False

from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):

    database_path: str = "../mgc.db"
    backend_url: str = "http://127.0.0.1:8000"

    # settings setup configuration
    model_config = SettingsConfigDict(
        env_file=".env-example",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# single initialized configuration instance
settings = AppSettings()
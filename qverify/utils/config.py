"""Environment-backed settings with lazy validation of required tokens."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Loaded once from environment and `.env`; tokens validated on demand."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    hf_token: str | None = None
    ibm_quantum_token: str | None = None
    ibm_quantum_instance: str | None = None
    wandb_api_key: str | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None

    def require(self, field_name: str) -> str:
        """Return the value of `field_name` or raise if missing."""
        value = getattr(self, field_name, None)
        if value is None or value == "":
            raise RuntimeError(
                f"Required setting '{field_name}' is not set. "
                f"Add it to your environment or .env file. "
                f"See .env.example for the full list of expected variables."
            )
        return str(value)


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Return the process-wide Settings singleton."""
    return Settings()

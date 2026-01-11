"""Configuration management for GitConnect.

Loads environment variables from .env file and provides typed access
to configuration values using pydantic-settings.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict



class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Neo4j Aura Connection
    neo4j_uri: str
    neo4j_username: str = "neo4j"
    neo4j_password: str

    # OpenAI API
    openai_api_key: str

    # Moorcheh API
    moorcheh_api_key: str

    # Gemini API
    gemini_api_key: str


    # GitHub (optional)
    github_token: Optional[str] = None

    # Application settings
    temp_clone_dir: str = "cloned_repos"
    log_level: str = "INFO"
    admin_secret: str = "super-secure-secret-for-dev"

    @property
    def neo4j_auth(self) -> tuple[str, str]:
        """Return Neo4j authentication tuple."""
        return (self.neo4j_username, self.neo4j_password)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()

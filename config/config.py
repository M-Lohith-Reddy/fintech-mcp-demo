"""
Configuration Management - Bank AI Assistant
Local ML model — NO external LLM API keys required.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings for ML-based Bank AI Assistant."""

    # ── ML Model ──────────────────────────────────────────────
    ml_model_path: str = "models/"
    llm_provider:  str = "local_ml"

    # ── Bank MCP Server ───────────────────────────────────────
    # Master API key for data_server.py — REQUIRED
    bank_api_key: str = Field(default="", alias="BANK_API_KEY")

    # ── GST API (optional) ────────────────────────────────────
    gst_api_url: Optional[str] = None
    gst_api_key: Optional[str] = None

    # ── Server ────────────────────────────────────────────────
    host:  str  = "0.0.0.0"
    port:  int  = 8000
    debug: bool = True

    # ── Logging ───────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Database (optional) ───────────────────────────────────
    database_url: Optional[str] = None

    # ── Security ──────────────────────────────────────────────
    enable_audit_log:    bool = True
    data_retention_days: int  = 90

    # ── Pydantic v2 style ─────────────────────────────────────
    model_config = {
        "env_file":            ".env",
        "case_sensitive":      False,
        "extra":               "ignore",   # replaces env_extra in v1
        "populate_by_name":    True,       # allows both field name and alias
    }


settings = Settings()
"""
Configuration Management - ML Model Version
NO external LLM API keys required
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings for ML-based system"""
    
    # ML Model Settings (LOCAL - NO API KEY NEEDED)
    ml_model_path: str = "models/"
    llm_provider: str = "local_ml"
    
    # Your GST API (optional - for external GST calculation if needed)
    gst_api_url: Optional[str] = None
    gst_api_key: Optional[str] = None
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Logging
    log_level: str = "INFO"
    
    # Database (for storing training data, user queries, etc.)
    database_url: Optional[str] = None
    
    # Security
    enable_audit_log: bool = True
    data_retention_days: int = 90
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        env_extra = "ignore"


settings = Settings()
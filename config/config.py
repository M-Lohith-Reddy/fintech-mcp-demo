"""
Configuration management
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Cohere API
    cohere_api_key: str
    llm_provider: str = "cohere"
    cohere_model: str = "command-r7b-12-2024"

    # Your GST API
    gst_api_url: Optional[str] = None
    gst_api_key: Optional[str] = None
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        env_extra = "ignore"
        


settings = Settings()
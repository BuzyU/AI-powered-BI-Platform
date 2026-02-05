# AI-Powered BI Platform - Configuration
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional, List
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"  # Allow extra fields in .env
    )
    
    # Application
    APP_NAME: str = "BI Intelligence Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    
    # Environment
    ENV: str = "development"
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"
    
    # Supabase Configuration
    SUPABASE_URL: str = "https://aivlfocxshysxxcxxkae.supabase.co"
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_SERVICE_KEY: Optional[str] = None
    
    # Database (Supabase PostgreSQL)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/bi_platform"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    
    # Redis (Optional)
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # File Storage (Uses Supabase Storage)
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE_MB: int = 100
    ALLOWED_EXTENSIONS: set = {"csv", "xls", "xlsx", "json"}
    
    # LLM Configuration
    LLM_MODEL_PATH: str = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    LLM_N_CTX: int = 2048
    LLM_N_THREADS: int = 4
    LLM_N_GPU_LAYERS: int = 0
    GROQ_API_KEY: Optional[str] = None
    
    # ChromaDB (Vector Store)
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    
    # Security
    SECRET_KEY: str = "change-this-in-production-use-a-secure-random-key"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # Cleanup Settings
    CLEANUP_INTERVAL_HOURS: int = 24
    SESSION_EXPIRE_HOURS: int = 168
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()


settings = get_settings()

# AI-Powered BI Platform - Configuration
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    APP_NAME: str = "BI Intelligence Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/bi_platform"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    # File Storage
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE_MB: int = 100
    ALLOWED_EXTENSIONS: set = {"csv", "xls", "xlsx", "json"}
    
    # MinIO (Object Storage)
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "bi-platform")
    MINIO_SECURE: bool = False
    
    # LLM Configuration
    LLM_MODEL_PATH: str = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    LLM_N_CTX: int = 2048
    LLM_N_THREADS: int = 4
    LLM_N_GPU_LAYERS: int = 0  # CPU only by default
    
    # ChromaDB (Vector Store)
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-this-in-production-use-a-secure-random-key")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # CORS
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:5173"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()


settings = get_settings()

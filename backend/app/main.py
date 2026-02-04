# AI-Powered BI Platform - Main Application Entry (Simplified)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.api.routes import upload

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from pathlib import Path
import shutil

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    logger.info(f"Starting BI Intelligence Platform v{settings.APP_VERSION}")
    
    # Clean up upload directory on startup
    upload_path = Path(settings.UPLOAD_DIR)
    if upload_path.exists():
        logger.info(f"Cleaning upload directory: {upload_path}")
        count = 0
        try:
            # Iterate over all items in the upload directory
            for item in upload_path.iterdir():
                if item.is_file():
                    item.unlink()
                    count += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    count += 1
            logger.info(f"Cleanup complete. Removed {count} items from uploads.")
        except Exception as e:
            logger.error(f"Error during startup cleanup: {e}")
        
    yield
    logger.info("Shutting down BI Intelligence Platform")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered Business Intelligence & CRM Intelligence Platform",
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(upload.router, prefix="/api", tags=["API"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/api/docs"
    }

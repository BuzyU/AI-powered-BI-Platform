# AI-Powered BI Platform - Main Application Entry (Optimized)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os

from app.config import settings
from app.api.routes import upload
from app.api.routes import session_dashboard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


from pathlib import Path
import shutil
from datetime import datetime, timedelta
import asyncio

# Background cleanup task
async def periodic_cleanup():
    """Periodically clean up old sessions and temporary files."""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            
            logger.info("Running periodic cleanup...")
            
            # Clean old temp files (older than 24 hours)
            upload_path = Path(settings.UPLOAD_DIR)
            if upload_path.exists():
                cutoff = datetime.now() - timedelta(hours=24)
                for item in upload_path.iterdir():
                    try:
                        mtime = datetime.fromtimestamp(item.stat().st_mtime)
                        if mtime < cutoff:
                            if item.is_file():
                                item.unlink()
                            elif item.is_dir():
                                shutil.rmtree(item)
                            logger.info(f"Cleaned up old file: {item.name}")
                    except Exception as e:
                        logger.error(f"Error cleaning {item}: {e}")
            
            # Clean expired session cache (keep data, clear in-memory cache)
            from app.services import state
            state._cleanup_memory_cache()
            
            logger.info("Periodic cleanup completed")
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Periodic cleanup error: {e}")

cleanup_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    global cleanup_task
    
    logger.info(f"Starting BI Intelligence Platform v{settings.APP_VERSION}")
    
    # Clean up upload directory on startup
    upload_path = Path(settings.UPLOAD_DIR)
    upload_path.mkdir(exist_ok=True)
    
    # Start background cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Shutdown
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
    
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

# CORS middleware - configure for production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if os.getenv("ENV") == "production" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(upload.router, prefix="/api", tags=["API"])
app.include_router(session_dashboard.router, prefix="/api", tags=["Dashboard"])


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

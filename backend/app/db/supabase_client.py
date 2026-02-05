# Supabase Client Configuration
from supabase import create_client, Client
from app.config import settings
import os
import logging

logger = logging.getLogger(__name__)

# Supabase credentials - loaded from settings (which reads .env)
# Fall back to os.getenv for container environments where env vars are set directly
SUPABASE_URL = settings.SUPABASE_URL or os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = settings.SUPABASE_ANON_KEY or os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = settings.SUPABASE_SERVICE_KEY or os.getenv("SUPABASE_SERVICE_KEY", "")

# Validate required credentials
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    logger.warning("Supabase credentials not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY environment variables.")
    supabase = None
else:
    # Create Supabase client
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    logger.info(f"Supabase client initialized for: {SUPABASE_URL}")


def get_supabase_client() -> Client:
    """Get Supabase client instance."""
    if supabase is None:
        raise RuntimeError("Supabase not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY.")
    return supabase


def get_admin_client() -> Client:
    """Get Supabase admin client with service role key."""
    if supabase is None:
        raise RuntimeError("Supabase not configured.")
    if SUPABASE_SERVICE_KEY:
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return supabase


def check_supabase_connection() -> dict:
    """Check if Supabase connection is working."""
    if supabase is None:
        return {
            "connected": False,
            "url": None,
            "error": "Supabase credentials not configured"
        }
    try:
        # Try to list buckets to verify connection
        buckets = supabase.storage.list_buckets()
        bucket_names = [b.name for b in buckets] if buckets else []
        return {
            "connected": True,
            "url": SUPABASE_URL,
            "buckets": bucket_names
        }
    except Exception as e:
        return {
            "connected": False,
            "url": SUPABASE_URL,
            "error": str(e)
        }


# Storage helpers
class SupabaseStorage:
    """Helper class for Supabase storage operations."""
    
    BUCKET_NAME = "bi-uploads"
    
    # Content type mapping for different file types
    # NOTE: Supabase bucket may restrict MIME types - model files (.h5, .onnx, .pkl)
    # might fail upload if bucket doesn't allow binary/application types
    # CSV files typically work with text/csv
    CONTENT_TYPES = {
        'csv': 'text/csv',
        'json': 'application/json',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'xls': 'application/vnd.ms-excel',
        'pkl': 'text/plain',  # Workaround for Supabase MIME restrictions
        'pickle': 'text/plain',
        'h5': 'text/plain',  # Workaround for Supabase MIME restrictions
        'hdf5': 'text/plain',
        'onnx': 'text/plain',
        'pt': 'text/plain',
        'pth': 'text/plain',
        'joblib': 'text/plain',
    }
    
    # Supabase Free tier file size limit (50MB)
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB in bytes
    
    @staticmethod
    def get_content_type(filename: str) -> str:
        """Get content type based on file extension."""
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        return SupabaseStorage.CONTENT_TYPES.get(ext, 'application/octet-stream')
    
    @staticmethod
    def upload_file(file_path: str, file_content: bytes, content_type: str = None) -> dict:
        """Upload file to Supabase storage (synchronous)."""
        if supabase is None:
            return {"success": False, "error": "Supabase not configured", "skipped": True}
        
        try:
            file_size = len(file_content)
            file_size_mb = file_size / (1024 * 1024)
            
            # Check file size limit
            if file_size > SupabaseStorage.MAX_FILE_SIZE:
                logger.warning(f"File too large for Supabase: {file_size_mb:.2f}MB (max 50MB). Skipping cloud upload.")
                return {
                    "success": False, 
                    "error": f"File too large ({file_size_mb:.2f}MB). Supabase limit is 50MB. File saved locally only.",
                    "skipped": True
                }
            
            # Auto-detect content type if not provided
            if content_type is None:
                content_type = SupabaseStorage.get_content_type(file_path)
            
            logger.info(f"Uploading to Supabase: {file_path} ({content_type})")
            logger.info(f"Content size: {file_size_mb:.2f}MB")
            
            # Check if file already exists and remove it first (upsert behavior)
            try:
                supabase.storage.from_(SupabaseStorage.BUCKET_NAME).remove([file_path])
                logger.info(f"Removed existing file: {file_path}")
            except Exception as rm_err:
                logger.debug(f"No existing file to remove: {rm_err}")
            
            response = supabase.storage.from_(SupabaseStorage.BUCKET_NAME).upload(
                path=file_path,
                file=file_content,
                file_options={"content-type": content_type, "upsert": "true"}
            )
            
            logger.info(f"Supabase upload SUCCESS: {file_path}")
            
            # Get the public URL
            try:
                public_url = supabase.storage.from_(SupabaseStorage.BUCKET_NAME).get_public_url(file_path)
                logger.info(f"File available at: {public_url}")
            except:
                public_url = None
            
            return {
                "success": True, 
                "path": file_path, 
                "data": str(response),
                "public_url": public_url
            }
        except Exception as e:
            import traceback
            logger.error(f"Supabase upload failed: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    @staticmethod
    async def upload_file_async(file_path: str, file_content: bytes, content_type: str = None) -> dict:
        """Async wrapper for upload_file."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: SupabaseStorage.upload_file(file_path, file_content, content_type)
        )
    
    @staticmethod
    def download_file(file_path: str) -> bytes:
        """Download file from Supabase storage."""
        if supabase is None:
            raise RuntimeError("Supabase not configured")
        response = supabase.storage.from_(SupabaseStorage.BUCKET_NAME).download(file_path)
        return response
    
    @staticmethod
    def get_public_url(file_path: str) -> str:
        """Get public URL for a file."""
        if supabase is None:
            return ""
        response = supabase.storage.from_(SupabaseStorage.BUCKET_NAME).get_public_url(file_path)
        return response
    
    @staticmethod
    def delete_file(file_path: str) -> dict:
        """Delete file from Supabase storage."""
        if supabase is None:
            return {"success": False, "error": "Supabase not configured"}
        try:
            response = supabase.storage.from_(SupabaseStorage.BUCKET_NAME).remove([file_path])
            return {"success": True, "data": response}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def list_files(folder: str = "") -> list:
        """List files in a folder."""
        if supabase is None:
            return []
        response = supabase.storage.from_(SupabaseStorage.BUCKET_NAME).list(folder)
        return response


# Database helpers using Supabase client
class SupabaseDB:
    """Helper class for Supabase database operations."""
    
    @staticmethod
    def table(table_name: str):
        """Get table reference."""
        if supabase is None:
            raise RuntimeError("Supabase not configured")
        return supabase.table(table_name)
    
    @staticmethod
    async def insert(table_name: str, data: dict) -> dict:
        """Insert data into table."""
        if supabase is None:
            return {}
        response = supabase.table(table_name).insert(data).execute()
        return response.data
    
    @staticmethod
    async def select(table_name: str, columns: str = "*", filters: dict = None) -> list:
        """Select data from table."""
        if supabase is None:
            return []
        query = supabase.table(table_name).select(columns)
        if filters:
            for key, value in filters.items():
                query = query.eq(key, value)
        response = query.execute()
        return response.data
    
    @staticmethod
    async def update(table_name: str, data: dict, filters: dict) -> dict:
        """Update data in table."""
        if supabase is None:
            return {}
        query = supabase.table(table_name).update(data)
        for key, value in filters.items():
            query = query.eq(key, value)
        response = query.execute()
        return response.data
    
    @staticmethod
    async def delete(table_name: str, filters: dict) -> dict:
        """Delete data from table."""
        if supabase is None:
            return {}
        query = supabase.table(table_name).delete()
        for key, value in filters.items():
            query = query.eq(key, value)
        response = query.execute()
        return response.data

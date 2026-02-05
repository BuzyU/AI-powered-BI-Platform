# Supabase Client Configuration
from supabase import create_client, Client
from app.config import settings
import os

# Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://aivlfocxshysxxcxxkae.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFpdmxmb2N4c2h5c3h4Y3h4a2FlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzAyNzAwNDgsImV4cCI6MjA4NTg0NjA0OH0.uX0StCRDK5t7YNjmgOyD6hl2MHX_o8TT5EOyPBWWzLk")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")  # Add service role key for admin operations

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def get_supabase_client() -> Client:
    """Get Supabase client instance."""
    return supabase


def get_admin_client() -> Client:
    """Get Supabase admin client with service role key."""
    if SUPABASE_SERVICE_KEY:
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return supabase


# Storage helpers
class SupabaseStorage:
    """Helper class for Supabase storage operations."""
    
    BUCKET_NAME = "bi-uploads"
    
    @staticmethod
    async def upload_file(file_path: str, file_content: bytes, content_type: str = "application/octet-stream") -> dict:
        """Upload file to Supabase storage."""
        try:
            response = supabase.storage.from_(SupabaseStorage.BUCKET_NAME).upload(
                file_path,
                file_content,
                {"content-type": content_type}
            )
            return {"success": True, "path": file_path, "data": response}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    async def download_file(file_path: str) -> bytes:
        """Download file from Supabase storage."""
        response = supabase.storage.from_(SupabaseStorage.BUCKET_NAME).download(file_path)
        return response
    
    @staticmethod
    async def get_public_url(file_path: str) -> str:
        """Get public URL for a file."""
        response = supabase.storage.from_(SupabaseStorage.BUCKET_NAME).get_public_url(file_path)
        return response
    
    @staticmethod
    async def delete_file(file_path: str) -> dict:
        """Delete file from Supabase storage."""
        try:
            response = supabase.storage.from_(SupabaseStorage.BUCKET_NAME).remove([file_path])
            return {"success": True, "data": response}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    async def list_files(folder: str = "") -> list:
        """List files in a folder."""
        response = supabase.storage.from_(SupabaseStorage.BUCKET_NAME).list(folder)
        return response


# Database helpers using Supabase client
class SupabaseDB:
    """Helper class for Supabase database operations."""
    
    @staticmethod
    def table(table_name: str):
        """Get table reference."""
        return supabase.table(table_name)
    
    @staticmethod
    async def insert(table_name: str, data: dict) -> dict:
        """Insert data into table."""
        response = supabase.table(table_name).insert(data).execute()
        return response.data
    
    @staticmethod
    async def select(table_name: str, columns: str = "*", filters: dict = None) -> list:
        """Select data from table."""
        query = supabase.table(table_name).select(columns)
        if filters:
            for key, value in filters.items():
                query = query.eq(key, value)
        response = query.execute()
        return response.data
    
    @staticmethod
    async def update(table_name: str, data: dict, filters: dict) -> dict:
        """Update data in table."""
        query = supabase.table(table_name).update(data)
        for key, value in filters.items():
            query = query.eq(key, value)
        response = query.execute()
        return response.data
    
    @staticmethod
    async def delete(table_name: str, filters: dict) -> dict:
        """Delete data from table."""
        query = supabase.table(table_name).delete()
        for key, value in filters.items():
            query = query.eq(key, value)
        response = query.execute()
        return response.data

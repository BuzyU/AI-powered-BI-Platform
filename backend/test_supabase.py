"""Test Supabase Storage Upload"""
from supabase import create_client
import sys

SUPABASE_URL = "https://aivlfocxshysxxcxxkae.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFpdmxmb2N4c2h5c3h4Y3h4a2FlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzAyNzAwNDgsImV4cCI6MjA4NTg0NjA0OH0.uX0StCRDK5t7YNjmgOyD6hl2MHX_o8TT5EOyPBWWzLk"

def test():
    print("=" * 50)
    print("SUPABASE UPLOAD TEST")
    print("=" * 50)
    
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    
    # Test small CSV upload
    test_files = [
        ("test/sample.csv", b"name,age,city\nAlice,30,NYC\nBob,25,LA", "text/csv"),
        ("test/sample.json", b'{"users": [{"name": "Alice"}, {"name": "Bob"}]}', "application/json"),
        ("test/sample.xlsx", b"PK\x03\x04", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ]
    
    for path, content, content_type in test_files:
        print(f"\nUploading: {path}")
        try:
            # Remove existing
            try:
                client.storage.from_("bi-uploads").remove([path])
            except:
                pass
            
            # Upload
            result = client.storage.from_("bi-uploads").upload(
                path=path,
                file=content,
                file_options={"content-type": content_type, "upsert": "true"}
            )
            print(f"  ✓ Success: {result}")
            
            # Get URL
            url = client.storage.from_("bi-uploads").get_public_url(path)
            print(f"  → URL: {url}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print("\n" + "=" * 50)
    print("TEST COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    test()

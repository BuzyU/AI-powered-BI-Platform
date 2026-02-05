-- =====================================================
-- BI INTELLIGENCE PLATFORM - STORAGE BUCKET SETUP
-- =====================================================
-- Run this in Supabase SQL Editor AFTER running 01_tables_and_policies.sql
-- =====================================================

-- Create storage bucket for file uploads
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'bi-uploads',
    'bi-uploads',
    false,  -- Private bucket (access through signed URLs)
    104857600,  -- 100MB max file size
    ARRAY['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/json', 'text/plain']
)
ON CONFLICT (id) DO UPDATE SET
    file_size_limit = 104857600,
    allowed_mime_types = ARRAY['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/json', 'text/plain'];

-- =====================================================
-- STORAGE POLICIES
-- =====================================================

-- Allow service role full access
CREATE POLICY "Service role full access to bi-uploads"
ON storage.objects FOR ALL
USING (bucket_id = 'bi-uploads' AND auth.role() = 'service_role')
WITH CHECK (bucket_id = 'bi-uploads' AND auth.role() = 'service_role');

-- Allow anon role to upload files (backend will use this)
CREATE POLICY "Anon can upload to bi-uploads"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'bi-uploads' AND auth.role() = 'anon');

-- Allow anon role to read files
CREATE POLICY "Anon can read from bi-uploads"
ON storage.objects FOR SELECT
USING (bucket_id = 'bi-uploads' AND auth.role() = 'anon');

-- Allow anon role to update files
CREATE POLICY "Anon can update bi-uploads"
ON storage.objects FOR UPDATE
USING (bucket_id = 'bi-uploads' AND auth.role() = 'anon')
WITH CHECK (bucket_id = 'bi-uploads' AND auth.role() = 'anon');

-- Allow anon role to delete files
CREATE POLICY "Anon can delete from bi-uploads"
ON storage.objects FOR DELETE
USING (bucket_id = 'bi-uploads' AND auth.role() = 'anon');

-- =====================================================
-- DONE! Storage bucket created with policies.
-- =====================================================

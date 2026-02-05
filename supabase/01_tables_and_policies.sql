-- =====================================================
-- BI INTELLIGENCE PLATFORM - SUPABASE DATABASE SETUP
-- =====================================================
-- Run this in Supabase SQL Editor: https://supabase.com/dashboard/project/aivlfocxshysxxcxxkae/sql
-- =====================================================

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =====================================================
-- 1. TABLES
-- =====================================================

-- Tenants (Organizations)
CREATE TABLE IF NOT EXISTS tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Datasets (Uploaded files metadata)
CREATE TABLE IF NOT EXISTS datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(20) NOT NULL,
    file_path VARCHAR(500),
    storage_path VARCHAR(500), -- Supabase storage path
    
    -- Classification
    role VARCHAR(50), -- transactional, master_entity, interaction, financial, operational, reference
    role_confidence FLOAT,
    
    -- Metadata
    row_count INTEGER,
    column_count INTEGER,
    columns JSONB,
    date_range_start DATE,
    date_range_end DATE,
    
    -- Quality
    health_score FLOAT,
    
    -- Status
    status VARCHAR(20) DEFAULT 'uploaded',
    error_message TEXT,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Column Mappings
CREATE TABLE IF NOT EXISTS column_mappings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    
    original_column VARCHAR(255) NOT NULL,
    canonical_field VARCHAR(100),
    confidence FLOAT,
    user_confirmed BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Customers (Canonical customer entity)
CREATE TABLE IF NOT EXISTS customers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    
    external_id VARCHAR(255),
    external_id_hash VARCHAR(64),
    name VARCHAR(255),
    email_hash VARCHAR(64),
    
    -- Computed fields
    first_seen DATE,
    last_seen DATE,
    total_transactions INTEGER DEFAULT 0,
    total_revenue FLOAT DEFAULT 0,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT uq_customer_tenant_external UNIQUE (tenant_id, external_id_hash)
);

-- Offerings (Products/Services)
CREATE TABLE IF NOT EXISTS offerings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    
    external_id VARCHAR(255),
    name VARCHAR(500),
    description TEXT,
    category VARCHAR(255),
    
    -- Offering type classification
    offering_type VARCHAR(50),
    offering_type_confidence FLOAT,
    
    -- Cost
    unit_cost FLOAT,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Transactions
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    
    external_id VARCHAR(255),
    customer_id UUID REFERENCES customers(id),
    offering_id UUID REFERENCES offerings(id),
    
    transaction_date DATE NOT NULL,
    quantity FLOAT,
    unit_price FLOAT,
    total_amount FLOAT,
    cost FLOAT,
    profit FLOAT,
    
    source_dataset_id UUID REFERENCES datasets(id),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Interactions (Support tickets, feedback)
CREATE TABLE IF NOT EXISTS interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    customer_id UUID REFERENCES customers(id),
    
    interaction_type VARCHAR(50),
    interaction_date TIMESTAMPTZ,
    content TEXT,
    
    -- AI-extracted fields
    sentiment VARCHAR(20),
    sentiment_score FLOAT,
    topics TEXT[],
    
    resolution_status VARCHAR(50),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Daily Metrics (Aggregated analytics)
CREATE TABLE IF NOT EXISTS metrics_daily (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    
    metric_date DATE NOT NULL,
    offering_id UUID REFERENCES offerings(id),
    
    revenue FLOAT DEFAULT 0,
    cost FLOAT DEFAULT 0,
    profit FLOAT DEFAULT 0,
    transaction_count INTEGER DEFAULT 0,
    unique_customers INTEGER DEFAULT 0,
    avg_order_value FLOAT,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT uq_metrics_daily UNIQUE (tenant_id, metric_date, offering_id)
);

-- Insights (AI-generated insights)
CREATE TABLE IF NOT EXISTS insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    
    insight_type VARCHAR(50),
    severity VARCHAR(20),
    
    title VARCHAR(500),
    description TEXT,
    evidence JSONB,
    recommendation TEXT,
    expected_impact VARCHAR(255),
    confidence FLOAT,
    
    llm_explanation TEXT,
    
    is_dismissed BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Analysis Jobs (Track processing status)
CREATE TABLE IF NOT EXISTS analysis_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    
    job_type VARCHAR(50),
    status VARCHAR(20) DEFAULT 'pending',
    
    progress INTEGER DEFAULT 0,
    current_step VARCHAR(100),
    
    dataset_ids UUID[],
    result JSONB,
    error_message TEXT,
    
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- User Sessions (For session management)
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    
    user_persona VARCHAR(50) DEFAULT 'business', -- business, analytics, ml_engineer
    preferences JSONB DEFAULT '{}',
    
    last_active TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '7 days')
);

-- Chat History (For Q&A feature)
CREATE TABLE IF NOT EXISTS chat_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES user_sessions(id) ON DELETE CASCADE,
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    
    role VARCHAR(20) NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =====================================================
-- 2. INDEXES
-- =====================================================

CREATE INDEX IF NOT EXISTS idx_datasets_tenant ON datasets(tenant_id);
CREATE INDEX IF NOT EXISTS idx_datasets_status ON datasets(tenant_id, status);

CREATE INDEX IF NOT EXISTS idx_customers_tenant ON customers(tenant_id);
CREATE INDEX IF NOT EXISTS idx_customers_external ON customers(tenant_id, external_id_hash);

CREATE INDEX IF NOT EXISTS idx_offerings_tenant ON offerings(tenant_id);
CREATE INDEX IF NOT EXISTS idx_offerings_type ON offerings(tenant_id, offering_type);

CREATE INDEX IF NOT EXISTS idx_transactions_tenant_date ON transactions(tenant_id, transaction_date);
CREATE INDEX IF NOT EXISTS idx_transactions_customer ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_offering ON transactions(offering_id);

CREATE INDEX IF NOT EXISTS idx_interactions_tenant ON interactions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_interactions_customer ON interactions(customer_id);

CREATE INDEX IF NOT EXISTS idx_metrics_tenant_date ON metrics_daily(tenant_id, metric_date);

CREATE INDEX IF NOT EXISTS idx_insights_tenant ON insights(tenant_id);
CREATE INDEX IF NOT EXISTS idx_insights_severity ON insights(tenant_id, severity);

CREATE INDEX IF NOT EXISTS idx_jobs_tenant ON analysis_jobs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON analysis_jobs(status);

CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON user_sessions(expires_at);

CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history(session_id);

-- =====================================================
-- 3. UPDATED_AT TRIGGER
-- =====================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_datasets_updated_at
    BEFORE UPDATE ON datasets
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- 4. ROW LEVEL SECURITY (RLS) POLICIES
-- =====================================================

-- Enable RLS on all tables
ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;
ALTER TABLE datasets ENABLE ROW LEVEL SECURITY;
ALTER TABLE column_mappings ENABLE ROW LEVEL SECURITY;
ALTER TABLE customers ENABLE ROW LEVEL SECURITY;
ALTER TABLE offerings ENABLE ROW LEVEL SECURITY;
ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE interactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE metrics_daily ENABLE ROW LEVEL SECURITY;
ALTER TABLE insights ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_history ENABLE ROW LEVEL SECURITY;

-- Allow service role full access (for backend)
CREATE POLICY "Service role has full access to tenants"
    ON tenants FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "Service role has full access to datasets"
    ON datasets FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "Service role has full access to column_mappings"
    ON column_mappings FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "Service role has full access to customers"
    ON customers FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "Service role has full access to offerings"
    ON offerings FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "Service role has full access to transactions"
    ON transactions FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "Service role has full access to interactions"
    ON interactions FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "Service role has full access to metrics_daily"
    ON metrics_daily FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "Service role has full access to insights"
    ON insights FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "Service role has full access to analysis_jobs"
    ON analysis_jobs FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "Service role has full access to user_sessions"
    ON user_sessions FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "Service role has full access to chat_history"
    ON chat_history FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

-- Allow anon role to read/write for public API (controlled by backend)
CREATE POLICY "Anon can manage tenants"
    ON tenants FOR ALL
    USING (auth.role() = 'anon')
    WITH CHECK (auth.role() = 'anon');

CREATE POLICY "Anon can manage datasets"
    ON datasets FOR ALL
    USING (auth.role() = 'anon')
    WITH CHECK (auth.role() = 'anon');

CREATE POLICY "Anon can manage column_mappings"
    ON column_mappings FOR ALL
    USING (auth.role() = 'anon')
    WITH CHECK (auth.role() = 'anon');

CREATE POLICY "Anon can manage customers"
    ON customers FOR ALL
    USING (auth.role() = 'anon')
    WITH CHECK (auth.role() = 'anon');

CREATE POLICY "Anon can manage offerings"
    ON offerings FOR ALL
    USING (auth.role() = 'anon')
    WITH CHECK (auth.role() = 'anon');

CREATE POLICY "Anon can manage transactions"
    ON transactions FOR ALL
    USING (auth.role() = 'anon')
    WITH CHECK (auth.role() = 'anon');

CREATE POLICY "Anon can manage interactions"
    ON interactions FOR ALL
    USING (auth.role() = 'anon')
    WITH CHECK (auth.role() = 'anon');

CREATE POLICY "Anon can manage metrics_daily"
    ON metrics_daily FOR ALL
    USING (auth.role() = 'anon')
    WITH CHECK (auth.role() = 'anon');

CREATE POLICY "Anon can manage insights"
    ON insights FOR ALL
    USING (auth.role() = 'anon')
    WITH CHECK (auth.role() = 'anon');

CREATE POLICY "Anon can manage analysis_jobs"
    ON analysis_jobs FOR ALL
    USING (auth.role() = 'anon')
    WITH CHECK (auth.role() = 'anon');

CREATE POLICY "Anon can manage user_sessions"
    ON user_sessions FOR ALL
    USING (auth.role() = 'anon')
    WITH CHECK (auth.role() = 'anon');

CREATE POLICY "Anon can manage chat_history"
    ON chat_history FOR ALL
    USING (auth.role() = 'anon')
    WITH CHECK (auth.role() = 'anon');

-- =====================================================
-- 5. CREATE DEFAULT TENANT
-- =====================================================

INSERT INTO tenants (id, name) 
VALUES ('00000000-0000-0000-0000-000000000001', 'Default Tenant')
ON CONFLICT DO NOTHING;

-- =====================================================
-- DONE! All tables, indexes, and policies created.
-- =====================================================

# SQLAlchemy Database Models
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, DateTime, Date,
    ForeignKey, Index, UniqueConstraint, ARRAY, JSON
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

from app.db.session import Base


class Tenant(Base):
    """Multi-tenant organization."""
    __tablename__ = "tenants"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    datasets = relationship("Dataset", back_populates="tenant", cascade="all, delete-orphan")
    customers = relationship("Customer", back_populates="tenant", cascade="all, delete-orphan")
    offerings = relationship("Offering", back_populates="tenant", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="tenant", cascade="all, delete-orphan")
    insights = relationship("Insight", back_populates="tenant", cascade="all, delete-orphan")


class Dataset(Base):
    """Uploaded dataset metadata."""
    __tablename__ = "datasets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(20), nullable=False)  # csv, xlsx, json
    file_path = Column(String(500))
    
    # Classification
    role = Column(String(50))  # transactional, master_entity, interaction, financial, operational, reference
    role_confidence = Column(Float)
    
    # Metadata
    row_count = Column(Integer)
    column_count = Column(Integer)
    columns = Column(JSON)  # List of column names
    date_range_start = Column(Date)
    date_range_end = Column(Date)
    
    # Quality
    health_score = Column(Float)
    
    # Status
    status = Column(String(20), default="uploaded")  # uploaded, classified, mapped, processed, error
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tenant = relationship("Tenant", back_populates="datasets")
    column_mappings = relationship("ColumnMapping", back_populates="dataset", cascade="all, delete-orphan")


class ColumnMapping(Base):
    """Column mapping from user data to canonical schema."""
    __tablename__ = "column_mappings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    
    original_column = Column(String(255), nullable=False)
    canonical_field = Column(String(100))  # Null if not mapped
    confidence = Column(Float)
    user_confirmed = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="column_mappings")


class Customer(Base):
    """Canonical customer entity."""
    __tablename__ = "customers"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    
    external_id = Column(String(255))
    external_id_hash = Column(String(64))  # SHA256 for privacy
    name = Column(String(255))
    email_hash = Column(String(64))
    
    # Computed fields
    first_seen = Column(Date)
    last_seen = Column(Date)
    total_transactions = Column(Integer, default=0)
    total_revenue = Column(Float, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('tenant_id', 'external_id_hash', name='uq_customer_tenant_external'),
        Index('idx_customer_tenant', 'tenant_id'),
    )
    
    # Relationships
    tenant = relationship("Tenant", back_populates="customers")
    transactions = relationship("Transaction", back_populates="customer")
    interactions = relationship("Interaction", back_populates="customer")


class Offering(Base):
    """Canonical offering (product/service) entity."""
    __tablename__ = "offerings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    
    external_id = Column(String(255))
    name = Column(String(500))
    description = Column(Text)
    category = Column(String(255))
    
    # Offering type classification
    offering_type = Column(String(50))  # physical_product, service, subscription, etc.
    offering_type_confidence = Column(Float)
    
    # Cost
    unit_cost = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_offering_tenant', 'tenant_id'),
        Index('idx_offering_type', 'tenant_id', 'offering_type'),
    )
    
    # Relationships
    tenant = relationship("Tenant", back_populates="offerings")
    transactions = relationship("Transaction", back_populates="offering")


class Transaction(Base):
    """Canonical transaction entity."""
    __tablename__ = "transactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    
    external_id = Column(String(255))
    customer_id = Column(UUID(as_uuid=True), ForeignKey("customers.id"))
    offering_id = Column(UUID(as_uuid=True), ForeignKey("offerings.id"))
    
    transaction_date = Column(Date, nullable=False)
    quantity = Column(Float)
    unit_price = Column(Float)
    total_amount = Column(Float)
    cost = Column(Float)
    profit = Column(Float)  # Calculated: total_amount - cost
    
    # Metadata
    source_dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_txn_tenant_date', 'tenant_id', 'transaction_date'),
        Index('idx_txn_customer', 'customer_id'),
        Index('idx_txn_offering', 'offering_id'),
    )
    
    # Relationships
    tenant = relationship("Tenant", back_populates="transactions")
    customer = relationship("Customer", back_populates="transactions")
    offering = relationship("Offering", back_populates="transactions")


class Interaction(Base):
    """Customer interaction (support ticket, feedback, etc.)."""
    __tablename__ = "interactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("customers.id"))
    
    interaction_type = Column(String(50))  # support_ticket, feedback, complaint
    interaction_date = Column(DateTime)
    content = Column(Text)
    
    # AI-extracted fields
    sentiment = Column(String(20))  # positive, negative, neutral
    sentiment_score = Column(Float)
    topics = Column(ARRAY(String))
    
    resolution_status = Column(String(50))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_interaction_tenant', 'tenant_id'),
        Index('idx_interaction_customer', 'customer_id'),
    )
    
    # Relationships
    customer = relationship("Customer", back_populates="interactions")


class MetricsDaily(Base):
    """Aggregated daily metrics for analytics."""
    __tablename__ = "metrics_daily"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    
    metric_date = Column(Date, nullable=False)
    offering_id = Column(UUID(as_uuid=True), ForeignKey("offerings.id"))
    
    revenue = Column(Float, default=0)
    cost = Column(Float, default=0)
    profit = Column(Float, default=0)
    transaction_count = Column(Integer, default=0)
    unique_customers = Column(Integer, default=0)
    avg_order_value = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('tenant_id', 'metric_date', 'offering_id', name='uq_metrics_daily'),
        Index('idx_metrics_tenant_date', 'tenant_id', 'metric_date'),
    )


class Insight(Base):
    """Generated insights and recommendations."""
    __tablename__ = "insights"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    
    insight_type = Column(String(50))  # loss_alert, growth_opportunity, risk_warning
    severity = Column(String(20))  # low, medium, high, critical
    
    title = Column(String(500))
    description = Column(Text)
    evidence = Column(JSON)  # Supporting metrics
    recommendation = Column(Text)
    expected_impact = Column(String(255))
    confidence = Column(Float)
    
    # LLM explanation
    llm_explanation = Column(Text)
    
    # Status
    is_dismissed = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_insight_tenant', 'tenant_id'),
        Index('idx_insight_severity', 'tenant_id', 'severity'),
    )
    
    # Relationships
    tenant = relationship("Tenant", back_populates="insights")


class AnalysisJob(Base):
    """Track analysis job status."""
    __tablename__ = "analysis_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    
    job_type = Column(String(50))  # classify, map, analyze, full_pipeline
    status = Column(String(20), default="pending")  # pending, processing, completed, failed
    
    progress = Column(Integer, default=0)  # 0-100
    current_step = Column(String(100))
    
    dataset_ids = Column(ARRAY(UUID(as_uuid=True)))
    result = Column(JSON)
    error_message = Column(Text)
    
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_job_tenant', 'tenant_id'),
        Index('idx_job_status', 'status'),
    )

# Pydantic Schemas for API Request/Response
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from uuid import UUID
from enum import Enum


# Enums
class DatasetRole(str, Enum):
    TRANSACTIONAL = "transactional"
    MASTER_ENTITY = "master_entity"
    INTERACTION = "interaction"
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    REFERENCE = "reference"


class OfferingType(str, Enum):
    PHYSICAL_PRODUCT = "physical_product"
    SERVICE = "service"
    SUBSCRIPTION = "subscription"
    EVENT = "event"
    BUNDLE = "bundle"
    DIGITAL_ASSET = "digital_asset"
    LICENSE = "license"
    MEMBERSHIP = "membership"
    RENTAL = "rental"
    USAGE_BASED = "usage_based"
    FREEMIUM = "freemium"
    COMMISSION_BASED = "commission_based"
    TRAINING = "training"
    SUPPORT_PLAN = "support_plan"
    CUSTOM_CONTRACT = "custom_contract"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class InsightSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Base schemas
class TenantBase(BaseModel):
    name: str


class TenantCreate(TenantBase):
    pass


class TenantResponse(TenantBase):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    created_at: datetime


# Dataset schemas
class DatasetBase(BaseModel):
    filename: str
    file_type: str


class DatasetUploadResponse(BaseModel):
    id: UUID
    filename: str
    file_type: str
    sheets: Optional[List[str]] = None
    status: str
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    created_at: datetime


class DatasetListResponse(BaseModel):
    success: bool = True
    datasets: List[DatasetUploadResponse]


class ClassificationResult(BaseModel):
    dataset_id: UUID
    detected_role: DatasetRole
    role_confidence: float
    date_range: Optional[Dict[str, str]] = None
    dominant_entities: List[str]
    requires_confirmation: bool


class ClassifyRequest(BaseModel):
    dataset_ids: List[UUID]


class ClassifyJobResponse(BaseModel):
    job_id: UUID
    status: JobStatus


class ClassifyResultResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    results: Optional[List[ClassificationResult]] = None
    error: Optional[str] = None


# Mapping schemas
class ColumnMappingItem(BaseModel):
    original_column: str
    canonical_field: Optional[str] = None
    confidence: Optional[float] = None
    confirmed: bool = False


class MappingWarning(BaseModel):
    type: str
    message: str
    severity: str


class MappingResponse(BaseModel):
    dataset_id: UUID
    status: str
    mappings: List[ColumnMappingItem]
    unmapped_columns: List[str]
    missing_required: List[str]
    health_score: float
    warnings: List[MappingWarning]


class MappingUpdateRequest(BaseModel):
    mappings: List[ColumnMappingItem]
    confirm_all: bool = False


# Analysis schemas
class AnalyzeRequest(BaseModel):
    dataset_ids: List[UUID]


class AnalyzeJobResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    progress: int = 0
    current_step: Optional[str] = None


# Dashboard schemas
class KPIItem(BaseModel):
    name: str
    value: float
    format: str  # currency, percent, number
    trend: Optional[float] = None


class ChartData(BaseModel):
    id: str
    type: str  # line, bar, pie, scatter
    title: str
    data: Dict[str, Any]


class AlertItem(BaseModel):
    severity: str
    message: str
    link: Optional[str] = None


class DashboardSheet(BaseModel):
    id: str
    name: str
    type: str


class DashboardResponse(BaseModel):
    tenant_id: UUID
    generated_at: datetime
    date_range: Dict[str, str]
    sheets: List[DashboardSheet]


class SheetDataResponse(BaseModel):
    sheet_id: str
    kpis: Optional[List[KPIItem]] = None
    charts: Optional[List[ChartData]] = None
    alerts: Optional[List[AlertItem]] = None
    data: Optional[Dict[str, Any]] = None


# Offering schemas
class OfferingMetrics(BaseModel):
    id: UUID
    name: str
    offering_type: Optional[str] = None
    revenue: float
    profit: float
    cost: float
    profit_margin: float
    transaction_count: int
    unique_customers: int
    stability_score: Optional[float] = None
    risk_score: Optional[float] = None


# Insight schemas
class InsightEvidence(BaseModel):
    metrics: Dict[str, Any]


class InsightResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    insight_type: str
    severity: InsightSeverity
    title: str
    description: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None
    expected_impact: Optional[str] = None
    confidence: float
    llm_explanation: Optional[str] = None
    created_at: datetime


class InsightListResponse(BaseModel):
    insights: List[InsightResponse]
    total: int


# Q&A schemas
class AskRequest(BaseModel):
    question: str
    context: Optional[Dict[str, Any]] = None


class AnswerDetail(BaseModel):
    offering: Optional[str] = None
    loss: Optional[float] = None
    reason: Optional[str] = None


class AnswerResponse(BaseModel):
    question: str
    answer: Dict[str, Any]


# Generic responses
class SuccessResponse(BaseModel):
    success: bool = True
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[Dict[str, Any]] = None

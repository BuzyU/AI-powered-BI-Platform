# Analysis API Routes
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID
import structlog

from app.api.deps import get_session, get_tenant
from app.models.db import Tenant, Dataset, AnalysisJob
from app.models.schemas import AnalyzeRequest, AnalyzeJobResponse, JobStatus
from app.services.pipeline import run_full_analysis_pipeline

router = APIRouter()
logger = structlog.get_logger()


@router.post("/analyze", response_model=AnalyzeJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_analysis(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_tenant)
):
    """
    Start full analysis pipeline on confirmed datasets.
    This includes:
    - Relationship inference
    - Data normalization
    - Analytics computation
    - Insight generation
    """
    # Validate all datasets are mapped
    for dataset_id in request.dataset_ids:
        result = await db.execute(
            select(Dataset)
            .where(Dataset.id == dataset_id, Dataset.tenant_id == tenant.id)
        )
        dataset = result.scalar_one_or_none()
        
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )
        
        if dataset.status not in ["mapped", "processed"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dataset {dataset_id} must be mapped before analysis"
            )
    
    # Check for existing running job
    result = await db.execute(
        select(AnalysisJob)
        .where(
            AnalysisJob.tenant_id == tenant.id,
            AnalysisJob.job_type == "full_analysis",
            AnalysisJob.status == "processing"
        )
    )
    existing_job = result.scalar_one_or_none()
    
    if existing_job:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Analysis already in progress"
        )
    
    # Create analysis job
    job = AnalysisJob(
        tenant_id=tenant.id,
        job_type="full_analysis",
        status="pending",
        dataset_ids=request.dataset_ids,
        progress=0,
        current_step="Initializing analysis"
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    
    # Start background analysis
    background_tasks.add_task(
        run_full_analysis_pipeline,
        job_id=job.id,
        dataset_ids=request.dataset_ids,
        tenant_id=tenant.id
    )
    
    logger.info("Analysis job started", job_id=str(job.id))
    
    return AnalyzeJobResponse(
        job_id=job.id,
        status=JobStatus.PROCESSING,
        progress=0,
        current_step="Initializing analysis"
    )


@router.get("/analyze/{job_id}", response_model=AnalyzeJobResponse)
async def get_analysis_status(
    job_id: UUID,
    db: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_tenant)
):
    """Get analysis job status and progress."""
    result = await db.execute(
        select(AnalysisJob)
        .where(AnalysisJob.id == job_id, AnalysisJob.tenant_id == tenant.id)
    )
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    return AnalyzeJobResponse(
        job_id=job.id,
        status=JobStatus(job.status),
        progress=job.progress or 0,
        current_step=job.current_step
    )


@router.get("/analyze/{job_id}/result")
async def get_analysis_result(
    job_id: UUID,
    db: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_tenant)
):
    """Get analysis job result."""
    result = await db.execute(
        select(AnalysisJob)
        .where(AnalysisJob.id == job_id, AnalysisJob.tenant_id == tenant.id)
    )
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job not completed. Current status: {job.status}"
        )
    
    return {
        "job_id": job.id,
        "status": job.status,
        "completed_at": job.completed_at,
        "result": job.result
    }

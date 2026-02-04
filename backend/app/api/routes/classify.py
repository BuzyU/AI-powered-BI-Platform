# Classification API Routes
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from uuid import UUID
import structlog

from app.api.deps import get_session, get_tenant
from app.models.db import Tenant, Dataset, AnalysisJob
from app.models.schemas import (
    ClassifyRequest, ClassifyJobResponse, ClassifyResultResponse,
    ClassificationResult, DatasetRole, JobStatus
)
from app.layers.l2_classification.role_detector import classify_dataset_role
from app.services.pipeline import run_classification_job

router = APIRouter()
logger = structlog.get_logger()


@router.post("/classify", response_model=ClassifyJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def classify_datasets(
    request: ClassifyRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_tenant)
):
    """
    Start classification job for uploaded datasets.
    Returns a job ID for tracking progress.
    """
    # Validate datasets exist
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
    
    # Create analysis job
    job = AnalysisJob(
        tenant_id=tenant.id,
        job_type="classify",
        status="pending",
        dataset_ids=request.dataset_ids,
        progress=0,
        current_step="Starting classification"
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    
    # Start background classification
    background_tasks.add_task(
        run_classification_job,
        job_id=job.id,
        dataset_ids=request.dataset_ids,
        tenant_id=tenant.id
    )
    
    logger.info("Classification job started", job_id=str(job.id))
    
    return ClassifyJobResponse(
        job_id=job.id,
        status=JobStatus.PROCESSING
    )


@router.get("/classify/{job_id}", response_model=ClassifyResultResponse)
async def get_classification_result(
    job_id: UUID,
    db: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_tenant)
):
    """Get classification job status and results."""
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
    
    response = ClassifyResultResponse(
        job_id=job.id,
        status=JobStatus(job.status),
        results=None,
        error=job.error_message
    )
    
    if job.status == "completed" and job.result:
        response.results = [
            ClassificationResult(**r) for r in job.result.get("classifications", [])
        ]
    
    return response


@router.post("/classify/quick")
async def quick_classify(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_tenant)
):
    """
    Quickly classify a single dataset synchronously.
    For small datasets or immediate feedback.
    """
    result = await db.execute(
        select(Dataset)
        .where(Dataset.id == dataset_id, Dataset.tenant_id == tenant.id)
    )
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    # Run classification
    try:
        classification = await classify_dataset_role(dataset.file_path, dataset.file_type)
        
        # Update dataset
        dataset.role = classification["role"]
        dataset.role_confidence = classification["confidence"]
        dataset.date_range_start = classification.get("date_range_start")
        dataset.date_range_end = classification.get("date_range_end")
        dataset.status = "classified"
        
        await db.commit()
        
        return {
            "dataset_id": dataset.id,
            "detected_role": classification["role"],
            "role_confidence": classification["confidence"],
            "date_range": {
                "start": str(classification.get("date_range_start")) if classification.get("date_range_start") else None,
                "end": str(classification.get("date_range_end")) if classification.get("date_range_end") else None
            },
            "dominant_entities": classification.get("entities", []),
            "requires_confirmation": classification["confidence"] < 0.8
        }
        
    except Exception as e:
        logger.error("Classification failed", dataset_id=str(dataset_id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )

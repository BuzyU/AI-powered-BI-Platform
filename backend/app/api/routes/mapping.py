# Mapping API Routes
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID
import structlog

from app.api.deps import get_session, get_tenant
from app.models.db import Tenant, Dataset, ColumnMapping
from app.models.schemas import (
    MappingResponse, MappingUpdateRequest, ColumnMappingItem, MappingWarning
)
from app.layers.l3_schema_mapping.mapper import generate_column_mappings
from app.layers.l3_schema_mapping.health import calculate_health_score

router = APIRouter()
logger = structlog.get_logger()


@router.get("/mapping/{dataset_id}", response_model=MappingResponse)
async def get_mapping(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_tenant)
):
    """Get column mappings for a dataset."""
    # Get dataset
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
    
    # Get existing mappings
    result = await db.execute(
        select(ColumnMapping)
        .where(ColumnMapping.dataset_id == dataset_id)
    )
    existing_mappings = result.scalars().all()
    
    # If no mappings exist, generate them
    if not existing_mappings:
        try:
            mappings_data = await generate_column_mappings(
                dataset.file_path, 
                dataset.file_type,
                dataset.columns
            )
            
            # Save mappings to database
            for mapping in mappings_data["mappings"]:
                col_mapping = ColumnMapping(
                    dataset_id=dataset_id,
                    original_column=mapping["original_column"],
                    canonical_field=mapping.get("canonical_field"),
                    confidence=mapping.get("confidence"),
                    user_confirmed=False
                )
                db.add(col_mapping)
            
            await db.commit()
            
            # Fetch again
            result = await db.execute(
                select(ColumnMapping)
                .where(ColumnMapping.dataset_id == dataset_id)
            )
            existing_mappings = result.scalars().all()
            
        except Exception as e:
            logger.error("Mapping generation failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate mappings: {str(e)}"
            )
    
    # Build response
    mappings = [
        ColumnMappingItem(
            original_column=m.original_column,
            canonical_field=m.canonical_field,
            confidence=m.confidence,
            confirmed=m.user_confirmed
        )
        for m in existing_mappings
    ]
    
    # Calculate health score
    health_result = await calculate_health_score(
        mappings=[m.__dict__ for m in existing_mappings],
        dataset_role=dataset.role
    )
    
    # Update dataset health
    dataset.health_score = health_result["score"]
    await db.commit()
    
    return MappingResponse(
        dataset_id=dataset_id,
        status="pending_confirmation" if any(not m.confirmed for m in mappings) else "confirmed",
        mappings=mappings,
        unmapped_columns=health_result.get("unmapped", []),
        missing_required=health_result.get("missing_required", []),
        health_score=health_result["score"],
        warnings=[
            MappingWarning(type=w["type"], message=w["message"], severity=w["severity"])
            for w in health_result.get("warnings", [])
        ]
    )


@router.put("/mapping/{dataset_id}")
async def update_mapping(
    dataset_id: UUID,
    request: MappingUpdateRequest,
    db: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_tenant)
):
    """Update column mappings for a dataset."""
    # Verify dataset exists
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
    
    # Update mappings
    for mapping_update in request.mappings:
        result = await db.execute(
            select(ColumnMapping)
            .where(
                ColumnMapping.dataset_id == dataset_id,
                ColumnMapping.original_column == mapping_update.original_column
            )
        )
        mapping = result.scalar_one_or_none()
        
        if mapping:
            if mapping_update.canonical_field is not None:
                mapping.canonical_field = mapping_update.canonical_field
            mapping.user_confirmed = mapping_update.confirmed or request.confirm_all
    
    # If confirm_all, confirm all mappings
    if request.confirm_all:
        result = await db.execute(
            select(ColumnMapping)
            .where(ColumnMapping.dataset_id == dataset_id)
        )
        for mapping in result.scalars().all():
            mapping.user_confirmed = True
        
        dataset.status = "mapped"
    
    await db.commit()
    
    logger.info("Mappings updated", dataset_id=str(dataset_id))
    
    return {"success": True, "message": "Mappings updated"}


@router.post("/mapping/{dataset_id}/generate")
async def regenerate_mapping(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_tenant)
):
    """Regenerate column mappings for a dataset."""
    # Get dataset
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
    
    # Delete existing mappings
    result = await db.execute(
        select(ColumnMapping)
        .where(ColumnMapping.dataset_id == dataset_id)
    )
    for mapping in result.scalars().all():
        await db.delete(mapping)
    
    await db.commit()
    
    # Generate new mappings
    return await get_mapping(dataset_id, db, tenant)

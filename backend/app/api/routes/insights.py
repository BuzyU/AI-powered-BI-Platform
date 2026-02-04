# Insights API Routes
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID
from typing import Optional
import structlog

from app.api.deps import get_session, get_tenant
from app.models.db import Tenant, Insight
from app.models.schemas import InsightResponse, InsightListResponse, InsightSeverity

router = APIRouter()
logger = structlog.get_logger()


@router.get("/insights", response_model=InsightListResponse)
async def get_insights(
    severity: Optional[str] = Query(default=None),
    insight_type: Optional[str] = Query(default=None),
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0),
    db: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_tenant)
):
    """Get all insights for the tenant."""
    query = select(Insight).where(
        Insight.tenant_id == tenant.id,
        Insight.is_dismissed == False
    )
    
    if severity:
        query = query.where(Insight.severity == severity)
    if insight_type:
        query = query.where(Insight.insight_type == insight_type)
    
    query = query.order_by(
        Insight.severity.desc(),
        Insight.created_at.desc()
    ).offset(offset).limit(limit)
    
    result = await db.execute(query)
    insights = result.scalars().all()
    
    # Get total count
    count_query = select(func.count(Insight.id)).where(
        Insight.tenant_id == tenant.id,
        Insight.is_dismissed == False
    )
    if severity:
        count_query = count_query.where(Insight.severity == severity)
    if insight_type:
        count_query = count_query.where(Insight.insight_type == insight_type)
    
    from sqlalchemy import func
    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0
    
    return InsightListResponse(
        insights=[
            InsightResponse(
                id=i.id,
                insight_type=i.insight_type,
                severity=InsightSeverity(i.severity),
                title=i.title,
                description=i.description,
                evidence=i.evidence,
                recommendation=i.recommendation,
                expected_impact=i.expected_impact,
                confidence=i.confidence,
                llm_explanation=i.llm_explanation,
                created_at=i.created_at
            )
            for i in insights
        ],
        total=total
    )


@router.get("/insights/{insight_id}", response_model=InsightResponse)
async def get_insight(
    insight_id: UUID,
    db: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_tenant)
):
    """Get a specific insight."""
    result = await db.execute(
        select(Insight)
        .where(Insight.id == insight_id, Insight.tenant_id == tenant.id)
    )
    insight = result.scalar_one_or_none()
    
    if not insight:
        raise HTTPException(status_code=404, detail="Insight not found")
    
    return InsightResponse(
        id=insight.id,
        insight_type=insight.insight_type,
        severity=InsightSeverity(insight.severity),
        title=insight.title,
        description=insight.description,
        evidence=insight.evidence,
        recommendation=insight.recommendation,
        expected_impact=insight.expected_impact,
        confidence=insight.confidence,
        llm_explanation=insight.llm_explanation,
        created_at=insight.created_at
    )


@router.post("/insights/{insight_id}/dismiss")
async def dismiss_insight(
    insight_id: UUID,
    db: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_tenant)
):
    """Dismiss an insight."""
    result = await db.execute(
        select(Insight)
        .where(Insight.id == insight_id, Insight.tenant_id == tenant.id)
    )
    insight = result.scalar_one_or_none()
    
    if not insight:
        raise HTTPException(status_code=404, detail="Insight not found")
    
    insight.is_dismissed = True
    await db.commit()
    
    return {"success": True, "message": "Insight dismissed"}


@router.post("/insights/{insight_id}/explain")
async def explain_insight(
    insight_id: UUID,
    db: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_tenant)
):
    """Generate LLM explanation for an insight."""
    from app.layers.l10_llm.explainer import generate_explanation
    
    result = await db.execute(
        select(Insight)
        .where(Insight.id == insight_id, Insight.tenant_id == tenant.id)
    )
    insight = result.scalar_one_or_none()
    
    if not insight:
        raise HTTPException(status_code=404, detail="Insight not found")
    
    # Generate explanation if not exists
    if not insight.llm_explanation:
        try:
            explanation = await generate_explanation(
                insight_type=insight.insight_type,
                title=insight.title,
                evidence=insight.evidence,
                recommendation=insight.recommendation
            )
            insight.llm_explanation = explanation
            await db.commit()
        except Exception as e:
            logger.error("Failed to generate explanation", error=str(e))
            raise HTTPException(
                status_code=500,
                detail="Failed to generate explanation"
            )
    
    return {
        "insight_id": insight.id,
        "explanation": insight.llm_explanation
    }

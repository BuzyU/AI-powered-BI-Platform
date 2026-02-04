# API Dependencies
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends, HTTPException, status
from uuid import UUID

from app.db.session import get_db, async_session_maker
from app.models.db import Tenant


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency."""
    async for session in get_db():
        yield session


async def get_tenant(
    tenant_id: UUID = None,
    db: AsyncSession = Depends(get_session)
) -> Tenant:
    """Get current tenant (for multi-tenancy)."""
    # For MVP, use a default tenant or get from header
    # In production, this would come from auth token
    from sqlalchemy import select
    
    if tenant_id:
        result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
        tenant = result.scalar_one_or_none()
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        return tenant
    
    # Get or create default tenant
    result = await db.execute(select(Tenant).where(Tenant.name == "Default"))
    tenant = result.scalar_one_or_none()
    
    if not tenant:
        tenant = Tenant(name="Default")
        db.add(tenant)
        await db.commit()
        await db.refresh(tenant)
    
    return tenant

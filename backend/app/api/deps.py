# API Dependencies
from typing import AsyncGenerator, TYPE_CHECKING
from fastapi import Depends, HTTPException, status, Header
from uuid import UUID
import re

# Defer database imports to avoid import-time errors
if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.models.db import Tenant


# Session ID validation regex - UUID format only
SESSION_ID_PATTERN = re.compile(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$', re.IGNORECASE)


def validate_session_id(session_id: str) -> str:
    """Validate session ID format to prevent path traversal and injection attacks."""
    if not session_id:
        raise HTTPException(400, "Session ID is required")
    
    # Allow special default for backward compatibility (but don't persist with it)
    if session_id == "default_session":
        return session_id
    
    # Must be valid UUID format
    if not SESSION_ID_PATTERN.match(session_id):
        raise HTTPException(400, "Invalid session ID format. Must be a valid UUID.")
    
    return session_id


async def get_validated_session_id(
    x_session_id: str = Header("default_session")
) -> str:
    """FastAPI dependency for validated session ID."""
    return validate_session_id(x_session_id)


async def get_session():
    """Get database session dependency."""
    from app.db.session import get_db
    async for session in get_db():
        yield session


async def get_tenant(
    tenant_id: UUID = None,
    db = Depends(get_session)
):
    """Get current tenant (for multi-tenancy)."""
    from sqlalchemy import select
    from app.models.db import Tenant
    from app.db.session import async_session_maker
    
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

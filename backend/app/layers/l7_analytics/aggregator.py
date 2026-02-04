# L7: Analytics Layer - Aggregator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, extract
from typing import Dict, Any, List
from datetime import date
from uuid import UUID

from app.models.db import Transaction, Offering


async def aggregate_metrics(
    db: AsyncSession,
    tenant_id: UUID,
    start_date: date,
    end_date: date,
    granularity: str = "month"  # day, week, month
) -> Dict[str, Any]:
    """
    Aggregate metrics over time for trend charts.
    Returns data suitable for line/bar charts.
    """
    # Determine grouping based on granularity
    if granularity == "day":
        date_trunc = func.date_trunc('day', Transaction.transaction_date)
    elif granularity == "week":
        date_trunc = func.date_trunc('week', Transaction.transaction_date)
    else:  # month
        date_trunc = func.date_trunc('month', Transaction.transaction_date)
    
    result = await db.execute(
        select(
            date_trunc.label("period"),
            func.sum(Transaction.total_amount).label("revenue"),
            func.sum(Transaction.profit).label("profit"),
            func.sum(Transaction.cost).label("cost"),
            func.count(Transaction.id).label("txn_count"),
            func.count(func.distinct(Transaction.customer_id)).label("customers")
        )
        .where(
            Transaction.tenant_id == tenant_id,
            Transaction.transaction_date >= start_date,
            Transaction.transaction_date <= end_date
        )
        .group_by(date_trunc)
        .order_by(date_trunc)
    )
    
    rows = result.all()
    
    labels = []
    revenue_series = []
    profit_series = []
    cost_series = []
    
    for row in rows:
        if row.period:
            if granularity == "day":
                labels.append(row.period.strftime("%Y-%m-%d"))
            elif granularity == "week":
                labels.append(row.period.strftime("W%W %Y"))
            else:
                labels.append(row.period.strftime("%b %Y"))
            
            revenue_series.append(float(row.revenue or 0))
            profit_series.append(float(row.profit or 0))
            cost_series.append(float(row.cost or 0))
    
    return {
        "labels": labels,
        "series": [
            {"name": "Revenue", "values": revenue_series},
            {"name": "Profit", "values": profit_series},
            {"name": "Cost", "values": cost_series}
        ]
    }


async def aggregate_by_offering_type(
    db: AsyncSession,
    tenant_id: UUID,
    start_date: date,
    end_date: date
) -> Dict[str, Any]:
    """Aggregate metrics by offering type."""
    result = await db.execute(
        select(
            Offering.offering_type,
            func.sum(Transaction.total_amount).label("revenue"),
            func.sum(Transaction.profit).label("profit"),
            func.count(Transaction.id).label("txn_count")
        )
        .join(Transaction, Transaction.offering_id == Offering.id)
        .where(
            Offering.tenant_id == tenant_id,
            Transaction.transaction_date >= start_date,
            Transaction.transaction_date <= end_date
        )
        .group_by(Offering.offering_type)
    )
    
    data = []
    for row in result.all():
        data.append({
            "type": row.offering_type or "Unknown",
            "revenue": float(row.revenue or 0),
            "profit": float(row.profit or 0),
            "transactions": row.txn_count
        })
    
    return {"by_type": data}


async def get_top_offerings(
    db: AsyncSession,
    tenant_id: UUID,
    start_date: date,
    end_date: date,
    limit: int = 10,
    sort_by: str = "revenue"
) -> List[Dict[str, Any]]:
    """Get top offerings by revenue or profit."""
    sort_col = Transaction.total_amount if sort_by == "revenue" else Transaction.profit
    
    result = await db.execute(
        select(
            Offering.id,
            Offering.name,
            Offering.offering_type,
            func.sum(Transaction.total_amount).label("revenue"),
            func.sum(Transaction.profit).label("profit")
        )
        .join(Transaction, Transaction.offering_id == Offering.id)
        .where(
            Offering.tenant_id == tenant_id,
            Transaction.transaction_date >= start_date,
            Transaction.transaction_date <= end_date
        )
        .group_by(Offering.id, Offering.name, Offering.offering_type)
        .order_by(func.sum(sort_col).desc())
        .limit(limit)
    )
    
    return [
        {
            "id": str(row.id),
            "name": row.name,
            "type": row.offering_type,
            "revenue": float(row.revenue or 0),
            "profit": float(row.profit or 0)
        }
        for row in result.all()
    ]


async def get_bottom_offerings(
    db: AsyncSession,
    tenant_id: UUID,
    start_date: date,
    end_date: date,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Get bottom offerings by profit (loss-makers)."""
    result = await db.execute(
        select(
            Offering.id,
            Offering.name,
            Offering.offering_type,
            func.sum(Transaction.total_amount).label("revenue"),
            func.sum(Transaction.profit).label("profit")
        )
        .join(Transaction, Transaction.offering_id == Offering.id)
        .where(
            Offering.tenant_id == tenant_id,
            Transaction.transaction_date >= start_date,
            Transaction.transaction_date <= end_date
        )
        .group_by(Offering.id, Offering.name, Offering.offering_type)
        .having(func.sum(Transaction.profit) < 0)
        .order_by(func.sum(Transaction.profit).asc())
        .limit(limit)
    )
    
    return [
        {
            "id": str(row.id),
            "name": row.name,
            "type": row.offering_type,
            "revenue": float(row.revenue or 0),
            "profit": float(row.profit or 0),
            "loss": abs(float(row.profit or 0))
        }
        for row in result.all()
    ]

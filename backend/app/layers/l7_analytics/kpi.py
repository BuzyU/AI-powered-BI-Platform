# L7: Analytics Layer - KPI Calculations
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Dict, Any, Optional
from datetime import date, timedelta
from uuid import UUID

from app.models.db import Transaction, Offering, Customer


async def calculate_kpis(
    db: AsyncSession,
    tenant_id: UUID,
    start_date: date,
    end_date: date
) -> Dict[str, Any]:
    """
    Calculate key performance indicators for the given period.
    All calculations are deterministic SQL aggregations.
    """
    # Current period metrics
    current_metrics = await get_period_metrics(db, tenant_id, start_date, end_date)
    
    # Previous period for trend calculation
    period_days = (end_date - start_date).days
    prev_end = start_date - timedelta(days=1)
    prev_start = prev_end - timedelta(days=period_days)
    prev_metrics = await get_period_metrics(db, tenant_id, prev_start, prev_end)
    
    # Calculate trends
    def calc_trend(current, previous):
        if previous and previous > 0:
            return round((current - previous) / previous, 2)
        return None
    
    return {
        "total_revenue": float(current_metrics["revenue"] or 0),
        "total_profit": float(current_metrics["profit"] or 0),
        "total_cost": float(current_metrics["cost"] or 0),
        "profit_margin": round(
            (current_metrics["profit"] / current_metrics["revenue"] * 100) 
            if current_metrics["revenue"] and current_metrics["revenue"] > 0 else 0, 
            2
        ),
        "transaction_count": current_metrics["txn_count"] or 0,
        "unique_customers": current_metrics["customers"] or 0,
        "avg_order_value": round(
            (current_metrics["revenue"] / current_metrics["txn_count"]) 
            if current_metrics["txn_count"] and current_metrics["txn_count"] > 0 else 0, 
            2
        ),
        "revenue_trend": calc_trend(current_metrics["revenue"] or 0, prev_metrics["revenue"]),
        "profit_trend": calc_trend(current_metrics["profit"] or 0, prev_metrics["profit"]),
        "margin_trend": calc_trend(
            current_metrics["profit"] / current_metrics["revenue"] if current_metrics["revenue"] else 0,
            prev_metrics["profit"] / prev_metrics["revenue"] if prev_metrics["revenue"] else 0
        ) if current_metrics["revenue"] and prev_metrics["revenue"] else None,
        "customer_trend": calc_trend(current_metrics["customers"] or 0, prev_metrics["customers"])
    }


async def get_period_metrics(
    db: AsyncSession,
    tenant_id: UUID,
    start_date: date,
    end_date: date
) -> Dict[str, Any]:
    """Get aggregated metrics for a period."""
    result = await db.execute(
        select(
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
    )
    row = result.one()
    
    return {
        "revenue": row.revenue or 0,
        "profit": row.profit or 0,
        "cost": row.cost or 0,
        "txn_count": row.txn_count or 0,
        "customers": row.customers or 0
    }


async def calculate_offering_metrics(
    db: AsyncSession,
    tenant_id: UUID,
    start_date: date,
    end_date: date
) -> Dict[str, Dict[str, Any]]:
    """Calculate metrics per offering."""
    result = await db.execute(
        select(
            Offering.id,
            Offering.name,
            Offering.offering_type,
            func.sum(Transaction.total_amount).label("revenue"),
            func.sum(Transaction.profit).label("profit"),
            func.sum(Transaction.cost).label("cost"),
            func.count(Transaction.id).label("txn_count"),
            func.count(func.distinct(Transaction.customer_id)).label("customers")
        )
        .join(Transaction, Transaction.offering_id == Offering.id)
        .where(
            Offering.tenant_id == tenant_id,
            Transaction.transaction_date >= start_date,
            Transaction.transaction_date <= end_date
        )
        .group_by(Offering.id, Offering.name, Offering.offering_type)
    )
    
    offerings = {}
    for row in result.all():
        revenue = float(row.revenue or 0)
        profit = float(row.profit or 0)
        
        offerings[str(row.id)] = {
            "id": str(row.id),
            "name": row.name,
            "type": row.offering_type,
            "revenue": revenue,
            "profit": profit,
            "cost": float(row.cost or 0),
            "margin": round((profit / revenue * 100) if revenue > 0 else 0, 2),
            "transactions": row.txn_count,
            "customers": row.customers
        }
    
    return offerings


async def calculate_customer_metrics(
    db: AsyncSession,
    tenant_id: UUID,
    start_date: date,
    end_date: date
) -> Dict[str, Dict[str, Any]]:
    """Calculate metrics per customer."""
    result = await db.execute(
        select(
            Customer.id,
            Customer.name,
            func.sum(Transaction.total_amount).label("revenue"),
            func.count(Transaction.id).label("txn_count"),
            func.min(Transaction.transaction_date).label("first_txn"),
            func.max(Transaction.transaction_date).label("last_txn")
        )
        .join(Transaction, Transaction.customer_id == Customer.id)
        .where(
            Customer.tenant_id == tenant_id,
            Transaction.transaction_date >= start_date,
            Transaction.transaction_date <= end_date
        )
        .group_by(Customer.id, Customer.name)
    )
    
    customers = {}
    for row in result.all():
        customers[str(row.id)] = {
            "id": str(row.id),
            "name": row.name,
            "revenue": float(row.revenue or 0),
            "transactions": row.txn_count,
            "first_transaction": row.first_txn,
            "last_transaction": row.last_txn,
            "is_repeat": row.txn_count > 1
        }
    
    return customers

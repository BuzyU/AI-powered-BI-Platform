# Dashboard API Routes
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from uuid import UUID
from datetime import datetime, date, timedelta
import structlog

from app.api.deps import get_session, get_tenant
from app.models.db import Tenant, Transaction, Offering, Customer, MetricsDaily, Insight
from app.models.schemas import (
    DashboardResponse, DashboardSheet, SheetDataResponse,
    KPIItem, ChartData, AlertItem
)
from app.layers.l7_analytics.kpi import calculate_kpis
from app.layers.l7_analytics.aggregator import aggregate_metrics

router = APIRouter()
logger = structlog.get_logger()


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(
    start_date: date = Query(default=None),
    end_date: date = Query(default=None),
    db: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_tenant)
):
    """Get dashboard overview with available sheets."""
    # Default date range: last 12 months
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=365)
    
    return DashboardResponse(
        tenant_id=tenant.id,
        generated_at=datetime.utcnow(),
        date_range={
            "start": start_date.isoformat(),
            "end": end_date.isoformat()
        },
        sheets=[
            DashboardSheet(id="overview", name="Business Overview", type="overview"),
            DashboardSheet(id="offerings", name="Product & Service Performance", type="offerings"),
            DashboardSheet(id="losses", name="Loss & Leakage Analysis", type="losses"),
            DashboardSheet(id="customers", name="Customer Intelligence", type="customers"),
            DashboardSheet(id="insights", name="AI Insights & Recommendations", type="insights"),
        ]
    )


@router.get("/dashboard/sheets/overview")
async def get_overview_sheet(
    start_date: date = Query(default=None),
    end_date: date = Query(default=None),
    db: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_tenant)
):
    """Get overview sheet with KPIs and trends."""
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=365)
    
    # Calculate KPIs
    kpis = await calculate_kpis(db, tenant.id, start_date, end_date)
    
    # Get trend data
    trend_data = await aggregate_metrics(db, tenant.id, start_date, end_date, "month")
    
    # Get alerts (critical insights)
    result = await db.execute(
        select(Insight)
        .where(
            Insight.tenant_id == tenant.id,
            Insight.severity.in_(["critical", "high"]),
            Insight.is_dismissed == False
        )
        .limit(5)
    )
    critical_insights = result.scalars().all()
    
    return SheetDataResponse(
        sheet_id="overview",
        kpis=[
            KPIItem(name="Total Revenue", value=kpis["total_revenue"], format="currency", trend=kpis.get("revenue_trend")),
            KPIItem(name="Total Profit", value=kpis["total_profit"], format="currency", trend=kpis.get("profit_trend")),
            KPIItem(name="Profit Margin", value=kpis["profit_margin"], format="percent", trend=kpis.get("margin_trend")),
            KPIItem(name="Total Customers", value=kpis["unique_customers"], format="number", trend=kpis.get("customer_trend")),
        ],
        charts=[
            ChartData(
                id="revenue_trend",
                type="line",
                title="Revenue & Profit Trend",
                data=trend_data
            )
        ],
        alerts=[
            AlertItem(
                severity=i.severity,
                message=i.title,
                link=f"/insights/{i.id}"
            )
            for i in critical_insights
        ]
    )


@router.get("/dashboard/sheets/offerings")
async def get_offerings_sheet(
    start_date: date = Query(default=None),
    end_date: date = Query(default=None),
    db: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_tenant)
):
    """Get offering performance sheet."""
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=365)
    
    # Get offering metrics
    result = await db.execute(
        select(
            Offering.id,
            Offering.name,
            Offering.offering_type,
            func.sum(Transaction.total_amount).label("revenue"),
            func.sum(Transaction.profit).label("profit"),
            func.sum(Transaction.cost).label("cost"),
            func.count(Transaction.id).label("transaction_count"),
            func.count(func.distinct(Transaction.customer_id)).label("unique_customers")
        )
        .join(Transaction, Transaction.offering_id == Offering.id)
        .where(
            Offering.tenant_id == tenant.id,
            Transaction.transaction_date >= start_date,
            Transaction.transaction_date <= end_date
        )
        .group_by(Offering.id, Offering.name, Offering.offering_type)
        .order_by(func.sum(Transaction.total_amount).desc())
    )
    offerings = result.all()
    
    offerings_data = []
    for o in offerings:
        margin = (o.profit / o.revenue * 100) if o.revenue and o.revenue > 0 else 0
        offerings_data.append({
            "id": str(o.id),
            "name": o.name,
            "type": o.offering_type,
            "revenue": float(o.revenue or 0),
            "profit": float(o.profit or 0),
            "cost": float(o.cost or 0),
            "margin": float(margin),
            "transactions": o.transaction_count,
            "customers": o.unique_customers
        })
    
    return SheetDataResponse(
        sheet_id="offerings",
        data={
            "offerings": offerings_data,
            "summary": {
                "total_offerings": len(offerings_data),
                "profitable": len([o for o in offerings_data if o["profit"] > 0]),
                "unprofitable": len([o for o in offerings_data if o["profit"] < 0])
            }
        }
    )


@router.get("/dashboard/sheets/losses")
async def get_losses_sheet(
    start_date: date = Query(default=None),
    end_date: date = Query(default=None),
    db: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_tenant)
):
    """Get loss and leakage analysis sheet."""
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=365)
    
    # Get loss-making offerings
    result = await db.execute(
        select(
            Offering.id,
            Offering.name,
            Offering.offering_type,
            func.sum(Transaction.profit).label("total_profit"),
            func.sum(Transaction.total_amount).label("revenue"),
            func.sum(Transaction.cost).label("cost"),
            func.count(Transaction.id).label("transaction_count")
        )
        .join(Transaction, Transaction.offering_id == Offering.id)
        .where(
            Offering.tenant_id == tenant.id,
            Transaction.transaction_date >= start_date,
            Transaction.transaction_date <= end_date
        )
        .group_by(Offering.id, Offering.name, Offering.offering_type)
        .having(func.sum(Transaction.profit) < 0)
        .order_by(func.sum(Transaction.profit).asc())
    )
    loss_items = result.all()
    
    total_loss = sum(item.total_profit for item in loss_items)
    
    return SheetDataResponse(
        sheet_id="losses",
        kpis=[
            KPIItem(name="Total Losses", value=abs(total_loss), format="currency"),
            KPIItem(name="Loss-Making Items", value=len(loss_items), format="number"),
        ],
        data={
            "loss_items": [
                {
                    "id": str(item.id),
                    "name": item.name,
                    "type": item.offering_type,
                    "loss": float(item.total_profit),
                    "revenue": float(item.revenue or 0),
                    "cost": float(item.cost or 0),
                    "transactions": item.transaction_count
                }
                for item in loss_items
            ]
        }
    )


@router.get("/dashboard/sheets/customers")
async def get_customers_sheet(
    start_date: date = Query(default=None),
    end_date: date = Query(default=None),
    db: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_tenant)
):
    """Get customer intelligence sheet."""
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=365)
    
    # Get customer metrics
    result = await db.execute(
        select(
            func.count(func.distinct(Transaction.customer_id)).label("total_customers"),
            func.sum(Transaction.total_amount).label("total_revenue")
        )
        .where(
            Transaction.tenant_id == tenant.id,
            Transaction.transaction_date >= start_date,
            Transaction.transaction_date <= end_date
        )
    )
    stats = result.one()
    
    # Repeat vs one-time customers
    result = await db.execute(
        select(
            Transaction.customer_id,
            func.count(Transaction.id).label("txn_count")
        )
        .where(
            Transaction.tenant_id == tenant.id,
            Transaction.transaction_date >= start_date,
            Transaction.transaction_date <= end_date
        )
        .group_by(Transaction.customer_id)
    )
    customer_txns = result.all()
    
    repeat_customers = len([c for c in customer_txns if c.txn_count > 1])
    one_time_customers = len([c for c in customer_txns if c.txn_count == 1])
    
    return SheetDataResponse(
        sheet_id="customers",
        kpis=[
            KPIItem(name="Total Customers", value=stats.total_customers or 0, format="number"),
            KPIItem(name="Repeat Customers", value=repeat_customers, format="number"),
            KPIItem(name="One-Time Customers", value=one_time_customers, format="number"),
            KPIItem(name="Avg Revenue per Customer", 
                   value=(stats.total_revenue / stats.total_customers) if stats.total_customers else 0, 
                   format="currency"),
        ],
        data={
            "retention_rate": (repeat_customers / stats.total_customers * 100) if stats.total_customers else 0
        }
    )

# L10: LLM Layer - Q&A Handler
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, Optional
from uuid import UUID
from datetime import date, timedelta
import structlog

from app.layers.l7_analytics.kpi import calculate_kpis
from app.layers.l7_analytics.aggregator import (
    get_top_offerings, get_bottom_offerings, aggregate_by_offering_type
)

logger = structlog.get_logger()


async def answer_question(
    db: AsyncSession,
    tenant_id: UUID,
    question: str,
    question_type: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Answer a natural language question using deterministic analytics.
    The LLM is used for formatting the response, not computing values.
    """
    # Default date range
    end_date = date.today()
    start_date = end_date - timedelta(days=365)
    
    handlers = {
        "revenue": handle_revenue_question,
        "loss": handle_loss_question,
        "profit": handle_profit_question,
        "growth": handle_growth_question,
        "risk": handle_risk_question,
        "customer": handle_customer_question,
        "discontinue": handle_discontinue_question,
        "compare": handle_compare_question,
    }
    
    handler = handlers.get(question_type, handle_general_question)
    
    return await handler(db, tenant_id, question, start_date, end_date, context)


async def handle_revenue_question(
    db: AsyncSession,
    tenant_id: UUID,
    question: str,
    start_date: date,
    end_date: date,
    context: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Handle revenue-related questions."""
    kpis = await calculate_kpis(db, tenant_id, start_date, end_date)
    top = await get_top_offerings(db, tenant_id, start_date, end_date, limit=5, sort_by="revenue")
    
    return {
        "summary": f"Your total revenue for the period is ${kpis['total_revenue']:,.2f}.",
        "data": {
            "total_revenue": kpis["total_revenue"],
            "trend": kpis.get("revenue_trend"),
            "top_offerings": top
        },
        "explanation": "Revenue is calculated as the sum of all transaction amounts.",
        "period": f"{start_date} to {end_date}"
    }


async def handle_loss_question(
    db: AsyncSession,
    tenant_id: UUID,
    question: str,
    start_date: date,
    end_date: date,
    context: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Handle loss/leakage questions."""
    losses = await get_bottom_offerings(db, tenant_id, start_date, end_date, limit=10)
    
    total_loss = sum(item["profit"] for item in losses)
    
    items = []
    for item in losses[:5]:
        items.append({
            "offering": item["name"],
            "loss": abs(item["profit"]),
            "reason": determine_loss_reason(item)
        })
    
    return {
        "summary": f"You are losing ${abs(total_loss):,.2f} across {len(losses)} offerings.",
        "top_losses": items,
        "total_loss": abs(total_loss),
        "explanation": "Losses occur when cost exceeds revenue. This can be due to pricing below cost, high overhead allocation, or volume shortfalls."
    }


async def handle_profit_question(
    db: AsyncSession,
    tenant_id: UUID,
    question: str,
    start_date: date,
    end_date: date,
    context: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Handle profit-related questions."""
    kpis = await calculate_kpis(db, tenant_id, start_date, end_date)
    top = await get_top_offerings(db, tenant_id, start_date, end_date, limit=5, sort_by="profit")
    
    return {
        "summary": f"Your total profit is ${kpis['total_profit']:,.2f} with a {kpis['profit_margin']:.1f}% margin.",
        "data": {
            "total_profit": kpis["total_profit"],
            "profit_margin": kpis["profit_margin"],
            "trend": kpis.get("profit_trend"),
            "most_profitable": top
        },
        "explanation": "Profit is calculated as revenue minus cost. Margin is profit as a percentage of revenue."
    }


async def handle_growth_question(
    db: AsyncSession,
    tenant_id: UUID,
    question: str,
    start_date: date,
    end_date: date,
    context: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Handle growth/focus questions."""
    top = await get_top_offerings(db, tenant_id, start_date, end_date, limit=5, sort_by="profit")
    
    # Filter for high-margin offerings
    focus_candidates = [
        o for o in top 
        if o.get("revenue", 0) > 0 and 
        (o.get("profit", 0) / o["revenue"]) > 0.2
    ]
    
    return {
        "summary": "Focus on your highest-margin offerings with growth potential.",
        "recommendations": focus_candidates,
        "explanation": "Offerings with high margins and stable demand are the best candidates for increased investment."
    }


async def handle_risk_question(
    db: AsyncSession,
    tenant_id: UUID,
    question: str,
    start_date: date,
    end_date: date,
    context: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Handle risk-related questions."""
    losses = await get_bottom_offerings(db, tenant_id, start_date, end_date, limit=5)
    
    risky = []
    for item in losses:
        risky.append({
            "offering": item["name"],
            "risk_factors": ["Negative profit margin"],
            "recommendation": "Review pricing or discontinue"
        })
    
    return {
        "summary": f"{len(risky)} offerings show risk indicators.",
        "risky_offerings": risky,
        "explanation": "Risk is assessed based on profit margin, revenue stability, and customer concentration."
    }


async def handle_customer_question(
    db: AsyncSession,
    tenant_id: UUID,
    question: str,
    start_date: date,
    end_date: date,
    context: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Handle customer-related questions."""
    kpis = await calculate_kpis(db, tenant_id, start_date, end_date)
    
    avg_revenue = kpis["total_revenue"] / kpis["unique_customers"] if kpis["unique_customers"] > 0 else 0
    
    return {
        "summary": f"You have {kpis['unique_customers']} customers with average revenue of ${avg_revenue:,.2f} per customer.",
        "data": {
            "total_customers": kpis["unique_customers"],
            "avg_revenue_per_customer": avg_revenue,
            "total_transactions": kpis["transaction_count"]
        },
        "explanation": "Customer metrics are calculated from transaction data."
    }


async def handle_discontinue_question(
    db: AsyncSession,
    tenant_id: UUID,
    question: str,
    start_date: date,
    end_date: date,
    context: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Handle discontinuation recommendation questions."""
    losses = await get_bottom_offerings(db, tenant_id, start_date, end_date, limit=10)
    
    # Offerings with significant losses
    discontinue_candidates = [
        {
            "offering": item["name"],
            "loss": abs(item["profit"]),
            "recommendation": "Consider discontinuing" if abs(item["profit"]) > 1000 else "Review pricing"
        }
        for item in losses
        if abs(item["profit"]) > 500
    ]
    
    return {
        "summary": f"{len(discontinue_candidates)} offerings are candidates for discontinuation or major revision.",
        "candidates": discontinue_candidates[:5],
        "explanation": "Candidates are selected based on sustained losses that exceed reasonable thresholds."
    }


async def handle_compare_question(
    db: AsyncSession,
    tenant_id: UUID,
    question: str,
    start_date: date,
    end_date: date,
    context: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Handle comparison questions."""
    current_kpis = await calculate_kpis(db, tenant_id, start_date, end_date)
    
    period_days = (end_date - start_date).days
    prev_end = start_date - timedelta(days=1)
    prev_start = prev_end - timedelta(days=period_days)
    
    prev_kpis = await calculate_kpis(db, tenant_id, prev_start, prev_end)
    
    return {
        "summary": "Period comparison analysis:",
        "current_period": {
            "revenue": current_kpis["total_revenue"],
            "profit": current_kpis["total_profit"],
            "margin": current_kpis["profit_margin"],
            "start": str(start_date),
            "end": str(end_date)
        },
        "previous_period": {
            "revenue": prev_kpis["total_revenue"] if prev_kpis else 0,
            "profit": prev_kpis["total_profit"] if prev_kpis else 0,
            "margin": prev_kpis["profit_margin"] if prev_kpis else 0,
            "start": str(prev_start),
            "end": str(prev_end)
        },
        "changes": {
            "revenue_change": current_kpis.get("revenue_trend"),
            "profit_change": current_kpis.get("profit_trend")
        }
    }


async def handle_general_question(
    db: AsyncSession,
    tenant_id: UUID,
    question: str,
    start_date: date,
    end_date: date,
    context: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Handle general questions with overview data."""
    kpis = await calculate_kpis(db, tenant_id, start_date, end_date)
    
    return {
        "summary": "Here's an overview of your business performance:",
        "metrics": {
            "revenue": kpis["total_revenue"],
            "profit": kpis["total_profit"],
            "margin": kpis["profit_margin"],
            "customers": kpis["unique_customers"],
            "transactions": kpis["transaction_count"]
        },
        "suggestion": "Try asking more specific questions like 'Where am I losing money?' or 'Which products should I focus on?'"
    }


def determine_loss_reason(item: Dict[str, Any]) -> str:
    """Determine the likely reason for losses."""
    profit = item.get("profit", 0)
    revenue = item.get("revenue", 0)
    
    if revenue == 0:
        return "No revenue generated"
    
    cost = revenue - profit
    if cost > revenue * 2:
        return "Cost significantly exceeds revenue - pricing issue"
    elif cost > revenue:
        return "Pricing below cost"
    else:
        return "Unknown - requires further analysis"

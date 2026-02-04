# L10: LLM Layer - Explainer
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger()

# Prompt templates for different insight types
EXPLANATION_TEMPLATES = {
    "LOSS_001": """Based on the analysis, this offering is generating losses. 

Key findings:
- Current loss: ${loss:.2f}
- Revenue: ${revenue:.2f}
- Cost: ${cost:.2f}

The unit cost exceeds the selling price, meaning every sale results in a net loss. This could be due to:
1. **Pricing below cost** - The price may have been set without full cost visibility
2. **Hidden overhead** - Indirect costs may not be fully allocated
3. **Volume expectations** - Pricing assumed higher volumes that would reduce unit costs

Recommended action: {recommendation}

Expected impact: {expected_impact}""",

    "LOSS_002": """This offering has a dangerously thin profit margin of {margin:.1f}%.

While technically profitable, margins this low leave no room for:
- Unexpected costs
- Customer discounts
- Market fluctuations

Industry best practice suggests a minimum 15-20% margin for business sustainability.

Recommended action: {recommendation}""",

    "RISK_001": """Revenue concentration risk detected.

{top_customer_pct:.0f}% of this offering's revenue comes from a single customer. This creates significant business risk because:
1. Loss of this customer would severely impact revenue
2. The customer may leverage this dependency for discounts
3. Business planning becomes unpredictable

Recommended action: {recommendation}""",

    "GROWTH_001": """This is a high-potential growth opportunity.

The offering shows:
- Strong margin: {margin:.0f}%
- Low current volume: {transactions} transactions

The healthy margin suggests good product-market fit and pricing power. The low volume indicates untapped market potential.

Recommended action: {recommendation}""",

    "default": """Based on the data analysis, the following insight has been identified:

{description}

Evidence:
{evidence_summary}

Recommended action: {recommendation}

Expected impact: {expected_impact}"""
}


async def generate_explanation(
    insight_type: str,
    title: str,
    evidence: Optional[Dict[str, Any]],
    recommendation: Optional[str]
) -> str:
    """
    Generate a natural language explanation for an insight.
    Uses templates for consistent, high-quality explanations.
    
    In a full implementation, this would use the LLM for dynamic generation.
    This version uses templates for reliability and speed.
    """
    template = EXPLANATION_TEMPLATES.get(insight_type, EXPLANATION_TEMPLATES["default"])
    
    # Extract metrics from evidence
    metrics = evidence.get("metrics", {}) if evidence else {}
    
    try:
        explanation = template.format(
            title=title,
            loss=abs(metrics.get("profit", 0)),
            revenue=metrics.get("revenue", 0),
            cost=metrics.get("revenue", 0) - metrics.get("profit", 0),
            margin=metrics.get("margin", 0),
            transactions=metrics.get("transactions", 0),
            top_customer_pct=metrics.get("top_customer_pct", 0),
            description=title,
            recommendation=recommendation or "Review and take appropriate action.",
            expected_impact="Improve business performance.",
            evidence_summary=format_evidence(evidence)
        )
        return explanation
    except Exception as e:
        logger.warning("Template formatting failed", error=str(e))
        return f"{title}\n\n{recommendation or 'Please review this finding.'}"


def format_evidence(evidence: Optional[Dict[str, Any]]) -> str:
    """Format evidence dictionary for display."""
    if not evidence:
        return "No additional data available."
    
    lines = []
    metrics = evidence.get("metrics", {})
    
    if metrics.get("revenue"):
        lines.append(f"- Revenue: ${metrics['revenue']:,.2f}")
    if metrics.get("profit"):
        lines.append(f"- Profit: ${metrics['profit']:,.2f}")
    if metrics.get("margin"):
        lines.append(f"- Margin: {metrics['margin']:.1f}%")
    if metrics.get("transactions"):
        lines.append(f"- Transactions: {metrics['transactions']:,}")
    
    return "\n".join(lines) if lines else "No metrics available."


async def generate_summary(
    kpis: Dict[str, Any],
    insights: list,
    period_description: str = "this period"
) -> str:
    """Generate an executive summary of the business performance."""
    summary_parts = []
    
    # Revenue summary
    revenue = kpis.get("total_revenue", 0)
    profit = kpis.get("total_profit", 0)
    margin = kpis.get("profit_margin", 0)
    
    summary_parts.append(f"**Performance Summary for {period_description}:**\n")
    summary_parts.append(f"Total revenue was ${revenue:,.2f} with a profit of ${profit:,.2f} ({margin:.1f}% margin).\n")
    
    # Trend summary
    revenue_trend = kpis.get("revenue_trend")
    if revenue_trend is not None:
        if revenue_trend > 0.1:
            summary_parts.append(f"📈 Revenue grew {revenue_trend*100:.0f}% compared to the previous period.\n")
        elif revenue_trend < -0.1:
            summary_parts.append(f"📉 Revenue declined {abs(revenue_trend)*100:.0f}% compared to the previous period.\n")
        else:
            summary_parts.append("📊 Revenue remained relatively stable.\n")
    
    # Critical insights
    critical = [i for i in insights if i.get("severity") in ["critical", "high"]]
    if critical:
        summary_parts.append(f"\n**⚠️ {len(critical)} critical/high-priority items require attention:**\n")
        for insight in critical[:3]:
            summary_parts.append(f"- {insight.get('title')}\n")
    
    # Opportunities
    opportunities = [i for i in insights if i.get("severity") == "opportunity"]
    if opportunities:
        summary_parts.append(f"\n**💡 {len(opportunities)} growth opportunities identified:**\n")
        for opp in opportunities[:2]:
            summary_parts.append(f"- {opp.get('title')}\n")
    
    return "".join(summary_parts)

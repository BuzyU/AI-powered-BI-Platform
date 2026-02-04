# L8: Rule Engine - Business Rules and Recommendations
from typing import Dict, Any, List, Callable
from dataclasses import dataclass
from uuid import UUID
import statistics


@dataclass
class Rule:
    """Business rule definition."""
    id: str
    name: str
    description: str
    severity: str  # critical, high, medium, low, opportunity
    condition: Callable[[Dict, Dict], bool]
    recommendation_template: str
    expected_impact_template: str


# Define business rules
RULES: List[Rule] = [
    Rule(
        id="LOSS_001",
        name="Negative Profit Offering",
        description="Offering is generating losses",
        severity="critical",
        condition=lambda o, m: m.get("profit", 0) < 0,
        recommendation_template="Review pricing or discontinue '{name}'. Current loss: ${loss:.2f}. Consider increasing price by {increase_pct:.0f}% to break even.",
        expected_impact_template="Eliminate ${loss:.2f} in annual losses"
    ),
    Rule(
        id="LOSS_002",
        name="Low Margin Offering",
        description="Offering has very low profit margin",
        severity="high",
        condition=lambda o, m: 0 < m.get("margin", 100) < 5,
        recommendation_template="'{name}' has only {margin:.1f}% margin. Consider price increase or cost reduction.",
        expected_impact_template="Improve margin from {margin:.1f}% to target 15%"
    ),
    Rule(
        id="RISK_001",
        name="High Customer Concentration",
        description="Revenue concentrated in few customers",
        severity="high",
        condition=lambda o, m: m.get("top_customer_pct", 0) > 50,
        recommendation_template="'{name}' has {top_customer_pct:.0f}% revenue from top customer. Diversify customer base.",
        expected_impact_template="Reduce dependency risk"
    ),
    Rule(
        id="RISK_002",
        name="Declining Revenue Trend",
        description="Offering showing declining revenue",
        severity="high",
        condition=lambda o, m: m.get("revenue_trend", 0) < -0.15,
        recommendation_template="'{name}' revenue declined {decline_pct:.0f}%. Investigate cause and consider intervention.",
        expected_impact_template="Stabilize or reverse declining trend"
    ),
    Rule(
        id="RISK_003",
        name="High Volatility",
        description="Offering has unstable revenue",
        severity="medium",
        condition=lambda o, m: m.get("stability_score", 1) < 0.4,
        recommendation_template="'{name}' has unstable revenue (stability: {stability:.0f}%). Consider subscription model or contracts.",
        expected_impact_template="Improve revenue predictability"
    ),
    Rule(
        id="GROWTH_001",
        name="High Margin Low Volume",
        description="Profitable offering with growth potential",
        severity="opportunity",
        condition=lambda o, m: m.get("margin", 0) > 30 and m.get("transactions", 0) < 50,
        recommendation_template="'{name}' has {margin:.0f}% margin but only {transactions} sales. Increase marketing investment.",
        expected_impact_template="Potential ${potential_revenue:.2f} additional revenue with 2x volume"
    ),
    Rule(
        id="GROWTH_002",
        name="High Volume Low Margin",
        description="Popular offering with margin improvement opportunity",
        severity="opportunity",
        condition=lambda o, m: m.get("transactions", 0) > 100 and 5 < m.get("margin", 0) < 15,
        recommendation_template="'{name}' is popular ({transactions} sales) but has only {margin:.0f}% margin. Small price increase could significantly improve profit.",
        expected_impact_template="${margin_gain:.2f} additional profit with 5% price increase"
    ),
    Rule(
        id="CHURN_001",
        name="Customer Churn Risk",
        description="Customers showing reduced activity",
        severity="medium",
        condition=lambda o, m: m.get("repeat_rate_decline", 0) > 0.2,
        recommendation_template="Customer repeat rate declined {decline:.0f}%. Implement retention program.",
        expected_impact_template="Retain at-risk customers worth ${at_risk_revenue:.2f}"
    ),
]


def evaluate_rules(
    offerings: List[Dict[str, Any]],
    metrics: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Evaluate all rules against offerings and metrics.
    Returns list of triggered insights.
    """
    insights = []
    
    for offering in offerings:
        offering_id = offering.get("id")
        offering_metrics = metrics.get(str(offering_id), {})
        
        # Merge offering info with metrics
        combined = {**offering, **offering_metrics}
        
        for rule in RULES:
            try:
                if rule.condition(offering, offering_metrics):
                    insight = create_insight(rule, offering, offering_metrics, combined)
                    insights.append(insight)
            except Exception as e:
                # Skip rules that fail due to missing data
                continue
    
    # Sort by severity
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "opportunity": 4}
    insights.sort(key=lambda x: severity_order.get(x["severity"], 5))
    
    return insights


def create_insight(
    rule: Rule,
    offering: Dict[str, Any],
    metrics: Dict[str, Any],
    combined: Dict[str, Any]
) -> Dict[str, Any]:
    """Create an insight from a triggered rule."""
    # Prepare template variables
    template_vars = {
        "name": offering.get("name", "Unknown"),
        "loss": abs(metrics.get("profit", 0)),
        "margin": metrics.get("margin", 0),
        "transactions": metrics.get("transactions", 0),
        "revenue": metrics.get("revenue", 0),
        "profit": metrics.get("profit", 0),
        "increase_pct": calculate_breakeven_increase(metrics),
        "top_customer_pct": metrics.get("top_customer_pct", 0),
        "decline_pct": abs(metrics.get("revenue_trend", 0) * 100),
        "stability": metrics.get("stability_score", 0) * 100,
        "potential_revenue": metrics.get("revenue", 0),
        "margin_gain": metrics.get("revenue", 0) * 0.05,
    }
    
    # Format recommendation
    try:
        recommendation = rule.recommendation_template.format(**template_vars)
    except:
        recommendation = rule.recommendation_template
    
    try:
        expected_impact = rule.expected_impact_template.format(**template_vars)
    except:
        expected_impact = rule.expected_impact_template
    
    return {
        "insight_type": rule.id,
        "severity": rule.severity,
        "title": f"{rule.name}: {offering.get('name', 'Unknown')}",
        "description": rule.description,
        "recommendation": recommendation,
        "expected_impact": expected_impact,
        "confidence": 0.95,  # Rule-based = high confidence
        "evidence": {
            "offering_id": offering.get("id"),
            "offering_name": offering.get("name"),
            "metrics": {
                "revenue": metrics.get("revenue"),
                "profit": metrics.get("profit"),
                "margin": metrics.get("margin"),
                "transactions": metrics.get("transactions")
            }
        }
    }


def calculate_breakeven_increase(metrics: Dict[str, Any]) -> float:
    """Calculate price increase needed to break even."""
    profit = metrics.get("profit", 0)
    revenue = metrics.get("revenue", 0)
    
    if profit >= 0 or revenue <= 0:
        return 0
    
    # To break even: new_price * volume = cost
    # current: revenue = price * volume
    # loss = revenue - cost, so cost = revenue - profit
    cost = revenue - profit
    if revenue > 0:
        return ((cost / revenue) - 1) * 100
    return 0


def calculate_stability_score(revenue_series: List[float]) -> float:
    """
    Calculate stability score based on revenue variance.
    1.0 = perfectly stable, 0.0 = highly volatile
    """
    if len(revenue_series) < 2:
        return 0.5  # Unknown
    
    mean_rev = statistics.mean(revenue_series)
    if mean_rev == 0:
        return 0.0
    
    std_rev = statistics.stdev(revenue_series)
    cv = std_rev / mean_rev  # Coefficient of variation
    
    # Convert to 0-1 score (lower CV = higher stability)
    stability = max(0, 1 - cv)
    return round(stability, 2)


def calculate_risk_score(
    metrics: Dict[str, Any],
    stability: float
) -> float:
    """
    Calculate overall risk score for an offering.
    0.0 = low risk, 1.0 = high risk
    """
    risk_factors = []
    
    # Profit margin risk
    margin = metrics.get("margin", 0)
    if margin < 0:
        risk_factors.append(0.9)
    elif margin < 10:
        risk_factors.append(0.6)
    elif margin < 20:
        risk_factors.append(0.3)
    else:
        risk_factors.append(0.1)
    
    # Stability risk
    risk_factors.append(1 - stability)
    
    # Customer concentration risk
    top_customer_pct = metrics.get("top_customer_pct", 0)
    if top_customer_pct > 50:
        risk_factors.append(0.8)
    elif top_customer_pct > 30:
        risk_factors.append(0.5)
    else:
        risk_factors.append(0.2)
    
    # Trend risk
    trend = metrics.get("revenue_trend", 0)
    if trend < -0.2:
        risk_factors.append(0.9)
    elif trend < -0.1:
        risk_factors.append(0.6)
    elif trend < 0:
        risk_factors.append(0.3)
    else:
        risk_factors.append(0.1)
    
    return round(statistics.mean(risk_factors), 2)

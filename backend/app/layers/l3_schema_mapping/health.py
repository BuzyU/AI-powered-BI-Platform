# L3: Schema Mapping Layer - Health Score Calculator
from typing import Dict, Any, List


# Required fields by dataset role
REQUIRED_FIELDS = {
    "transactional": ["transaction_date", "amount"],
    "master_entity": [],
    "interaction": ["interaction_content"],
    "financial": ["amount"],
    "operational": [],
    "reference": []
}

# Recommended fields by dataset role
RECOMMENDED_FIELDS = {
    "transactional": ["customer_id", "offering_id", "quantity", "cost"],
    "master_entity": ["offering_name", "offering_category"],
    "interaction": ["customer_id", "sentiment_rating"],
    "financial": ["transaction_date"],
    "operational": [],
    "reference": []
}


async def calculate_health_score(
    mappings: List[Dict[str, Any]],
    dataset_role: str = None
) -> Dict[str, Any]:
    """
    Calculate data health score based on mapping completeness.
    Returns score (0-1), missing required fields, and warnings.
    """
    mapped_fields = {m["canonical_field"] for m in mappings if m.get("canonical_field")}
    all_columns = [m["original_column"] for m in mappings]
    unmapped = [m["original_column"] for m in mappings if not m.get("canonical_field")]
    
    # Base score from mapping coverage
    mapping_coverage = len(mapped_fields) / max(len(all_columns), 1)
    
    # Check required fields
    required = REQUIRED_FIELDS.get(dataset_role, [])
    missing_required = [f for f in required if f not in mapped_fields]
    
    # Check recommended fields
    recommended = RECOMMENDED_FIELDS.get(dataset_role, [])
    missing_recommended = [f for f in recommended if f not in mapped_fields]
    
    # Calculate penalties
    required_penalty = len(missing_required) * 0.2
    recommended_penalty = len(missing_recommended) * 0.05
    
    # Calculate final score
    score = max(0, min(1, mapping_coverage - required_penalty - recommended_penalty))
    
    # Generate warnings
    warnings = []
    
    if missing_required:
        for field in missing_required:
            warnings.append({
                "type": f"missing_{field}",
                "message": f"Required field '{field}' is not mapped.",
                "severity": "error"
            })
    
    if missing_recommended:
        for field in missing_recommended:
            msg = get_warning_message(field)
            warnings.append({
                "type": f"missing_{field}",
                "message": msg,
                "severity": "warning"
            })
    
    # Check for low confidence mappings
    low_confidence = [m for m in mappings if m.get("confidence") and m["confidence"] < 0.6]
    if low_confidence:
        warnings.append({
            "type": "low_confidence_mappings",
            "message": f"{len(low_confidence)} column(s) have low confidence mappings. Please verify.",
            "severity": "warning"
        })
    
    # Check for high unmapped ratio
    if len(unmapped) > len(all_columns) * 0.5:
        warnings.append({
            "type": "high_unmapped_ratio",
            "message": f"{len(unmapped)} columns are not mapped. Some data may not be used.",
            "severity": "info"
        })
    
    return {
        "score": round(score, 2),
        "missing_required": missing_required,
        "missing_recommended": missing_recommended,
        "unmapped": unmapped,
        "warnings": warnings
    }


def get_warning_message(field: str) -> str:
    """Get specific warning message for missing field."""
    messages = {
        "customer_id": "Customer ID not mapped. Customer analytics will be limited.",
        "offering_id": "Product/Service ID not mapped. Offering analytics will be limited.",
        "cost": "Cost column not detected. Profit calculations will be unavailable.",
        "quantity": "Quantity not mapped. Unit economics analysis will be limited.",
        "sentiment_rating": "Sentiment/Rating not found. Satisfaction analysis will be limited.",
    }
    return messages.get(field, f"'{field}' is not mapped.")

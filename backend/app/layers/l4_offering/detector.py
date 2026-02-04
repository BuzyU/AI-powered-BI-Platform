# L4: Offering Detection Layer
import pandas as pd
from typing import Dict, Any, List, Optional
from collections import defaultdict
import re

from app.layers.l1_ingestion.parser import parse_file


# Offering type signals
OFFERING_TYPE_SIGNALS = {
    "physical_product": {
        "keywords": ["product", "item", "goods", "hardware", "equipment", "material"],
        "patterns": {
            "quantity": "integer",  # Whole number quantities
            "has_sku": True,
            "has_inventory": True
        },
        "weight": 1.0
    },
    "service": {
        "keywords": ["service", "consulting", "hours", "session", "support", "labor"],
        "patterns": {
            "quantity": "decimal",  # Fractional hours
            "time_based": True
        },
        "weight": 0.95
    },
    "subscription": {
        "keywords": ["subscription", "plan", "monthly", "annual", "recurring", "membership"],
        "patterns": {
            "recurring_pattern": True,
            "same_customer_multiple": True
        },
        "weight": 0.9
    },
    "event": {
        "keywords": ["event", "ticket", "registration", "conference", "workshop", "training"],
        "patterns": {
            "date_specific": True,
            "limited_quantity": True
        },
        "weight": 0.85
    },
    "bundle": {
        "keywords": ["bundle", "package", "combo", "kit", "set"],
        "patterns": {
            "multiple_components": True
        },
        "weight": 0.8
    },
    "digital_asset": {
        "keywords": ["download", "digital", "software", "license", "access", "ebook"],
        "patterns": {
            "no_shipping": True,
            "instant_delivery": True
        },
        "weight": 0.85
    },
    "license": {
        "keywords": ["license", "seats", "users", "perpetual", "subscription"],
        "patterns": {
            "has_expiration": True,
            "user_based": True
        },
        "weight": 0.8
    },
    "rental": {
        "keywords": ["rental", "lease", "rent", "hire", "temporary"],
        "patterns": {
            "has_return_date": True,
            "duration_based": True
        },
        "weight": 0.75
    },
    "usage_based": {
        "keywords": ["usage", "consumption", "metered", "per-use", "credits"],
        "patterns": {
            "variable_quantity": True,
            "unit_pricing": True
        },
        "weight": 0.75
    }
}


async def detect_offering_types(
    transactions_df: pd.DataFrame,
    offerings_df: Optional[pd.DataFrame] = None,
    offering_id_col: str = "offering_id",
    offering_name_col: str = "offering_name"
) -> Dict[str, Dict[str, Any]]:
    """
    Detect offering types from transaction patterns and names.
    Returns offering_id -> {type, confidence, signals}
    """
    results = {}
    
    # Group transactions by offering
    if offering_id_col not in transactions_df.columns:
        return results
    
    grouped = transactions_df.groupby(offering_id_col)
    
    for offering_id, txns in grouped:
        signals = extract_offering_signals(txns, offering_id_col)
        
        # Get offering name if available
        name = ""
        if offerings_df is not None and offering_name_col in offerings_df.columns:
            matching = offerings_df[offerings_df[offering_id_col] == offering_id]
            if len(matching) > 0:
                name = str(matching.iloc[0].get(offering_name_col, ""))
        elif offering_name_col in txns.columns:
            name = str(txns.iloc[0].get(offering_name_col, ""))
        
        # Score each offering type
        scores = {}
        for otype, config in OFFERING_TYPE_SIGNALS.items():
            score = calculate_type_score(name, signals, config)
            scores[otype] = score * config["weight"]
        
        # Get best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        # Only assign if confidence is reasonable
        if best_score > 0.3:
            results[str(offering_id)] = {
                "type": best_type,
                "confidence": round(min(best_score, 0.99), 2),
                "signals": signals,
                "all_scores": {k: round(v, 3) for k, v in scores.items()}
            }
        else:
            results[str(offering_id)] = {
                "type": "physical_product",  # Default
                "confidence": 0.5,
                "signals": signals,
                "all_scores": {k: round(v, 3) for k, v in scores.items()}
            }
    
    return results


def extract_offering_signals(txns: pd.DataFrame, offering_id_col: str) -> Dict[str, Any]:
    """Extract signals from transaction patterns."""
    signals = {}
    
    # Quantity patterns
    if "quantity" in txns.columns:
        quantities = txns["quantity"].dropna()
        if len(quantities) > 0:
            # Check if quantities are integers
            int_ratio = (quantities == quantities.astype(int)).mean()
            signals["quantity_type"] = "integer" if int_ratio > 0.9 else "decimal"
            signals["avg_quantity"] = quantities.mean()
            signals["quantity_variance"] = quantities.var()
    
    # Customer patterns
    if "customer_id" in txns.columns:
        customers = txns["customer_id"].dropna()
        unique_customers = customers.nunique()
        total_txns = len(customers)
        
        if unique_customers > 0:
            signals["repeat_ratio"] = 1 - (unique_customers / total_txns)
            signals["has_repeat_customers"] = signals["repeat_ratio"] > 0.1
    
    # Date patterns
    if "transaction_date" in txns.columns:
        dates = pd.to_datetime(txns["transaction_date"], errors='coerce').dropna()
        if len(dates) > 1:
            date_diffs = dates.diff().dropna()
            if len(date_diffs) > 0:
                avg_days = date_diffs.mean().days
                signals["avg_days_between"] = avg_days
                signals["regular_interval"] = date_diffs.std().days < 5 if hasattr(date_diffs.std(), 'days') else False
    
    # Amount patterns
    if "amount" in txns.columns:
        amounts = txns["amount"].dropna()
        if len(amounts) > 0:
            signals["avg_amount"] = amounts.mean()
            signals["amount_variance"] = amounts.var()
            signals["consistent_pricing"] = amounts.nunique() < 5
    
    return signals


def calculate_type_score(name: str, signals: Dict[str, Any], config: Dict) -> float:
    """Calculate score for an offering type."""
    score = 0.0
    
    # Keyword matching in name
    name_lower = name.lower()
    keyword_matches = sum(1 for kw in config["keywords"] if kw in name_lower)
    score += (keyword_matches / len(config["keywords"])) * 0.5
    
    # Pattern matching
    patterns = config.get("patterns", {})
    pattern_matches = 0
    pattern_count = len(patterns)
    
    if patterns.get("quantity") == "integer" and signals.get("quantity_type") == "integer":
        pattern_matches += 1
    elif patterns.get("quantity") == "decimal" and signals.get("quantity_type") == "decimal":
        pattern_matches += 1
    
    if patterns.get("recurring_pattern") and signals.get("regular_interval"):
        pattern_matches += 1
    
    if patterns.get("same_customer_multiple") and signals.get("repeat_ratio", 0) > 0.3:
        pattern_matches += 1
    
    if patterns.get("time_based") and signals.get("quantity_type") == "decimal":
        pattern_matches += 1
    
    if pattern_count > 0:
        score += (pattern_matches / pattern_count) * 0.5
    
    return score

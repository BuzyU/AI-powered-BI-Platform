# L2: Classification Layer - Role Detector
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import date
import re
from rapidfuzz import fuzz

from app.layers.l1_ingestion.parser import parse_file, detect_date_columns, detect_numeric_columns


# Signal definitions for role detection
ROLE_SIGNALS = {
    "transactional": {
        "column_patterns": [
            r"(order|transaction|invoice|sale|purchase|txn).*id",
            r"(order|sale|transaction|invoice).*date",
            r"(total|amount|price|revenue|sales|cost)",
            r"quantity|qty|units",
        ],
        "keywords": ["order", "sale", "transaction", "invoice", "purchase", "payment"],
        "characteristics": {
            "has_date": True,
            "has_amount": True,
            "row_granularity": "event"
        },
        "weight": 1.0
    },
    "master_entity": {
        "column_patterns": [
            r"(product|item|service|customer|client|user).*id",
            r"(name|title|description)",
            r"(category|type|class|status)",
            r"(sku|code)",
        ],
        "keywords": ["product", "item", "service", "customer", "master", "catalog"],
        "characteristics": {
            "has_date": False,
            "unique_id_column": True,
            "row_granularity": "entity"
        },
        "weight": 0.9
    },
    "interaction": {
        "column_patterns": [
            r"(ticket|case|support|feedback).*id",
            r"(customer|user|client).*id",
            r"(content|message|description|comment|body)",
            r"(rating|score|satisfaction)",
        ],
        "keywords": ["ticket", "support", "feedback", "complaint", "rating", "review", "case"],
        "characteristics": {
            "has_text_content": True,
            "has_date": True
        },
        "weight": 0.85
    },
    "financial": {
        "column_patterns": [
            r"(cost|expense|budget|payment|invoice)",
            r"(debit|credit|balance)",
            r"(account|ledger)",
        ],
        "keywords": ["cost", "expense", "budget", "payment", "invoice", "ledger", "finance"],
        "characteristics": {
            "has_amount": True,
            "mostly_monetary": True
        },
        "weight": 0.8
    },
    "operational": {
        "column_patterns": [
            r"(inventory|stock|warehouse)",
            r"(shipment|delivery|tracking)",
            r"(schedule|shift|capacity)",
        ],
        "keywords": ["inventory", "stock", "shipment", "delivery", "warehouse", "schedule"],
        "characteristics": {
            "has_quantities": True
        },
        "weight": 0.75
    },
    "reference": {
        "column_patterns": [
            r"(code|id)",
            r"(name|label|description)",
        ],
        "keywords": ["category", "type", "status", "country", "region", "lookup"],
        "characteristics": {
            "small_row_count": True,
            "no_dates": True
        },
        "weight": 0.7
    }
}


async def classify_dataset_role(file_path: str, file_type: str) -> Dict[str, Any]:
    """
    Classify a dataset by its role in the business context.
    Returns role, confidence, and detected characteristics.
    """
    # Parse file
    df = await parse_file(file_path, file_type)
    
    # Extract signals
    columns_lower = [str(c).lower() for c in df.columns]
    
    # Detect column types
    date_columns = detect_date_columns(df)
    numeric_columns = detect_numeric_columns(df)
    
    # Calculate scores for each role
    scores = {}
    
    for role, signals in ROLE_SIGNALS.items():
        score = 0.0
        
        # Check column patterns
        pattern_matches = 0
        for pattern in signals["column_patterns"]:
            for col in columns_lower:
                if re.search(pattern, col):
                    pattern_matches += 1
                    break
        
        score += (pattern_matches / len(signals["column_patterns"])) * 0.4
        
        # Check keywords in column names
        keyword_matches = 0
        for keyword in signals["keywords"]:
            for col in columns_lower:
                if keyword in col:
                    keyword_matches += 1
                    break
        
        score += (keyword_matches / len(signals["keywords"])) * 0.3
        
        # Check characteristics
        char = signals["characteristics"]
        char_matches = 0
        char_count = len(char)
        
        if char.get("has_date") and date_columns:
            char_matches += 1
        elif char.get("no_dates") and not date_columns:
            char_matches += 1
        
        if char.get("has_amount") and numeric_columns:
            char_matches += 1
        
        if char.get("small_row_count") and len(df) < 100:
            char_matches += 1
        
        if char.get("has_text_content"):
            text_cols = [c for c in df.columns if df[c].dtype == 'object' and df[c].str.len().mean() > 50]
            if text_cols:
                char_matches += 1
        
        score += (char_matches / max(char_count, 1)) * 0.3
        
        # Apply weight
        scores[role] = score * signals["weight"]
    
    # Find best match
    best_role = max(scores, key=scores.get)
    confidence = min(scores[best_role], 0.99)
    
    # Detect date range
    date_range_start = None
    date_range_end = None
    
    if date_columns:
        for col in date_columns:
            try:
                dates = pd.to_datetime(df[col], errors='coerce').dropna()
                if len(dates) > 0:
                    date_range_start = dates.min().date()
                    date_range_end = dates.max().date()
                    break
            except:
                pass
    
    # Detect dominant entities
    entities = detect_entities(columns_lower)
    
    return {
        "role": best_role,
        "confidence": round(confidence, 2),
        "date_range_start": date_range_start,
        "date_range_end": date_range_end,
        "entities": entities,
        "scores": {k: round(v, 3) for k, v in scores.items()}
    }


def detect_entities(columns: List[str]) -> List[str]:
    """Detect entity types present in columns."""
    entity_patterns = {
        "customer": ["customer", "client", "buyer", "account", "user"],
        "product": ["product", "item", "sku", "goods"],
        "service": ["service"],
        "order": ["order", "transaction", "sale", "invoice"],
        "ticket": ["ticket", "case", "support"],
        "feedback": ["feedback", "review", "rating"],
    }
    
    detected = []
    for entity, patterns in entity_patterns.items():
        for pattern in patterns:
            for col in columns:
                if pattern in col:
                    if entity not in detected:
                        detected.append(entity)
                    break
    
    return detected

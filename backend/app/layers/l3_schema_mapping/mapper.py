# L3: Schema Mapping Layer - Column Mapper
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from rapidfuzz import fuzz
import re

from app.layers.l1_ingestion.parser import parse_file, detect_date_columns, detect_numeric_columns


# Canonical field definitions with variants
CANONICAL_FIELDS = {
    # Transaction fields
    "transaction_id": {
        "variants": ["id", "order_id", "invoice_id", "txn_id", "transaction_id", "sale_id", "record_id"],
        "type": "id",
        "required_for": ["transactional"]
    },
    "transaction_date": {
        "variants": ["date", "order_date", "sale_date", "transaction_date", "created_at", "invoice_date", "purchase_date"],
        "type": "date",
        "required_for": ["transactional"]
    },
    "amount": {
        "variants": ["amount", "total", "revenue", "price", "value", "sale_amount", "total_amount", "order_total", "grand_total"],
        "type": "numeric",
        "required_for": ["transactional"]
    },
    "quantity": {
        "variants": ["quantity", "qty", "units", "count", "volume", "num_items"],
        "type": "numeric",
        "required_for": []
    },
    "unit_price": {
        "variants": ["unit_price", "price", "rate", "cost_per_unit", "item_price", "selling_price"],
        "type": "numeric",
        "required_for": []
    },
    
    # Customer fields
    "customer_id": {
        "variants": ["customer_id", "client_id", "buyer_id", "account_id", "user_id", "cust_id"],
        "type": "id",
        "required_for": ["transactional"]
    },
    "customer_name": {
        "variants": ["customer_name", "client_name", "buyer_name", "account_name", "name", "full_name"],
        "type": "text",
        "required_for": []
    },
    "customer_email": {
        "variants": ["email", "customer_email", "contact_email", "email_address"],
        "type": "email",
        "required_for": []
    },
    
    # Offering fields
    "offering_id": {
        "variants": ["product_id", "service_id", "item_id", "sku", "offering_id", "prod_id", "item_code"],
        "type": "id",
        "required_for": ["transactional"]
    },
    "offering_name": {
        "variants": ["product_name", "service_name", "item_name", "description", "name", "title", "product", "item"],
        "type": "text",
        "required_for": []
    },
    "offering_category": {
        "variants": ["category", "type", "class", "segment", "product_category", "product_type", "group"],
        "type": "text",
        "required_for": []
    },
    
    # Cost fields
    "cost": {
        "variants": ["cost", "cogs", "expense", "unit_cost", "cost_price", "purchase_cost", "item_cost"],
        "type": "numeric",
        "required_for": []
    },
    
    # Interaction fields
    "interaction_content": {
        "variants": ["content", "message", "body", "text", "description", "comment", "feedback", "review"],
        "type": "text",
        "required_for": ["interaction"]
    },
    "sentiment_rating": {
        "variants": ["rating", "score", "satisfaction", "stars", "sentiment"],
        "type": "numeric",
        "required_for": []
    }
}


async def generate_column_mappings(
    file_path: str, 
    file_type: str,
    columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate column mappings from source to canonical schema.
    Uses name similarity, type inference, and value patterns.
    """
    # Parse file if columns not provided
    if not columns:
        df = await parse_file(file_path, file_type)
        columns = df.columns.tolist()
    else:
        df = await parse_file(file_path, file_type)
    
    mappings = []
    used_canonical = set()
    
    for col in columns:
        best_match = None
        best_score = 0.0
        
        col_lower = str(col).lower().strip()
        
        for canonical, config in CANONICAL_FIELDS.items():
            if canonical in used_canonical:
                continue
            
            # Name similarity
            name_scores = [fuzz.ratio(col_lower, v.lower()) / 100 for v in config["variants"]]
            name_score = max(name_scores)
            
            # Exact match bonus
            if col_lower in [v.lower() for v in config["variants"]]:
                name_score = 1.0
            
            # Type compatibility
            type_score = check_type_compatibility(df[col], config["type"])
            
            # Combined score
            combined = (name_score * 0.6) + (type_score * 0.4)
            
            if combined > best_score and combined > 0.4:
                best_score = combined
                best_match = canonical
        
        if best_match:
            used_canonical.add(best_match)
        
        mappings.append({
            "original_column": col,
            "canonical_field": best_match,
            "confidence": round(best_score, 2) if best_match else None
        })
    
    return {"mappings": mappings}


def check_type_compatibility(series: pd.Series, expected_type: str) -> float:
    """Check how well a pandas series matches the expected type."""
    sample = series.dropna().head(100)
    if len(sample) == 0:
        return 0.5  # Unknown
    
    if expected_type == "id":
        # IDs should be unique or mostly unique
        unique_ratio = len(sample.unique()) / len(sample)
        return 0.8 if unique_ratio > 0.9 else 0.5
    
    elif expected_type == "date":
        if pd.api.types.is_datetime64_any_dtype(series):
            return 1.0
        try:
            parsed = pd.to_datetime(sample, errors='coerce')
            valid_ratio = parsed.notna().sum() / len(sample)
            return valid_ratio
        except:
            return 0.0
    
    elif expected_type == "numeric":
        if pd.api.types.is_numeric_dtype(series):
            return 1.0
        try:
            cleaned = sample.astype(str).str.replace(r'[$,€£]', '', regex=True)
            parsed = pd.to_numeric(cleaned, errors='coerce')
            valid_ratio = parsed.notna().sum() / len(sample)
            return valid_ratio
        except:
            return 0.0
    
    elif expected_type == "email":
        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        matches = sample.astype(str).str.match(email_pattern, na=False)
        return matches.sum() / len(sample)
    
    elif expected_type == "text":
        if series.dtype == 'object':
            avg_len = sample.astype(str).str.len().mean()
            return 0.8 if avg_len > 3 else 0.4
        return 0.3
    
    return 0.5

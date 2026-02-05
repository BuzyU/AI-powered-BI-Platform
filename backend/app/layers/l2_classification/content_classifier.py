# Smart Content Classification Engine
# Detects 30+ content types using keyword density, column patterns, and data signatures

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ContentCategory(Enum):
    """High-level content categories."""
    BUSINESS = "business"
    ANALYTICS = "analytics"
    ML_MODEL = "ml_model"
    LEGAL = "legal"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    UNKNOWN = "unknown"


class ContentType(Enum):
    """Specific content types within categories."""
    # Business Types
    SALES_TRANSACTIONS = "sales_transactions"
    PRODUCTS_CATALOG = "products_catalog"
    SERVICES_DATA = "services_data"
    SUBSCRIPTIONS = "subscriptions"
    CUSTOMERS = "customers"
    EVENTS = "events"
    BUNDLES_PACKAGES = "bundles_packages"
    DIGITAL_ASSETS = "digital_assets"
    LICENSES = "licenses"
    MEMBERSHIPS = "memberships"
    RENTALS = "rentals"
    USAGE_DATA = "usage_data"
    SUPPORT_TICKETS = "support_tickets"
    CUSTOMER_FEEDBACK = "customer_feedback"
    MARKETING_DATA = "marketing_data"
    FINANCIAL_DATA = "financial_data"
    OPERATIONAL_DATA = "operational_data"
    INVENTORY = "inventory"
    ORDERS = "orders"
    INVOICES = "invoices"
    EMPLOYEES = "employees"
    SUPPLIERS = "suppliers"
    
    # Analytics Types
    AI_ANALYTICS = "ai_analytics"
    SURVEY_DATA = "survey_data"
    TIME_SERIES = "time_series"
    METRICS_KPI = "metrics_kpi"
    
    # ML/AI Types
    TRAINING_DATASET = "training_dataset"
    TRANSFORMER_MODEL = "transformer_model"
    CLASSIFICATION_MODEL = "classification_model"
    REGRESSION_MODEL = "regression_model"
    GAN_MODEL = "gan_model"
    NEURAL_NETWORK = "neural_network"
    EMBEDDINGS = "embeddings"
    MODEL_WEIGHTS = "model_weights"
    
    # Legal Types
    LEGAL_CASES = "legal_cases"
    CONTRACTS = "contracts"
    COMPLIANCE_DATA = "compliance_data"
    
    # Healthcare Types
    PATIENT_DATA = "patient_data"
    MEDICAL_RECORDS = "medical_records"
    CLINICAL_TRIALS = "clinical_trials"
    
    # Generic
    REFERENCE_DATA = "reference_data"
    UNKNOWN = "unknown"


@dataclass
class ContentSignature:
    """Defines a signature pattern for content detection."""
    content_type: ContentType
    category: ContentCategory
    required_columns: List[str]  # Must have at least one
    optional_columns: List[str]  # Boost confidence
    keyword_patterns: List[str]  # Regex patterns in values
    value_patterns: Dict[str, str]  # Column -> value pattern
    min_confidence: float = 0.3


# Content Signatures Database
CONTENT_SIGNATURES: List[ContentSignature] = [
    # === BUSINESS DATA ===
    ContentSignature(
        content_type=ContentType.SALES_TRANSACTIONS,
        category=ContentCategory.BUSINESS,
        required_columns=["sale", "transaction", "order", "revenue", "amount", "total", "price", "quantity", "qty"],
        optional_columns=["customer", "product", "date", "invoice", "discount", "tax", "payment", "channel"],
        keyword_patterns=[r"\$\d+", r"USD|EUR|GBP", r"credit|debit|cash"],
        value_patterns={"amount": r"^\d+\.?\d*$", "quantity": r"^\d+$"},
    ),
    ContentSignature(
        content_type=ContentType.PRODUCTS_CATALOG,
        category=ContentCategory.BUSINESS,
        required_columns=["product", "sku", "item", "catalog", "upc", "ean", "asin"],
        optional_columns=["name", "description", "category", "price", "cost", "brand", "weight", "dimensions"],
        keyword_patterns=[r"SKU-\d+", r"PROD-\d+"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.SERVICES_DATA,
        category=ContentCategory.BUSINESS,
        required_columns=["service", "hours", "rate", "billable", "consultation", "session"],
        optional_columns=["client", "provider", "duration", "fee", "booking"],
        keyword_patterns=[r"\d+\s*hrs?", r"\d+\s*hours?", r"per\s*hour"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.SUBSCRIPTIONS,
        category=ContentCategory.BUSINESS,
        required_columns=["subscription", "plan", "recurring", "mrr", "arr", "monthly", "annual", "tier"],
        optional_columns=["subscriber", "start_date", "end_date", "renewal", "churn", "cancel"],
        keyword_patterns=[r"monthly|annual|yearly", r"basic|pro|premium|enterprise"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.CUSTOMERS,
        category=ContentCategory.BUSINESS,
        required_columns=["customer", "client", "account", "user", "member", "contact"],
        optional_columns=["name", "email", "phone", "address", "segment", "tier", "ltv", "lifetime"],
        keyword_patterns=[r"@.*\.(com|org|net)", r"\+?\d{10,}"],
        value_patterns={"email": r"^[^@]+@[^@]+\.[^@]+$"},
    ),
    ContentSignature(
        content_type=ContentType.EVENTS,
        category=ContentCategory.BUSINESS,
        required_columns=["event", "ticket", "attendee", "registration", "venue", "conference"],
        optional_columns=["date", "time", "location", "capacity", "speaker", "session"],
        keyword_patterns=[r"event-\d+", r"ticket-\d+"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.BUNDLES_PACKAGES,
        category=ContentCategory.BUSINESS,
        required_columns=["bundle", "package", "combo", "kit", "set"],
        optional_columns=["items", "components", "discount", "savings"],
        keyword_patterns=[r"bundle|package|combo"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.DIGITAL_ASSETS,
        category=ContentCategory.BUSINESS,
        required_columns=["download", "digital", "file", "asset", "media", "content"],
        optional_columns=["format", "size", "url", "license", "version"],
        keyword_patterns=[r"\.(pdf|mp3|mp4|zip|exe)", r"download"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.LICENSES,
        category=ContentCategory.BUSINESS,
        required_columns=["license", "key", "activation", "seat", "entitlement"],
        optional_columns=["expiry", "valid", "type", "edition"],
        keyword_patterns=[r"[A-Z0-9]{4}-[A-Z0-9]{4}", r"license\s*key"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.MEMBERSHIPS,
        category=ContentCategory.BUSINESS,
        required_columns=["membership", "member", "club", "gym", "association"],
        optional_columns=["tier", "benefits", "dues", "renewal"],
        keyword_patterns=[r"gold|silver|platinum|bronze"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.RENTALS,
        category=ContentCategory.BUSINESS,
        required_columns=["rental", "lease", "rent", "tenant", "property"],
        optional_columns=["duration", "deposit", "return", "checkout", "checkin"],
        keyword_patterns=[r"per\s*(day|week|month)", r"rental\s*period"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.USAGE_DATA,
        category=ContentCategory.BUSINESS,
        required_columns=["usage", "consumption", "meter", "units", "api_calls", "bandwidth"],
        optional_columns=["timestamp", "user", "resource", "quota", "limit"],
        keyword_patterns=[r"\d+\s*(GB|MB|KB|calls|requests)"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.SUPPORT_TICKETS,
        category=ContentCategory.BUSINESS,
        required_columns=["ticket", "case", "issue", "support", "help", "incident"],
        optional_columns=["status", "priority", "assigned", "resolution", "category", "sla"],
        keyword_patterns=[r"TICKET-\d+", r"open|closed|pending|resolved"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.CUSTOMER_FEEDBACK,
        category=ContentCategory.BUSINESS,
        required_columns=["feedback", "review", "rating", "nps", "satisfaction", "survey"],
        optional_columns=["comment", "score", "recommend", "sentiment"],
        keyword_patterns=[r"[1-5]\s*stars?", r"nps|csat"],
        value_patterns={"rating": r"^[1-5]$"},
    ),
    ContentSignature(
        content_type=ContentType.MARKETING_DATA,
        category=ContentCategory.BUSINESS,
        required_columns=["campaign", "marketing", "ad", "impression", "click", "conversion", "lead"],
        optional_columns=["spend", "roi", "ctr", "cpc", "channel", "source"],
        keyword_patterns=[r"utm_", r"campaign-\d+"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.FINANCIAL_DATA,
        category=ContentCategory.BUSINESS,
        required_columns=["finance", "accounting", "ledger", "debit", "credit", "balance", "expense", "income"],
        optional_columns=["account", "journal", "fiscal", "budget", "variance"],
        keyword_patterns=[r"GL-\d+", r"acc-\d+"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.OPERATIONAL_DATA,
        category=ContentCategory.BUSINESS,
        required_columns=["operation", "process", "workflow", "task", "efficiency", "throughput"],
        optional_columns=["cycle_time", "utilization", "capacity", "bottleneck"],
        keyword_patterns=[r"OPS-\d+"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.INVENTORY,
        category=ContentCategory.BUSINESS,
        required_columns=["inventory", "stock", "warehouse", "bin", "location", "quantity_on_hand"],
        optional_columns=["reorder", "safety_stock", "lead_time", "supplier"],
        keyword_patterns=[r"WH-\d+", r"BIN-\d+"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.ORDERS,
        category=ContentCategory.BUSINESS,
        required_columns=["order", "purchase", "po", "requisition"],
        optional_columns=["status", "ship", "delivery", "tracking"],
        keyword_patterns=[r"ORD-\d+", r"PO-\d+"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.INVOICES,
        category=ContentCategory.BUSINESS,
        required_columns=["invoice", "bill", "statement", "due"],
        optional_columns=["paid", "outstanding", "terms", "tax"],
        keyword_patterns=[r"INV-\d+", r"BILL-\d+"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.EMPLOYEES,
        category=ContentCategory.BUSINESS,
        required_columns=["employee", "staff", "worker", "hr", "personnel"],
        optional_columns=["department", "title", "salary", "hire_date", "manager"],
        keyword_patterns=[r"EMP-\d+"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.SUPPLIERS,
        category=ContentCategory.BUSINESS,
        required_columns=["supplier", "vendor", "manufacturer", "partner"],
        optional_columns=["contact", "terms", "rating", "lead_time"],
        keyword_patterns=[r"SUP-\d+", r"VEN-\d+"],
        value_patterns={},
    ),
    
    # === LEGAL DATA ===
    ContentSignature(
        content_type=ContentType.LEGAL_CASES,
        category=ContentCategory.LEGAL,
        required_columns=["case", "verdict", "court", "plaintiff", "defendant", "litigation", "lawsuit"],
        optional_columns=["judge", "attorney", "filing", "docket", "ruling", "settlement"],
        keyword_patterns=[r"case\s*#?\d+", r"v\.\s*", r"plaintiff|defendant"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.CONTRACTS,
        category=ContentCategory.LEGAL,
        required_columns=["contract", "agreement", "terms", "clause", "party", "signatory"],
        optional_columns=["effective_date", "expiry", "renewal", "termination"],
        keyword_patterns=[r"contract-\d+", r"agreement"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.COMPLIANCE_DATA,
        category=ContentCategory.LEGAL,
        required_columns=["compliance", "regulation", "audit", "violation", "policy"],
        optional_columns=["risk", "control", "finding", "remediation"],
        keyword_patterns=[r"SOX|GDPR|HIPAA|PCI"],
        value_patterns={},
    ),
    
    # === ML/AI DATA ===
    ContentSignature(
        content_type=ContentType.TRAINING_DATASET,
        category=ContentCategory.ML_MODEL,
        required_columns=["label", "target", "class", "feature", "x_train", "y_train"],
        optional_columns=["split", "fold", "weight", "sample"],
        keyword_patterns=[r"train|test|valid", r"feature_\d+"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.EMBEDDINGS,
        category=ContentCategory.ML_MODEL,
        required_columns=["embedding", "vector", "dimension", "latent"],
        optional_columns=["token", "word", "sentence"],
        keyword_patterns=[r"dim_\d+", r"vec_\d+"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.METRICS_KPI,
        category=ContentCategory.ANALYTICS,
        required_columns=["metric", "kpi", "measure", "indicator", "score"],
        optional_columns=["target", "actual", "variance", "trend"],
        keyword_patterns=[r"YoY|MoM|QoQ", r"\d+%"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.TIME_SERIES,
        category=ContentCategory.ANALYTICS,
        required_columns=["timestamp", "datetime", "date", "time", "period"],
        optional_columns=["value", "measure", "forecast", "actual"],
        keyword_patterns=[r"\d{4}-\d{2}-\d{2}", r"\d{2}/\d{2}/\d{4}"],
        value_patterns={},
    ),
    ContentSignature(
        content_type=ContentType.SURVEY_DATA,
        category=ContentCategory.ANALYTICS,
        required_columns=["response", "question", "answer", "respondent", "survey"],
        optional_columns=["scale", "choice", "open_ended"],
        keyword_patterns=[r"Q\d+", r"strongly\s*(agree|disagree)"],
        value_patterns={},
    ),
]


class SmartContentClassifier:
    """
    Intelligent content classification engine.
    Analyzes uploaded data to determine its type, role, and category.
    """
    
    def __init__(self):
        self.signatures = CONTENT_SIGNATURES
        
    def classify(self, df: pd.DataFrame, filename: str = "") -> Dict[str, Any]:
        """
        Main classification method.
        Returns detailed classification with confidence scores.
        """
        if df is None or df.empty:
            return self._unknown_result()
        
        # Collect all signals
        column_scores = self._analyze_columns(df)
        value_scores = self._analyze_values(df)
        pattern_scores = self._analyze_patterns(df)
        filename_hints = self._analyze_filename(filename)
        
        # Combine scores and find best match
        combined_scores = self._combine_scores(
            column_scores, value_scores, pattern_scores, filename_hints
        )
        
        # Get top matches
        sorted_matches = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_matches or sorted_matches[0][1] < 0.2:
            return self._unknown_result()
        
        best_match = sorted_matches[0]
        content_type = best_match[0]
        confidence = min(best_match[1] * 100, 99)  # Cap at 99%
        
        # Find signature for additional info
        signature = self._get_signature(content_type)
        category = signature.category if signature else ContentCategory.UNKNOWN
        
        # Determine data role
        role = self._determine_role(df, content_type)
        
        # Generate human-readable description
        description = self._generate_description(content_type, df, confidence)
        
        return {
            "content_type": content_type.value,
            "category": category.value,
            "confidence": round(confidence, 1),
            "role": role,
            "description": description,
            "alternate_matches": [
                {"type": m[0].value, "confidence": round(m[1] * 100, 1)}
                for m in sorted_matches[1:4] if m[1] > 0.15
            ],
            "detected_entities": self._detect_entities(df),
            "time_coverage": self._detect_time_coverage(df),
            "signals": {
                "column_match": round(column_scores.get(content_type, 0) * 100, 1),
                "value_match": round(value_scores.get(content_type, 0) * 100, 1),
                "pattern_match": round(pattern_scores.get(content_type, 0) * 100, 1),
            }
        }
    
    def _analyze_columns(self, df: pd.DataFrame) -> Dict[ContentType, float]:
        """Analyze column names for signature matches."""
        scores = {}
        columns_lower = [c.lower().replace("_", " ").replace("-", " ") for c in df.columns]
        columns_text = " ".join(columns_lower)
        
        for sig in self.signatures:
            score = 0.0
            required_matches = 0
            optional_matches = 0
            
            # Check required columns
            for req in sig.required_columns:
                if any(req in col for col in columns_lower) or req in columns_text:
                    required_matches += 1
            
            # Check optional columns
            for opt in sig.optional_columns:
                if any(opt in col for col in columns_lower) or opt in columns_text:
                    optional_matches += 1
            
            if required_matches > 0:
                # Base score from required matches
                score = required_matches / len(sig.required_columns) * 0.7
                # Bonus from optional matches
                if sig.optional_columns:
                    score += (optional_matches / len(sig.optional_columns)) * 0.3
                
            scores[sig.content_type] = score
        
        return scores
    
    def _analyze_values(self, df: pd.DataFrame) -> Dict[ContentType, float]:
        """Analyze actual data values for patterns."""
        scores = {}
        
        # Sample first 50 rows for efficiency
        sample = df.head(50)
        sample_text = sample.astype(str).values.flatten()
        sample_text = " ".join([str(v) for v in sample_text if pd.notna(v)])
        
        for sig in self.signatures:
            score = 0.0
            pattern_matches = 0
            
            for pattern in sig.keyword_patterns:
                try:
                    if re.search(pattern, sample_text, re.IGNORECASE):
                        pattern_matches += 1
                except:
                    pass
            
            if sig.keyword_patterns:
                score = pattern_matches / len(sig.keyword_patterns)
            
            scores[sig.content_type] = score
        
        return scores
    
    def _analyze_patterns(self, df: pd.DataFrame) -> Dict[ContentType, float]:
        """Analyze data type patterns and distributions."""
        scores = {}
        
        # Determine data characteristics
        num_cols = df.select_dtypes(include=[np.number]).columns
        date_cols = []
        for col in df.columns:
            try:
                pd.to_datetime(df[col].head(10), errors='raise')
                date_cols.append(col)
            except:
                pass
        
        has_dates = len(date_cols) > 0
        has_numbers = len(num_cols) > 0
        has_ids = any("id" in c.lower() for c in df.columns)
        
        for sig in self.signatures:
            score = 0.0
            
            # Time series needs dates
            if sig.content_type == ContentType.TIME_SERIES and has_dates:
                score += 0.5
            
            # Financial data needs numbers
            if sig.content_type in [ContentType.SALES_TRANSACTIONS, ContentType.FINANCIAL_DATA]:
                if has_numbers:
                    score += 0.3
            
            # Relational data needs IDs
            if sig.content_type in [ContentType.CUSTOMERS, ContentType.ORDERS]:
                if has_ids:
                    score += 0.2
            
            scores[sig.content_type] = scores.get(sig.content_type, 0) + score
        
        return scores
    
    def _analyze_filename(self, filename: str) -> Dict[ContentType, float]:
        """Extract hints from filename."""
        scores = {}
        name_lower = filename.lower()
        
        filename_hints = {
            "sales": ContentType.SALES_TRANSACTIONS,
            "transaction": ContentType.SALES_TRANSACTIONS,
            "order": ContentType.ORDERS,
            "customer": ContentType.CUSTOMERS,
            "product": ContentType.PRODUCTS_CATALOG,
            "service": ContentType.SERVICES_DATA,
            "subscription": ContentType.SUBSCRIPTIONS,
            "ticket": ContentType.SUPPORT_TICKETS,
            "support": ContentType.SUPPORT_TICKETS,
            "feedback": ContentType.CUSTOMER_FEEDBACK,
            "review": ContentType.CUSTOMER_FEEDBACK,
            "inventory": ContentType.INVENTORY,
            "invoice": ContentType.INVOICES,
            "employee": ContentType.EMPLOYEES,
            "hr": ContentType.EMPLOYEES,
            "marketing": ContentType.MARKETING_DATA,
            "campaign": ContentType.MARKETING_DATA,
            "finance": ContentType.FINANCIAL_DATA,
            "legal": ContentType.LEGAL_CASES,
            "case": ContentType.LEGAL_CASES,
            "contract": ContentType.CONTRACTS,
            "train": ContentType.TRAINING_DATASET,
            "model": ContentType.MODEL_WEIGHTS,
            "embed": ContentType.EMBEDDINGS,
        }
        
        for hint, content_type in filename_hints.items():
            if hint in name_lower:
                scores[content_type] = scores.get(content_type, 0) + 0.3
        
        return scores
    
    def _combine_scores(self, *score_dicts) -> Dict[ContentType, float]:
        """Combine multiple score dictionaries."""
        combined = {}
        
        for scores in score_dicts:
            for content_type, score in scores.items():
                combined[content_type] = combined.get(content_type, 0) + score
        
        # Normalize
        if combined:
            max_score = max(combined.values())
            if max_score > 0:
                combined = {k: v / max_score for k, v in combined.items()}
        
        return combined
    
    def _get_signature(self, content_type: ContentType) -> Optional[ContentSignature]:
        """Get signature for a content type."""
        for sig in self.signatures:
            if sig.content_type == content_type:
                return sig
        return None
    
    def _determine_role(self, df: pd.DataFrame, content_type: ContentType) -> str:
        """Determine the data role (transactional, master, etc.)."""
        role_mapping = {
            ContentType.SALES_TRANSACTIONS: "transactional",
            ContentType.ORDERS: "transactional",
            ContentType.INVOICES: "transactional",
            ContentType.SUPPORT_TICKETS: "interaction",
            ContentType.CUSTOMER_FEEDBACK: "interaction",
            ContentType.CUSTOMERS: "master_entity",
            ContentType.PRODUCTS_CATALOG: "master_entity",
            ContentType.EMPLOYEES: "master_entity",
            ContentType.SUPPLIERS: "master_entity",
            ContentType.FINANCIAL_DATA: "financial",
            ContentType.INVENTORY: "operational",
            ContentType.OPERATIONAL_DATA: "operational",
            ContentType.SUBSCRIPTIONS: "transactional",
            ContentType.USAGE_DATA: "transactional",
            ContentType.MARKETING_DATA: "analytical",
            ContentType.METRICS_KPI: "analytical",
            ContentType.TIME_SERIES: "analytical",
            ContentType.TRAINING_DATASET: "ml_data",
            ContentType.EMBEDDINGS: "ml_data",
            ContentType.LEGAL_CASES: "legal",
            ContentType.CONTRACTS: "legal",
        }
        return role_mapping.get(content_type, "reference")
    
    def _detect_entities(self, df: pd.DataFrame) -> List[str]:
        """Detect dominant entities in the data."""
        entities = []
        columns_lower = [c.lower() for c in df.columns]
        
        entity_keywords = {
            "customer": ["customer", "client", "user", "account", "buyer"],
            "product": ["product", "item", "sku", "goods"],
            "order": ["order", "transaction", "purchase"],
            "employee": ["employee", "staff", "worker"],
            "ticket": ["ticket", "case", "issue"],
            "invoice": ["invoice", "bill"],
        }
        
        for entity, keywords in entity_keywords.items():
            if any(any(kw in col for kw in keywords) for col in columns_lower):
                entities.append(entity)
        
        return entities
    
    def _detect_time_coverage(self, df: pd.DataFrame) -> Optional[Dict[str, str]]:
        """Detect date range in the data."""
        for col in df.columns:
            try:
                dates = pd.to_datetime(df[col], errors='coerce')
                valid_dates = dates.dropna()
                if len(valid_dates) > 0:
                    return {
                        "start": valid_dates.min().strftime("%Y-%m-%d"),
                        "end": valid_dates.max().strftime("%Y-%m-%d"),
                        "column": col
                    }
            except:
                pass
        return None
    
    def _generate_description(self, content_type: ContentType, df: pd.DataFrame, confidence: float) -> str:
        """Generate human-readable description."""
        descriptions = {
            ContentType.SALES_TRANSACTIONS: "Sales/Transaction Data - Contains purchase records with revenue information",
            ContentType.PRODUCTS_CATALOG: "Product Catalog - Master list of products/items with attributes",
            ContentType.SERVICES_DATA: "Services Data - Service offerings with pricing and duration",
            ContentType.SUBSCRIPTIONS: "Subscription Data - Recurring revenue and subscriber information",
            ContentType.CUSTOMERS: "Customer Master Data - Customer profiles and contact information",
            ContentType.SUPPORT_TICKETS: "Support Tickets - Customer service cases and issues",
            ContentType.CUSTOMER_FEEDBACK: "Customer Feedback - Reviews, ratings, and satisfaction data",
            ContentType.FINANCIAL_DATA: "Financial Data - Accounting, ledger, or budget information",
            ContentType.INVENTORY: "Inventory Data - Stock levels and warehouse information",
            ContentType.ORDERS: "Order Data - Purchase orders and fulfillment records",
            ContentType.MARKETING_DATA: "Marketing Data - Campaign performance and lead information",
            ContentType.LEGAL_CASES: "Legal Case Data - Litigation, verdicts, and court records",
            ContentType.TRAINING_DATASET: "ML Training Dataset - Labeled data for machine learning",
            ContentType.TIME_SERIES: "Time Series Data - Sequential measurements over time",
        }
        
        base_desc = descriptions.get(content_type, f"{content_type.value.replace('_', ' ').title()} Data")
        return f"{base_desc} ({df.shape[0]:,} rows, {df.shape[1]} columns)"
    
    def _unknown_result(self) -> Dict[str, Any]:
        """Return result for unclassified data."""
        return {
            "content_type": ContentType.UNKNOWN.value,
            "category": ContentCategory.UNKNOWN.value,
            "confidence": 0,
            "role": "unknown",
            "description": "Unable to classify data type",
            "alternate_matches": [],
            "detected_entities": [],
            "time_coverage": None,
            "signals": {}
        }


# Model File Classifier
class ModelFileClassifier:
    """Classifies ML model files based on structure and metadata."""
    
    MODEL_SIGNATURES = {
        "transformer": {
            "keys": ["attention", "layer_norm", "encoder", "decoder", "transformer", "bert", "gpt", "embedding"],
            "type": ContentType.TRANSFORMER_MODEL,
        },
        "classification": {
            "keys": ["classifier", "fc", "softmax", "logits", "num_classes"],
            "type": ContentType.CLASSIFICATION_MODEL,
        },
        "gan": {
            "keys": ["generator", "discriminator", "latent", "gan"],
            "type": ContentType.GAN_MODEL,
        },
        "cnn": {
            "keys": ["conv", "pool", "batch_norm", "relu"],
            "type": ContentType.NEURAL_NETWORK,
        },
    }
    
    def classify_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a model file based on its structure."""
        layers = model_info.get("layers", [])
        layer_names = " ".join([str(l).lower() for l in layers])
        
        scores = {}
        
        for model_type, signature in self.MODEL_SIGNATURES.items():
            score = sum(1 for key in signature["keys"] if key in layer_names)
            if score > 0:
                scores[signature["type"]] = score / len(signature["keys"])
        
        if not scores:
            return {
                "content_type": ContentType.MODEL_WEIGHTS.value,
                "category": ContentCategory.ML_MODEL.value,
                "confidence": 50,
                "model_type": "generic_neural_network",
                "description": "Neural network model weights"
            }
        
        best_match = max(scores.items(), key=lambda x: x[1])
        
        return {
            "content_type": best_match[0].value,
            "category": ContentCategory.ML_MODEL.value,
            "confidence": round(best_match[1] * 100, 1),
            "model_type": best_match[0].value,
            "description": f"{best_match[0].value.replace('_', ' ').title()} detected"
        }


# Singleton instance
classifier = SmartContentClassifier()
model_classifier = ModelFileClassifier()


def classify_content(df: pd.DataFrame, filename: str = "") -> Dict[str, Any]:
    """Convenience function for content classification."""
    return classifier.classify(df, filename)


def classify_model(model_info: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for model classification."""
    return model_classifier.classify_model(model_info)

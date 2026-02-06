# Persona Detection Engine
# Detects WHO is using the system based on WHAT they uploaded

from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import pandas as pd
import re
from app.layers.l2_classification.content_classifier import classifier, ContentType, ContentCategory


class UserPersona(Enum):
    """Primary user personas based on upload patterns."""
    BUSINESS = "business"           # Sales, CRM, Finance, Operations
    ANALYTICS = "analytics"         # EDA, Research, Statistical Analysis
    ML_ENGINEER = "ml_engineer"     # Models, Predictions, Training Data
    DATA_SCIENTIST = "data_scientist"  # Mix of analytics + ML
    DEVELOPER = "developer"         # API logs, System data
    UNKNOWN = "unknown"


class DataCategory(Enum):
    """What type of data was uploaded."""
    # Business Data
    SALES_DATA = "sales_data"
    CUSTOMER_DATA = "customer_data"
    PRODUCT_DATA = "product_data"
    FINANCIAL_DATA = "financial_data"
    HR_DATA = "hr_data"
    INVENTORY_DATA = "inventory_data"
    MARKETING_DATA = "marketing_data"
    
    # Analytics Data
    SURVEY_DATA = "survey_data"
    RESEARCH_DATA = "research_data"
    TIME_SERIES = "time_series"
    STATISTICAL_DATA = "statistical_data"
    GEOGRAPHIC_DATA = "geographic_data"
    SENSOR_DATA = "sensor_data"
    
    # ML Data
    TRAINING_DATA = "training_data"
    PREDICTIONS = "predictions"
    MODEL_FILE = "model_file"
    EMBEDDINGS = "embeddings"
    FEATURE_DATA = "feature_data"
    
    # Generic
    TABULAR_DATA = "tabular_data"
    UNKNOWN = "unknown"


class DashboardType(str, Enum):
    """Type of dashboard layout/template."""
    BUSINESS = "power_bi_style"
    ANALYTICS = "eda_analytics"
    ML_METRICS = "ml_metrics"
    CV_DASHBOARD = "cv_dashboard"
    DATA_SCIENCE = "data_science"
    DEVELOPER = "developer_dashboard"
    GENERAL = "general_dashboard"
    MODEL = "ml_model_dashboard"


@dataclass
class PersonaDetectionResult:
    """Result of persona detection."""
    persona: UserPersona
    confidence: float
    data_categories: List[DataCategory]
    detected_domains: List[str]
    recommended_analysis: List[str]
    dashboard_type: DashboardType
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'persona': self.persona.value if isinstance(self.persona, Enum) else self.persona,
            'confidence': self.confidence,
            'data_categories': [c.value if isinstance(c, Enum) else c for c in self.data_categories],
            'detected_domains': self.detected_domains,
            'recommended_analysis': self.recommended_analysis,
            'dashboard_type': self.dashboard_type.value if isinstance(self.dashboard_type, Enum) else self.dashboard_type,
            'summary': self.summary
        }


class PersonaDetector:
    """
    Detects the user persona and data category based on uploaded files.
    This determines what kind of dashboard and analysis to show.
    """
    
    def __init__(self):
        self.business_indicators = self._init_business_indicators()
        self.analytics_indicators = self._init_analytics_indicators()
        self.ml_indicators = self._init_ml_indicators()
    
    def _init_business_indicators(self) -> Dict[str, List[str]]:
        """Business-related column/filename patterns."""
        return {
            'sales': ['sale', 'revenue', 'order', 'transaction', 'invoice', 'deal', 'booking'],
            'customer': ['customer', 'client', 'buyer', 'subscriber', 'member', 'user_id', 'account'],
            'product': ['product', 'item', 'sku', 'catalog', 'inventory', 'stock', 'merchandise'],
            'financial': ['profit', 'cost', 'expense', 'budget', 'balance', 'payment', 'invoice', 'tax'],
            'hr': ['employee', 'salary', 'department', 'hire', 'position', 'payroll', 'attendance'],
            'marketing': ['campaign', 'lead', 'conversion', 'click', 'impression', 'ad_', 'promo'],
            'operations': ['branch', 'store', 'warehouse', 'location', 'region', 'territory', 'supply']
        }
    
    def _init_analytics_indicators(self) -> Dict[str, List[str]]:
        """Analytics/Research data patterns."""
        return {
            'survey': ['response', 'survey', 'rating', 'score', 'feedback', 'opinion', 'satisfaction'],
            'research': ['experiment', 'control', 'treatment', 'sample', 'observation', 'study'],
            'time_series': ['timestamp', 'datetime', 'date', 'time', 'period', 'interval', 'frequency'],
            'geographic': ['latitude', 'longitude', 'lat', 'lng', 'geo', 'location', 'city', 'country', 'state'],
            'sensor': ['sensor', 'reading', 'measurement', 'temperature', 'humidity', 'pressure', 'aqi', 'pollution'],
            'statistical': ['mean', 'median', 'std', 'variance', 'correlation', 'distribution']
        }
    
    def _init_ml_indicators(self) -> Dict[str, List[str]]:
        """ML/Model related patterns."""
        return {
            'training': ['feature', 'label', 'target', 'train', 'test', 'validation', 'split'],
            'prediction': ['prediction', 'predicted', 'probability', 'confidence', 'score', 'output'],
            'classification': ['class', 'category', 'label', 'y_true', 'y_pred', 'classification'],
            'regression': ['actual', 'predicted', 'error', 'residual', 'mse', 'rmse', 'mae'],
            'embedding': ['embedding', 'vector', 'representation', 'encoding', 'latent'],
            'model_meta': ['epoch', 'loss', 'accuracy', 'precision', 'recall', 'f1', 'auc']
        }
    
    def detect_persona(
        self,
        datasets: List[Dict[str, Any]],
        classifications: List[Dict[str, Any]] = None,
        model_info: Optional[Dict[str, Any]] = None
    ) -> PersonaDetectionResult:
        """
        Detect user persona based on uploaded datasets.
        
        Args:
            datasets: List of dataset info with 'filename', 'df', 'metadata'
            classifications: Optional existing classifications
            model_info: Optional detected model information
        
        Returns:
            PersonaDetectionResult with persona and recommendations
        """
        scores = {
            UserPersona.BUSINESS: 0.0,
            UserPersona.ANALYTICS: 0.0,
            UserPersona.ML_ENGINEER: 0.0,
            UserPersona.DATA_SCIENTIST: 0.0,
            UserPersona.DEVELOPER: 0.0
        }
        
        detected_categories = []
        detected_domains = []
        
        # Boost ML score if model info is present
        if model_info:
            scores[UserPersona.ML_ENGINEER] += 5.0
            detected_categories.append(DataCategory.MODEL_FILE)
            detected_domains.append('Machine Learning')
            if 'model_type' in model_info:
                 detected_domains.append(f"Model/{model_info['model_type']}")

        for ds in datasets:
            filename = ds.get('filename', '').lower()
            metadata = ds.get('metadata', {})
            df = ds.get('df')
            
            # Check if it's a model file
            if metadata.get('is_model'):
                scores[UserPersona.ML_ENGINEER] += 5.0
                detected_categories.append(DataCategory.MODEL_FILE)
                detected_domains.append('Machine Learning')
                continue
            
            if df is None:
                continue
            
            # --- INTENT ENGINE INTEGRATION ---
            # Use SmartContentClassifier to detect specific content type
            classification = classifier.classify(df, filename)
            content_type_str = classification.get('content_type')
            confidence = classification.get('confidence', 0)
            
            # Map ContentType to UserPersona
            try:
                content_type = ContentType(content_type_str)
            except ValueError:
                content_type = ContentType.UNKNOWN

            # Business Data Types
            if content_type in [
                ContentType.SALES_TRANSACTIONS, ContentType.PRODUCTS_CATALOG, 
                ContentType.SERVICES_DATA, ContentType.SUBSCRIPTIONS, ContentType.CUSTOMERS,
                ContentType.EVENTS, ContentType.BUNDLES_PACKAGES, ContentType.LICENSES,
                ContentType.MEMBERSHIPS, ContentType.RENTALS, ContentType.INVOICES,
                ContentType.ORDERS, ContentType.EMPLOYEES, ContentType.SUPPLIERS,
                ContentType.FINANCIAL_DATA, ContentType.INVENTORY, ContentType.OPERATIONAL_DATA,
                ContentType.MARKETING_DATA, ContentType.LEGAL_CASES, ContentType.CONTRACTS
            ]:
                # Use confidence to weight the score
                weight = confidence / 100.0 * 2.5
                scores[UserPersona.BUSINESS] += weight
                detected_domains.append(f"Business/{content_type.value.replace('_', ' ').title()}")
                detected_categories.append(self._domain_to_category(content_type.value, 'business'))

            # Analytics Data Types
            elif content_type in [
                ContentType.METRICS_KPI, ContentType.TIME_SERIES, ContentType.SURVEY_DATA,
                ContentType.CUSTOMER_FEEDBACK, ContentType.AI_ANALYTICS
            ]:
                weight = confidence / 100.0 * 2.5
                scores[UserPersona.ANALYTICS] += weight
                detected_domains.append(f"Analytics/{content_type.value.replace('_', ' ').title()}")
                detected_categories.append(self._domain_to_category(content_type.value, 'analytics'))
            
            # ML Data Types
            elif content_type in [
                ContentType.TRAINING_DATASET, ContentType.EMBEDDINGS, ContentType.MODEL_WEIGHTS,
                ContentType.TRANSFORMER_MODEL, ContentType.CLASSIFICATION_MODEL, ContentType.REGRESSION_MODEL
            ]:
                weight = confidence / 100.0 * 3.0
                scores[UserPersona.ML_ENGINEER] += weight
                detected_domains.append(f"ML/{content_type.value.replace('_', ' ').title()}")
                detected_categories.append(self._domain_to_category(content_type.value, 'ml'))
            
            # Support/Dev Types
            elif content_type in [ContentType.SUPPORT_TICKETS, ContentType.USAGE_DATA]:
                 scores[UserPersona.BUSINESS] += confidence / 100.0
                 scores[UserPersona.DEVELOPER] += confidence / 100.0
                 detected_domains.append(f"Service/{content_type.value.replace('_', ' ').title()}")

            # Fallback for Unknown (legacy keyword check as backup)
            if content_type == ContentType.UNKNOWN:
                columns = [c.lower() for c in df.columns]
                # Score business indicators
                for domain, keywords in self.business_indicators.items():
                    matches = sum(1 for kw in keywords for col in columns if kw in col)
                    filename_matches = sum(1 for kw in keywords if kw in filename)
                    if matches > 0 or filename_matches > 0:
                        scores[UserPersona.BUSINESS] += matches * 0.5 + filename_matches * 1.0
                        detected_domains.append(f"Business/{domain.title()}")
                        detected_categories.append(self._domain_to_category(domain, 'business'))
                
                # Score analytics indicators
                for domain, keywords in self.analytics_indicators.items():
                    matches = sum(1 for kw in keywords for col in columns if kw in col)
                    filename_matches = sum(1 for kw in keywords if kw in filename)
                    if matches > 0 or filename_matches > 0:
                        scores[UserPersona.ANALYTICS] += matches * 0.5 + filename_matches * 1.0
                        detected_domains.append(f"Analytics/{domain.title()}")
                        detected_categories.append(self._domain_to_category(domain, 'analytics'))
                
                # Score ML indicators
                for domain, keywords in self.ml_indicators.items():
                    matches = sum(1 for kw in keywords for col in columns if kw in col)
                    filename_matches = sum(1 for kw in keywords if kw in filename)
                    if matches > 0 or filename_matches > 0:
                        scores[UserPersona.ML_ENGINEER] += matches * 0.5 + filename_matches * 1.0
                        detected_domains.append(f"ML/{domain.title()}")
                        detected_categories.append(self._domain_to_category(domain, 'ml'))
            
            # Check for mixed signals (Data Scientist)
            if scores[UserPersona.ANALYTICS] > 0 and scores[UserPersona.ML_ENGINEER] > 0:
                scores[UserPersona.DATA_SCIENTIST] = (scores[UserPersona.ANALYTICS] + scores[UserPersona.ML_ENGINEER]) * 0.8
        
        # Determine primary persona
        if not any(scores.values()):
            # Default to analytics if nothing detected
            scores[UserPersona.ANALYTICS] = 1.0
            detected_categories.append(DataCategory.TABULAR_DATA)
        
        primary_persona = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[primary_persona] / total_score if total_score > 0 else 0.5
        
        # Get recommendations
        recommended = self._get_recommendations(primary_persona, detected_categories)
        dashboard_type = self._get_dashboard_type(primary_persona, detected_categories)
        summary = self._generate_summary(primary_persona, detected_domains, datasets)
        
        return PersonaDetectionResult(
            persona=primary_persona,
            confidence=min(confidence, 1.0),
            data_categories=list(set(detected_categories)),
            detected_domains=list(set(detected_domains)),
            recommended_analysis=recommended,
            dashboard_type=dashboard_type,
            summary=summary
        )
    
    def _domain_to_category(self, domain: str, category_type: str) -> DataCategory:
        """Map domain to data category."""
        mapping = {
            'business': {
                'sales': DataCategory.SALES_DATA,
                'customer': DataCategory.CUSTOMER_DATA,
                'product': DataCategory.PRODUCT_DATA,
                'financial': DataCategory.FINANCIAL_DATA,
                'hr': DataCategory.HR_DATA,
                'marketing': DataCategory.MARKETING_DATA,
                'operations': DataCategory.INVENTORY_DATA,
            },
            'analytics': {
                'survey': DataCategory.SURVEY_DATA,
                'research': DataCategory.RESEARCH_DATA,
                'time_series': DataCategory.TIME_SERIES,
                'geographic': DataCategory.GEOGRAPHIC_DATA,
                'sensor': DataCategory.SENSOR_DATA,
                'statistical': DataCategory.STATISTICAL_DATA,
            },
            'ml': {
                'training': DataCategory.TRAINING_DATA,
                'prediction': DataCategory.PREDICTIONS,
                'classification': DataCategory.TRAINING_DATA,
                'regression': DataCategory.PREDICTIONS,
                'embedding': DataCategory.EMBEDDINGS,
                'model_meta': DataCategory.FEATURE_DATA,
            }
        }
        # Try finding partial match if exact key missing
        if domain not in mapping.get(category_type, {}):
             for key, val in mapping.get(category_type, {}).items():
                 if key in domain:
                     return val
        return mapping.get(category_type, {}).get(domain, DataCategory.UNKNOWN)
    
    def _get_recommendations(self, persona: UserPersona, categories: List[DataCategory]) -> List[str]:
        """Get recommended analysis based on persona."""
        recommendations = {
            UserPersona.BUSINESS: [
                "Revenue & Profit Analysis",
                "Customer Segmentation",
                "Sales Trends & Forecasting",
                "Product Performance",
                "Regional/Branch Comparison",
                "KPI Dashboard"
            ],
            UserPersona.ANALYTICS: [
                "Exploratory Data Analysis",
                "Statistical Summary",
                "Correlation Analysis",
                "Distribution Analysis",
                "Outlier Detection",
                "Trend Analysis"
            ],
            UserPersona.ML_ENGINEER: [
                "Model Performance Metrics",
                "Confusion Matrix",
                "Feature Importance",
                "Prediction Distribution",
                "Error Analysis",
                "Model Comparison"
            ],
            UserPersona.DATA_SCIENTIST: [
                "EDA & Profiling",
                "Feature Engineering Ideas",
                "Statistical Tests",
                "Model Baseline",
                "Data Quality Report",
                "Hypothesis Testing"
            ],
            UserPersona.DEVELOPER: [
                "Log Analysis",
                "Error Rate Tracking",
                "Performance Metrics",
                "API Usage Patterns"
            ]
        }
        return recommendations.get(persona, ["General Analysis"])
    
    def _get_dashboard_type(self, persona: UserPersona, categories: List[DataCategory]) -> DashboardType:
        """Determine dashboard type."""
        if DataCategory.MODEL_FILE in categories:
            return DashboardType.MODEL
        
        dashboard_map = {
            UserPersona.BUSINESS: DashboardType.BUSINESS,
            UserPersona.ANALYTICS: DashboardType.ANALYTICS,
            UserPersona.ML_ENGINEER: DashboardType.ML_METRICS,
            UserPersona.DATA_SCIENTIST: DashboardType.DATA_SCIENCE,
            UserPersona.DEVELOPER: DashboardType.DEVELOPER
        }
        return dashboard_map.get(persona, DashboardType.GENERAL)
    
    def _generate_summary(self, persona: UserPersona, domains: List[str], datasets: List[Dict]) -> str:
        """Generate a human-readable summary."""
        file_count = len(datasets)
        domain_str = ", ".join(domains[:3]) if domains else "general data"
        
        summaries = {
            UserPersona.BUSINESS: f"Detected {file_count} business data file(s) related to {domain_str}. Dashboard configured for business intelligence with KPIs, trends, and performance metrics.",
            UserPersona.ANALYTICS: f"Detected {file_count} analytical dataset(s) for {domain_str}. Dashboard configured for exploratory analysis with statistics, distributions, and correlations.",
            UserPersona.ML_ENGINEER: f"Detected {file_count} ML-related file(s) for {domain_str}. Dashboard configured for model performance tracking with metrics, predictions, and error analysis.",
            UserPersona.DATA_SCIENTIST: f"Detected {file_count} data science file(s) covering {domain_str}. Dashboard configured for comprehensive EDA and modeling insights.",
            UserPersona.DEVELOPER: f"Detected {file_count} technical data file(s). Dashboard configured for system monitoring and log analysis."
        }
        return summaries.get(persona, f"Detected {file_count} data file(s). General analysis dashboard configured.")


# Singleton
enhanced_persona_detector = PersonaDetector()

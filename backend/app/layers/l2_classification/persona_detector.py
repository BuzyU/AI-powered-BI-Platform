# Persona Detection Engine
# Detects WHO is using the system based on WHAT they uploaded

from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import pandas as pd
import re


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


@dataclass
class PersonaDetectionResult:
    """Result of persona detection."""
    persona: UserPersona
    confidence: float
    data_categories: List[DataCategory]
    detected_domains: List[str]
    recommended_analysis: List[str]
    dashboard_type: str
    summary: str


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
        classifications: List[Dict[str, Any]] = None
    ) -> PersonaDetectionResult:
        """
        Detect user persona based on uploaded datasets.
        
        Args:
            datasets: List of dataset info with 'filename', 'df', 'metadata'
            classifications: Optional existing classifications
        
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
    
    def _get_dashboard_type(self, persona: UserPersona, categories: List[DataCategory]) -> str:
        """Determine dashboard type."""
        if DataCategory.MODEL_FILE in categories:
            return "ml_model_dashboard"
        
        dashboard_map = {
            UserPersona.BUSINESS: "business_intelligence_dashboard",
            UserPersona.ANALYTICS: "exploratory_analytics_dashboard",
            UserPersona.ML_ENGINEER: "ml_performance_dashboard",
            UserPersona.DATA_SCIENTIST: "data_science_dashboard",
            UserPersona.DEVELOPER: "developer_dashboard"
        }
        return dashboard_map.get(persona, "general_dashboard")
    
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
persona_detector = PersonaDetector()

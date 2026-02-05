# Enhanced Persona Detection Engine
# Comprehensive detection for Business, Analytics, ML, and Computer Vision

from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)


class UserPersona(Enum):
    """Primary user personas based on upload patterns."""
    BUSINESS = "business"               # Sales, CRM, Finance, Operations
    ANALYTICS = "analytics"             # EDA, Research, Statistical Analysis
    ML_ENGINEER = "ml_engineer"         # ML Models, Predictions, Training Data
    COMPUTER_VISION = "computer_vision" # Image classification, Object Detection
    DATA_SCIENTIST = "data_scientist"   # Mix of analytics + ML
    DEVELOPER = "developer"             # API logs, System data
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
    
    # Computer Vision Data
    IMAGE_DATA = "image_data"
    OBJECT_DETECTION = "object_detection"
    IMAGE_CLASSIFICATION = "image_classification"
    SEGMENTATION = "segmentation"
    
    # Generic
    TABULAR_DATA = "tabular_data"
    UNKNOWN = "unknown"


class DashboardType(Enum):
    """Dashboard types for different personas."""
    POWER_BI_STYLE = "power_bi_style"           # Business KPIs, revenue, profit
    EDA_ANALYTICS = "eda_analytics"              # Statistical analysis, distributions
    ML_METRICS = "ml_metrics"                    # Confusion matrix, ROC, metrics
    CV_DASHBOARD = "cv_dashboard"                # Image metrics, detection results
    DATA_SCIENCE = "data_science"                # Mix of EDA + ML
    DEVELOPER = "developer"                      # Logs, system metrics
    GENERAL = "general"


@dataclass
class PersonaDetectionResult:
    """Result of persona detection."""
    persona: UserPersona
    confidence: float
    data_categories: List[DataCategory]
    detected_domains: List[str]
    recommended_analysis: List[str]
    dashboard_type: DashboardType
    dashboard_config: Dict[str, Any]
    summary: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'persona': self.persona.value,
            'confidence': self.confidence,
            'data_categories': [c.value for c in self.data_categories],
            'detected_domains': self.detected_domains,
            'recommended_analysis': self.recommended_analysis,
            'dashboard_type': self.dashboard_type.value,
            'dashboard_config': self.dashboard_config,
            'summary': self.summary
        }


class EnhancedPersonaDetector:
    """
    Enhanced persona detector with support for:
    - Business Analytics (Power BI style)
    - Data Analytics (EDA style)
    - ML Models (Classification/Regression metrics)
    - Computer Vision (Object Detection, Image Classification)
    """
    
    def __init__(self):
        self._init_all_indicators()
    
    def _init_all_indicators(self):
        """Initialize all detection patterns."""
        self.business_indicators = {
            'sales': {
                'columns': ['sale', 'revenue', 'order', 'transaction', 'invoice', 
                           'deal', 'booking', 'purchase', 'gross', 'net'],
                'filename': ['sales', 'orders', 'transactions', 'revenue'],
                'weight': 2.0
            },
            'customer': {
                'columns': ['customer', 'client', 'buyer', 'subscriber', 'member', 
                           'user_id', 'account', 'contact', 'lead', 'prospect'],
                'filename': ['customers', 'clients', 'crm', 'contacts'],
                'weight': 2.0
            },
            'product': {
                'columns': ['product', 'item', 'sku', 'catalog', 'inventory', 
                           'stock', 'merchandise', 'goods', 'upc'],
                'filename': ['products', 'inventory', 'catalog', 'items'],
                'weight': 1.5
            },
            'financial': {
                'columns': ['profit', 'cost', 'expense', 'budget', 'balance', 
                           'payment', 'tax', 'margin', 'ebitda', 'roi', 'cogs'],
                'filename': ['finance', 'budget', 'expenses', 'accounting', 'profit'],
                'weight': 2.5
            },
            'hr': {
                'columns': ['employee', 'salary', 'department', 'hire', 'position', 
                           'payroll', 'attendance', 'leave', 'bonus', 'benefits'],
                'filename': ['hr', 'employees', 'payroll', 'workforce'],
                'weight': 1.5
            },
            'marketing': {
                'columns': ['campaign', 'lead', 'conversion', 'click', 'impression', 
                           'ad_', 'promo', 'ctr', 'cpc', 'roi', 'attribution'],
                'filename': ['marketing', 'campaigns', 'ads', 'promotions'],
                'weight': 1.5
            },
            'operations': {
                'columns': ['branch', 'store', 'warehouse', 'location', 'region', 
                           'territory', 'supply', 'logistics', 'shipment'],
                'filename': ['operations', 'logistics', 'supply_chain'],
                'weight': 1.5
            }
        }
        
        self.analytics_indicators = {
            'survey': {
                'columns': ['response', 'survey', 'rating', 'score', 'feedback', 
                           'opinion', 'satisfaction', 'nps', 'likert'],
                'filename': ['survey', 'responses', 'feedback', 'ratings'],
                'weight': 1.5
            },
            'research': {
                'columns': ['experiment', 'control', 'treatment', 'sample', 
                           'observation', 'study', 'trial', 'cohort', 'group'],
                'filename': ['research', 'experiment', 'study', 'trial'],
                'weight': 1.5
            },
            'time_series': {
                'columns': ['timestamp', 'datetime', 'date', 'time', 'period', 
                           'interval', 'frequency', 'month', 'year', 'quarter'],
                'filename': ['timeseries', 'time_series', 'historical'],
                'weight': 1.0
            },
            'geographic': {
                'columns': ['latitude', 'longitude', 'lat', 'lng', 'geo', 
                           'location', 'city', 'country', 'state', 'zip', 'postal'],
                'filename': ['geo', 'location', 'geographic', 'map'],
                'weight': 1.0
            },
            'sensor': {
                'columns': ['sensor', 'reading', 'measurement', 'temperature', 
                           'humidity', 'pressure', 'aqi', 'pollution', 'voltage'],
                'filename': ['sensor', 'iot', 'readings', 'telemetry'],
                'weight': 1.5
            },
            'statistical': {
                'columns': ['mean', 'median', 'std', 'variance', 'correlation', 
                           'distribution', 'percentile', 'quantile'],
                'filename': ['statistics', 'analysis'],
                'weight': 1.0
            }
        }
        
        self.ml_indicators = {
            'training': {
                'columns': ['feature', 'label', 'target', 'train', 'test', 
                           'validation', 'split', 'fold', 'stratify'],
                'filename': ['train', 'training', 'dataset', 'features'],
                'weight': 2.0
            },
            'prediction': {
                'columns': ['prediction', 'predicted', 'probability', 'confidence', 
                           'score', 'output', 'proba', 'logit'],
                'filename': ['predictions', 'output', 'results', 'inference'],
                'weight': 2.5
            },
            'classification': {
                'columns': ['class', 'category', 'label', 'y_true', 'y_pred', 
                           'classification', 'true_label', 'pred_label'],
                'filename': ['classification', 'classes', 'labels'],
                'weight': 2.0
            },
            'regression': {
                'columns': ['actual', 'predicted', 'error', 'residual', 'mse', 
                           'rmse', 'mae', 'r2', 'explained_variance'],
                'filename': ['regression', 'forecast'],
                'weight': 2.0
            },
            'embedding': {
                'columns': ['embedding', 'vector', 'representation', 'encoding', 
                           'latent', 'dim_', 'pca', 'tsne', 'umap'],
                'filename': ['embeddings', 'vectors', 'encoded'],
                'weight': 1.5
            },
            'model_meta': {
                'columns': ['epoch', 'loss', 'accuracy', 'precision', 'recall', 
                           'f1', 'auc', 'roc', 'learning_rate', 'batch'],
                'filename': ['metrics', 'training_log', 'evaluation'],
                'weight': 2.0
            }
        }
        
        self.cv_indicators = {
            'object_detection': {
                'columns': ['bbox', 'bounding_box', 'x1', 'y1', 'x2', 'y2', 
                           'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height',
                           'iou', 'map', 'detection', 'anchor'],
                'filename': ['detection', 'bbox', 'objects', 'yolo', 'coco'],
                'weight': 3.0
            },
            'image_classification': {
                'columns': ['image', 'img', 'image_id', 'image_path', 'filename',
                           'class_name', 'class_id', 'top_1', 'top_5'],
                'filename': ['imagenet', 'classification', 'images', 'cifar'],
                'weight': 2.5
            },
            'segmentation': {
                'columns': ['mask', 'segment', 'pixel', 'region', 'dice', 
                           'jaccard', 'overlap', 'boundary'],
                'filename': ['segmentation', 'mask', 'semantic'],
                'weight': 2.5
            },
            'image_features': {
                'columns': ['image_feature', 'visual', 'cnn', 'resnet', 'vgg',
                           'inception', 'efficientnet', 'pooling'],
                'filename': ['features', 'extracted', 'cnn_features'],
                'weight': 2.0
            }
        }
        
        self.developer_indicators = {
            'logs': {
                'columns': ['log', 'message', 'level', 'error', 'warning', 
                           'info', 'debug', 'trace', 'stack'],
                'filename': ['log', 'logs', 'errors', 'debug'],
                'weight': 2.0
            },
            'api': {
                'columns': ['request', 'response', 'endpoint', 'method', 'status',
                           'latency', 'duration', 'path', 'url'],
                'filename': ['api', 'requests', 'traffic'],
                'weight': 2.0
            },
            'system': {
                'columns': ['cpu', 'memory', 'disk', 'network', 'process',
                           'thread', 'connection', 'socket'],
                'filename': ['system', 'metrics', 'monitoring'],
                'weight': 1.5
            }
        }
    
    def detect_persona(
        self,
        datasets: List[Dict[str, Any]],
        model_info: Optional[Dict[str, Any]] = None
    ) -> PersonaDetectionResult:
        """
        Detect user persona based on uploaded datasets and model info.
        """
        scores = {
            UserPersona.BUSINESS: 0.0,
            UserPersona.ANALYTICS: 0.0,
            UserPersona.ML_ENGINEER: 0.0,
            UserPersona.COMPUTER_VISION: 0.0,
            UserPersona.DATA_SCIENTIST: 0.0,
            UserPersona.DEVELOPER: 0.0
        }
        
        detected_categories: Set[DataCategory] = set()
        detected_domains: Set[str] = set()
        
        # Check if model was uploaded
        if model_info and model_info.get('is_model'):
            model_type = model_info.get('model_type', '').lower()
            
            # Check if it's a CV model
            cv_model_types = ['image', 'vision', 'cnn', 'resnet', 'vgg', 'yolo', 
                             'faster_rcnn', 'efficientnet', 'unet', 'detection']
            is_cv_model = any(cv_type in model_type for cv_type in cv_model_types)
            
            if is_cv_model:
                scores[UserPersona.COMPUTER_VISION] += 8.0
                detected_categories.add(DataCategory.MODEL_FILE)
                detected_categories.add(DataCategory.IMAGE_CLASSIFICATION)
                detected_domains.add('Computer Vision')
            else:
                scores[UserPersona.ML_ENGINEER] += 5.0
                detected_categories.add(DataCategory.MODEL_FILE)
                detected_domains.add('Machine Learning')
        
        # Analyze each dataset
        for ds in datasets:
            filename = ds.get('filename', '').lower()
            metadata = ds.get('metadata', {})
            df = ds.get('df')
            
            if df is None:
                continue
            
            columns = [c.lower() for c in df.columns]
            col_text = ' '.join(columns)
            
            # Check for CV data first (highest specificity)
            cv_score, cv_domains, cv_cats = self._score_indicators(
                columns, filename, self.cv_indicators
            )
            if cv_score > 0:
                scores[UserPersona.COMPUTER_VISION] += cv_score
                detected_domains.update(cv_domains)
                detected_categories.update(cv_cats)
                
                # Additional CV-specific checks
                if self._has_image_paths(df):
                    scores[UserPersona.COMPUTER_VISION] += 2.0
                    detected_categories.add(DataCategory.IMAGE_DATA)
                
                if self._has_bbox_columns(df):
                    scores[UserPersona.COMPUTER_VISION] += 3.0
                    detected_categories.add(DataCategory.OBJECT_DETECTION)
            
            # Score ML indicators
            ml_score, ml_domains, ml_cats = self._score_indicators(
                columns, filename, self.ml_indicators
            )
            if ml_score > 0:
                scores[UserPersona.ML_ENGINEER] += ml_score
                detected_domains.update(ml_domains)
                detected_categories.update(ml_cats)
            
            # Score business indicators
            biz_score, biz_domains, biz_cats = self._score_indicators(
                columns, filename, self.business_indicators
            )
            if biz_score > 0:
                scores[UserPersona.BUSINESS] += biz_score
                detected_domains.update(biz_domains)
                detected_categories.update(biz_cats)
            
            # Score analytics indicators
            ana_score, ana_domains, ana_cats = self._score_indicators(
                columns, filename, self.analytics_indicators
            )
            if ana_score > 0:
                scores[UserPersona.ANALYTICS] += ana_score
                detected_domains.update(ana_domains)
                detected_categories.update(ana_cats)
            
            # Score developer indicators
            dev_score, dev_domains, dev_cats = self._score_indicators(
                columns, filename, self.developer_indicators
            )
            if dev_score > 0:
                scores[UserPersona.DEVELOPER] += dev_score
                detected_domains.update(dev_domains)
                detected_categories.update(dev_cats)
            
            # Check data characteristics for additional signals
            self._score_by_data_characteristics(df, scores, detected_categories)
        
        # Data Scientist detection (mix of analytics + ML)
        if scores[UserPersona.ANALYTICS] > 2 and scores[UserPersona.ML_ENGINEER] > 2:
            scores[UserPersona.DATA_SCIENTIST] = (
                scores[UserPersona.ANALYTICS] + scores[UserPersona.ML_ENGINEER]
            ) * 0.7
        
        # Determine primary persona
        if not any(scores.values()):
            scores[UserPersona.ANALYTICS] = 1.0
            detected_categories.add(DataCategory.TABULAR_DATA)
        
        primary_persona = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[primary_persona] / total_score if total_score > 0 else 0.5
        
        # Get dashboard config based on persona
        dashboard_type = self._get_dashboard_type(primary_persona, detected_categories)
        dashboard_config = self._get_dashboard_config(
            primary_persona, dashboard_type, list(detected_categories)
        )
        
        recommended = self._get_recommendations(primary_persona, detected_categories)
        summary = self._generate_summary(primary_persona, detected_domains, datasets)
        
        return PersonaDetectionResult(
            persona=primary_persona,
            confidence=min(confidence, 1.0),
            data_categories=list(detected_categories),
            detected_domains=list(detected_domains),
            recommended_analysis=recommended,
            dashboard_type=dashboard_type,
            dashboard_config=dashboard_config,
            summary=summary
        )
    
    def _score_indicators(
        self, 
        columns: List[str], 
        filename: str, 
        indicators: Dict
    ) -> Tuple[float, Set[str], Set[DataCategory]]:
        """Score a set of indicators against columns and filename."""
        total_score = 0.0
        domains = set()
        categories = set()
        
        for domain, config in indicators.items():
            col_keywords = config['columns']
            file_keywords = config['filename']
            weight = config['weight']
            
            # Count column matches
            col_matches = sum(
                1 for kw in col_keywords 
                for col in columns 
                if kw in col
            )
            
            # Count filename matches
            file_matches = sum(1 for kw in file_keywords if kw in filename)
            
            if col_matches > 0 or file_matches > 0:
                score = (col_matches * 0.3 + file_matches * 0.7) * weight
                total_score += score
                domains.add(domain.replace('_', ' ').title())
                categories.add(self._domain_to_category(domain))
        
        return total_score, domains, categories
    
    def _domain_to_category(self, domain: str) -> DataCategory:
        """Map domain string to DataCategory."""
        mapping = {
            # Business
            'sales': DataCategory.SALES_DATA,
            'customer': DataCategory.CUSTOMER_DATA,
            'product': DataCategory.PRODUCT_DATA,
            'financial': DataCategory.FINANCIAL_DATA,
            'hr': DataCategory.HR_DATA,
            'marketing': DataCategory.MARKETING_DATA,
            'operations': DataCategory.INVENTORY_DATA,
            # Analytics
            'survey': DataCategory.SURVEY_DATA,
            'research': DataCategory.RESEARCH_DATA,
            'time_series': DataCategory.TIME_SERIES,
            'geographic': DataCategory.GEOGRAPHIC_DATA,
            'sensor': DataCategory.SENSOR_DATA,
            'statistical': DataCategory.STATISTICAL_DATA,
            # ML
            'training': DataCategory.TRAINING_DATA,
            'prediction': DataCategory.PREDICTIONS,
            'classification': DataCategory.PREDICTIONS,
            'regression': DataCategory.PREDICTIONS,
            'embedding': DataCategory.EMBEDDINGS,
            'model_meta': DataCategory.FEATURE_DATA,
            # CV
            'object_detection': DataCategory.OBJECT_DETECTION,
            'image_classification': DataCategory.IMAGE_CLASSIFICATION,
            'segmentation': DataCategory.SEGMENTATION,
            'image_features': DataCategory.IMAGE_DATA,
            # Developer
            'logs': DataCategory.TABULAR_DATA,
            'api': DataCategory.TABULAR_DATA,
            'system': DataCategory.SENSOR_DATA,
        }
        return mapping.get(domain, DataCategory.UNKNOWN)
    
    def _has_image_paths(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame contains image file paths."""
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].head(20).astype(str)
                image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
                if any(s.lower().endswith(image_extensions) for s in sample):
                    return True
        return False
    
    def _has_bbox_columns(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has bounding box columns."""
        bbox_patterns = ['x1', 'y1', 'x2', 'y2', 'xmin', 'ymin', 'xmax', 'ymax', 
                        'bbox', 'left', 'top', 'right', 'bottom']
        cols_lower = [c.lower() for c in df.columns]
        
        matches = sum(1 for p in bbox_patterns if any(p in c for c in cols_lower))
        return matches >= 4
    
    def _score_by_data_characteristics(
        self, 
        df: pd.DataFrame, 
        scores: Dict[UserPersona, float],
        categories: Set[DataCategory]
    ):
        """Score based on data characteristics like column types, ranges, etc."""
        n_cols = len(df.columns)
        n_rows = len(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_numeric = len(numeric_cols)
        
        # High number of numeric features suggests ML/analytics
        if n_numeric > 20:
            scores[UserPersona.ML_ENGINEER] += 1.0
            scores[UserPersona.DATA_SCIENTIST] += 0.5
        
        # Check for probability-like columns (0-1 range)
        prob_cols = 0
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                if col_data.min() >= 0 and col_data.max() <= 1:
                    prob_cols += 1
        
        if prob_cols >= 2:
            scores[UserPersona.ML_ENGINEER] += 1.5
            categories.add(DataCategory.PREDICTIONS)
        
        # Check for currency-like columns
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].head(10).astype(str)
                if any('$' in s or '€' in s or '£' in s for s in sample):
                    scores[UserPersona.BUSINESS] += 2.0
                    categories.add(DataCategory.FINANCIAL_DATA)
                    break
    
    def _get_dashboard_type(
        self, 
        persona: UserPersona, 
        categories: Set[DataCategory]
    ) -> DashboardType:
        """Determine dashboard type based on persona and data categories."""
        # CV takes precedence if detected
        cv_cats = {DataCategory.IMAGE_DATA, DataCategory.OBJECT_DETECTION, 
                   DataCategory.IMAGE_CLASSIFICATION, DataCategory.SEGMENTATION}
        if categories & cv_cats or persona == UserPersona.COMPUTER_VISION:
            return DashboardType.CV_DASHBOARD
        
        if DataCategory.MODEL_FILE in categories:
            return DashboardType.ML_METRICS
        
        dashboard_map = {
            UserPersona.BUSINESS: DashboardType.POWER_BI_STYLE,
            UserPersona.ANALYTICS: DashboardType.EDA_ANALYTICS,
            UserPersona.ML_ENGINEER: DashboardType.ML_METRICS,
            UserPersona.COMPUTER_VISION: DashboardType.CV_DASHBOARD,
            UserPersona.DATA_SCIENTIST: DashboardType.DATA_SCIENCE,
            UserPersona.DEVELOPER: DashboardType.DEVELOPER,
            UserPersona.UNKNOWN: DashboardType.GENERAL
        }
        return dashboard_map.get(persona, DashboardType.GENERAL)
    
    def _get_dashboard_config(
        self, 
        persona: UserPersona, 
        dashboard_type: DashboardType,
        categories: List[DataCategory]
    ) -> Dict[str, Any]:
        """Get dashboard configuration for the detected persona."""
        
        configs = {
            DashboardType.POWER_BI_STYLE: {
                'layout': 'business',
                'components': [
                    {'type': 'kpi_cards', 'metrics': ['revenue', 'profit', 'customers', 'orders']},
                    {'type': 'trend_chart', 'title': 'Revenue Trend'},
                    {'type': 'pie_chart', 'title': 'Sales Distribution'},
                    {'type': 'bar_chart', 'title': 'Top Products/Categories'},
                    {'type': 'comparison_chart', 'title': 'Regional Performance'},
                    {'type': 'data_table', 'title': 'Recent Transactions'}
                ],
                'theme': 'professional',
                'kpi_style': 'large_numbers',
                'show_filters': True,
                'show_date_range': True
            },
            DashboardType.EDA_ANALYTICS: {
                'layout': 'analytics',
                'components': [
                    {'type': 'data_overview', 'metrics': ['rows', 'columns', 'missing', 'types']},
                    {'type': 'distribution_plots', 'title': 'Column Distributions'},
                    {'type': 'correlation_matrix', 'title': 'Correlation Analysis'},
                    {'type': 'outlier_detection', 'title': 'Outlier Analysis'},
                    {'type': 'missing_data', 'title': 'Missing Data Pattern'},
                    {'type': 'descriptive_stats', 'title': 'Statistical Summary'},
                    {'type': 'histogram_grid', 'title': 'Value Distributions'},
                    {'type': 'box_plots', 'title': 'Distribution Comparison'}
                ],
                'theme': 'analytical',
                'show_statistics': True,
                'show_cleaning_suggestions': True
            },
            DashboardType.ML_METRICS: {
                'layout': 'ml',
                'components': [
                    {'type': 'model_info', 'title': 'Model Information'},
                    {'type': 'metric_cards', 'metrics': ['accuracy', 'precision', 'recall', 'f1']},
                    {'type': 'confusion_matrix', 'title': 'Confusion Matrix'},
                    {'type': 'roc_curve', 'title': 'ROC Curve'},
                    {'type': 'pr_curve', 'title': 'Precision-Recall Curve'},
                    {'type': 'feature_importance', 'title': 'Feature Importance'},
                    {'type': 'prediction_distribution', 'title': 'Prediction Distribution'},
                    {'type': 'error_analysis', 'title': 'Error Analysis'},
                    {'type': 'per_class_metrics', 'title': 'Per-Class Performance'}
                ],
                'theme': 'technical',
                'show_evaluation_options': True,
                'show_model_comparison': True
            },
            DashboardType.CV_DASHBOARD: {
                'layout': 'cv',
                'components': [
                    {'type': 'model_info', 'title': 'Vision Model'},
                    {'type': 'metric_cards', 'metrics': ['accuracy', 'mAP', 'mIoU', 'top5_accuracy']},
                    {'type': 'confusion_matrix', 'title': 'Class Confusion'},
                    {'type': 'class_distribution', 'title': 'Class Distribution'},
                    {'type': 'confidence_histogram', 'title': 'Confidence Distribution'},
                    {'type': 'iou_distribution', 'title': 'IoU Distribution'},
                    {'type': 'sample_predictions', 'title': 'Sample Predictions'},
                    {'type': 'per_class_metrics', 'title': 'Per-Class Performance'}
                ],
                'theme': 'technical',
                'show_image_samples': True,
                'show_detection_viz': True
            },
            DashboardType.DATA_SCIENCE: {
                'layout': 'data_science',
                'components': [
                    {'type': 'data_profile', 'title': 'Data Profile'},
                    {'type': 'feature_analysis', 'title': 'Feature Analysis'},
                    {'type': 'correlation_matrix', 'title': 'Correlations'},
                    {'type': 'distribution_analysis', 'title': 'Distributions'},
                    {'type': 'model_readiness', 'title': 'ML Readiness Score'},
                    {'type': 'suggested_models', 'title': 'Suggested Models'},
                    {'type': 'data_quality', 'title': 'Data Quality'}
                ],
                'theme': 'modern',
                'show_feature_engineering': True,
                'show_model_suggestions': True
            },
            DashboardType.DEVELOPER: {
                'layout': 'developer',
                'components': [
                    {'type': 'system_overview', 'title': 'System Overview'},
                    {'type': 'error_rate', 'title': 'Error Rate'},
                    {'type': 'latency_chart', 'title': 'Response Latency'},
                    {'type': 'endpoint_usage', 'title': 'API Endpoints'},
                    {'type': 'log_viewer', 'title': 'Recent Logs'}
                ],
                'theme': 'dark',
                'show_alerts': True
            },
            DashboardType.GENERAL: {
                'layout': 'general',
                'components': [
                    {'type': 'data_overview', 'title': 'Data Overview'},
                    {'type': 'column_summary', 'title': 'Column Summary'},
                    {'type': 'basic_charts', 'title': 'Visualizations'}
                ],
                'theme': 'default'
            }
        }
        
        return configs.get(dashboard_type, configs[DashboardType.GENERAL])
    
    def _get_recommendations(
        self, 
        persona: UserPersona, 
        categories: Set[DataCategory]
    ) -> List[str]:
        """Get recommended analysis based on persona."""
        recommendations = {
            UserPersona.BUSINESS: [
                "📊 Revenue & Profit Analysis",
                "👥 Customer Segmentation",
                "📈 Sales Trends & Forecasting",
                "🏆 Product Performance",
                "🗺️ Regional/Branch Comparison",
                "📋 KPI Dashboard",
                "💰 Margin Analysis"
            ],
            UserPersona.ANALYTICS: [
                "🔍 Exploratory Data Analysis",
                "📊 Statistical Summary",
                "🔗 Correlation Analysis",
                "📈 Distribution Analysis",
                "⚠️ Outlier Detection",
                "📉 Trend Analysis",
                "🧹 Data Cleaning Report",
                "📐 Skewness & Kurtosis Analysis"
            ],
            UserPersona.ML_ENGINEER: [
                "📊 Model Performance Metrics",
                "🎯 Confusion Matrix Analysis",
                "📈 ROC & PR Curves",
                "🏆 Feature Importance",
                "📉 Prediction Distribution",
                "❌ Error Analysis",
                "🔄 Cross-Validation Results",
                "⚖️ Model Comparison"
            ],
            UserPersona.COMPUTER_VISION: [
                "📊 Detection/Classification Metrics",
                "🎯 Confusion Matrix",
                "📐 IoU Analysis",
                "📈 Confidence Distribution",
                "🏷️ Per-Class Performance",
                "🖼️ Sample Predictions",
                "📉 mAP/mIoU Curves"
            ],
            UserPersona.DATA_SCIENTIST: [
                "🔍 EDA & Profiling",
                "🛠️ Feature Engineering Ideas",
                "📊 Statistical Tests",
                "🎯 Model Baseline",
                "📋 Data Quality Report",
                "🧪 Hypothesis Testing",
                "🔄 Cross-Validation Strategy"
            ],
            UserPersona.DEVELOPER: [
                "📋 Log Analysis",
                "❌ Error Rate Tracking",
                "⚡ Performance Metrics",
                "🔌 API Usage Patterns",
                "📊 System Health"
            ]
        }
        return recommendations.get(persona, ["📊 General Analysis"])
    
    def _generate_summary(
        self, 
        persona: UserPersona, 
        domains: Set[str], 
        datasets: List[Dict]
    ) -> str:
        """Generate a human-readable summary."""
        file_count = len(datasets)
        domain_str = ", ".join(list(domains)[:3]) if domains else "general data"
        
        summaries = {
            UserPersona.BUSINESS: f"📊 Detected {file_count} business data file(s) related to {domain_str}. Dashboard configured as Power BI-style with KPIs, revenue trends, and business performance metrics.",
            UserPersona.ANALYTICS: f"🔬 Detected {file_count} analytical dataset(s) for {domain_str}. Dashboard configured for comprehensive EDA with statistics, distributions, correlations, and outlier analysis.",
            UserPersona.ML_ENGINEER: f"🤖 Detected {file_count} ML-related file(s) for {domain_str}. Dashboard configured for model performance tracking with accuracy, precision, recall, F1, confusion matrix, and ROC curves.",
            UserPersona.COMPUTER_VISION: f"🖼️ Detected {file_count} computer vision file(s) for {domain_str}. Dashboard configured for CV metrics including mAP, IoU, class-wise performance, and detection visualization.",
            UserPersona.DATA_SCIENTIST: f"🧬 Detected {file_count} data science file(s) covering {domain_str}. Dashboard configured for comprehensive EDA, feature analysis, and modeling insights.",
            UserPersona.DEVELOPER: f"💻 Detected {file_count} technical data file(s). Dashboard configured for system monitoring, log analysis, and performance tracking."
        }
        return summaries.get(persona, f"📁 Detected {file_count} data file(s). General analysis dashboard configured.")


# Singleton
enhanced_persona_detector = EnhancedPersonaDetector()

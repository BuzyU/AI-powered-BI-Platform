# Comprehensive Metrics Calculator
# Calculates detailed metrics for ML models, business analytics, and data analysis

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """ML task types."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    OBJECT_DETECTION = "object_detection"
    IMAGE_CLASSIFICATION = "image_classification"
    UNKNOWN = "unknown"


@dataclass
class ColumnMapping:
    """Auto-detected column mappings."""
    actual_col: Optional[str] = None
    predicted_col: Optional[str] = None
    probability_cols: List[str] = None
    confidence_col: Optional[str] = None
    feature_cols: List[str] = None
    id_col: Optional[str] = None
    
    def __post_init__(self):
        if self.probability_cols is None:
            self.probability_cols = []
        if self.feature_cols is None:
            self.feature_cols = []


class SmartColumnDetector:
    """Auto-detect column roles from DataFrame."""
    
    # Patterns for different column types
    ACTUAL_PATTERNS = [
        'actual', 'true', 'y_true', 'label', 'target', 'ground_truth', 
        'real', 'observed', 'expected', 'correct', 'answer'
    ]
    
    PREDICTED_PATTERNS = [
        'predicted', 'pred', 'y_pred', 'prediction', 'output', 
        'forecast', 'estimate', 'classified', 'result'
    ]
    
    PROBABILITY_PATTERNS = [
        'probability', 'prob', 'proba', 'confidence', 'score', 
        'likelihood', 'certainty', 'softmax'
    ]
    
    ID_PATTERNS = [
        'id', 'index', 'row_id', 'sample_id', 'uuid', 'key'
    ]
    
    # CV-specific patterns
    CV_PATTERNS = {
        'bbox': ['bbox', 'box', 'bounding', 'x1', 'y1', 'x2', 'y2', 'xmin', 'ymin', 'xmax', 'ymax'],
        'iou': ['iou', 'jaccard', 'overlap'],
        'class': ['class', 'category', 'object', 'detection'],
        'image': ['image', 'img', 'file', 'path', 'filename']
    }
    
    @classmethod
    def detect_columns(cls, df: pd.DataFrame) -> ColumnMapping:
        """Auto-detect column roles from DataFrame."""
        columns = df.columns.tolist()
        columns_lower = [c.lower() for c in columns]
        
        mapping = ColumnMapping()
        used_cols = set()
        
        # Find actual column
        for pattern in cls.ACTUAL_PATTERNS:
            for i, col_lower in enumerate(columns_lower):
                if pattern in col_lower and columns[i] not in used_cols:
                    mapping.actual_col = columns[i]
                    used_cols.add(columns[i])
                    break
            if mapping.actual_col:
                break
        
        # Find predicted column
        for pattern in cls.PREDICTED_PATTERNS:
            for i, col_lower in enumerate(columns_lower):
                if pattern in col_lower and columns[i] not in used_cols:
                    mapping.predicted_col = columns[i]
                    used_cols.add(columns[i])
                    break
            if mapping.predicted_col:
                break
        
        # Find probability columns
        for pattern in cls.PROBABILITY_PATTERNS:
            for i, col_lower in enumerate(columns_lower):
                if pattern in col_lower and columns[i] not in used_cols:
                    mapping.probability_cols.append(columns[i])
                    used_cols.add(columns[i])
        
        # If only one probability column, it might be the confidence
        if len(mapping.probability_cols) == 1:
            mapping.confidence_col = mapping.probability_cols[0]
        
        # Find ID column
        for pattern in cls.ID_PATTERNS:
            for i, col_lower in enumerate(columns_lower):
                if pattern == col_lower or col_lower.endswith('_id'):
                    mapping.id_col = columns[i]
                    used_cols.add(columns[i])
                    break
            if mapping.id_col:
                break
        
        # Remaining numeric columns are features
        for col in columns:
            if col not in used_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    mapping.feature_cols.append(col)
        
        return mapping
    
    @classmethod
    def detect_task_type(cls, df: pd.DataFrame, mapping: ColumnMapping) -> TaskType:
        """Detect ML task type from data."""
        if mapping.actual_col is None:
            return TaskType.UNKNOWN
        
        actual = df[mapping.actual_col]
        
        # Check if it's numeric regression
        if pd.api.types.is_float_dtype(actual):
            unique_ratio = actual.nunique() / len(actual)
            if unique_ratio > 0.1:  # Many unique values = regression
                return TaskType.REGRESSION
        
        # Check unique values for classification
        n_unique = actual.nunique()
        
        if n_unique == 2:
            return TaskType.BINARY_CLASSIFICATION
        elif n_unique <= 100:
            return TaskType.MULTICLASS_CLASSIFICATION
        else:
            return TaskType.REGRESSION
    
    @classmethod
    def is_cv_data(cls, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check if data appears to be computer vision related."""
        columns_lower = [c.lower() for c in df.columns]
        
        # Check for bbox columns
        bbox_count = sum(1 for pattern in cls.CV_PATTERNS['bbox'] 
                        for col in columns_lower if pattern in col)
        if bbox_count >= 4:
            return True, 'object_detection'
        
        # Check for image paths
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].head(10).astype(str)
                if any(s.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.gif')) for s in sample):
                    return True, 'image_classification'
        
        return False, 'none'


class MetricsCalculator:
    """Calculate comprehensive metrics for ML models."""
    
    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics.
        
        Returns detailed metrics including:
        - Overall: Accuracy, Precision, Recall, F1, MCC
        - Per-class: Individual class metrics
        - Confusion matrix with annotations
        - ROC-AUC (if probabilities provided)
        - Classification report
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report, matthews_corrcoef,
            balanced_accuracy_score, cohen_kappa_score, log_loss,
            roc_auc_score, average_precision_score, precision_recall_curve,
            roc_curve
        )
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Get unique classes
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)
        is_binary = n_classes == 2
        
        if class_names is None:
            class_names = [str(c) for c in classes]
        
        metrics = {
            'task_type': 'binary_classification' if is_binary else 'multiclass_classification',
            'n_samples': len(y_true),
            'n_classes': n_classes,
            'class_names': class_names,
            'class_distribution': {}
        }
        
        # Class distribution
        for cls in classes:
            count = np.sum(y_true == cls)
            metrics['class_distribution'][str(cls)] = {
                'count': int(count),
                'percentage': round(count / len(y_true) * 100, 2)
            }
        
        # Overall metrics
        metrics['overall'] = {
            'accuracy': round(accuracy_score(y_true, y_pred) * 100, 2),
            'balanced_accuracy': round(balanced_accuracy_score(y_true, y_pred) * 100, 2),
            'precision_macro': round(precision_score(y_true, y_pred, average='macro', zero_division=0) * 100, 2),
            'precision_weighted': round(precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2),
            'recall_macro': round(recall_score(y_true, y_pred, average='macro', zero_division=0) * 100, 2),
            'recall_weighted': round(recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2),
            'f1_macro': round(f1_score(y_true, y_pred, average='macro', zero_division=0) * 100, 2),
            'f1_weighted': round(f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2),
            'matthews_corrcoef': round(matthews_corrcoef(y_true, y_pred), 4),
            'cohen_kappa': round(cohen_kappa_score(y_true, y_pred), 4)
        }
        
        # Confusion Matrix with annotations
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        metrics['confusion_matrix'] = {
            'matrix': cm.tolist(),
            'labels': class_names,
            'normalized': (cm / cm.sum(axis=1, keepdims=True)).tolist(),
            'annotations': MetricsCalculator._annotate_confusion_matrix(cm, class_names)
        }
        
        # Per-class metrics
        report = classification_report(y_true, y_pred, labels=classes, 
                                       target_names=class_names, output_dict=True, 
                                       zero_division=0)
        metrics['per_class'] = {}
        for cls_name in class_names:
            if cls_name in report:
                metrics['per_class'][cls_name] = {
                    'precision': round(report[cls_name]['precision'] * 100, 2),
                    'recall': round(report[cls_name]['recall'] * 100, 2),
                    'f1_score': round(report[cls_name]['f1-score'] * 100, 2),
                    'support': int(report[cls_name]['support'])
                }
        
        # Error analysis
        incorrect_mask = y_true != y_pred
        metrics['error_analysis'] = {
            'total_errors': int(np.sum(incorrect_mask)),
            'error_rate': round(np.mean(incorrect_mask) * 100, 2),
            'most_confused_pairs': MetricsCalculator._get_confused_pairs(y_true, y_pred, class_names)
        }
        
        # ROC and PR curves (if probabilities provided)
        if y_prob is not None:
            y_prob = np.array(y_prob)
            try:
                if is_binary:
                    # For binary, use probability of positive class
                    if len(y_prob.shape) == 2:
                        y_prob_pos = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob[:, 0]
                    else:
                        y_prob_pos = y_prob
                    
                    metrics['overall']['roc_auc'] = round(roc_auc_score(y_true, y_prob_pos) * 100, 2)
                    metrics['overall']['avg_precision'] = round(average_precision_score(y_true, y_prob_pos) * 100, 2)
                    
                    # ROC curve data
                    fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
                    metrics['roc_curve'] = {
                        'fpr': fpr[::max(1, len(fpr)//100)].tolist(),
                        'tpr': tpr[::max(1, len(tpr)//100)].tolist()
                    }
                    
                    # PR curve data
                    precision, recall, _ = precision_recall_curve(y_true, y_prob_pos)
                    metrics['pr_curve'] = {
                        'precision': precision[::max(1, len(precision)//100)].tolist(),
                        'recall': recall[::max(1, len(recall)//100)].tolist()
                    }
                else:
                    # Multi-class ROC-AUC
                    try:
                        metrics['overall']['roc_auc_ovr'] = round(
                            roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted') * 100, 2)
                    except:
                        pass
                
                # Log loss
                metrics['overall']['log_loss'] = round(log_loss(y_true, y_prob), 4)
            except Exception as e:
                logger.warning(f"Could not calculate probability metrics: {e}")
        
        # Prediction distribution
        pred_counts = pd.Series(y_pred).value_counts()
        metrics['prediction_distribution'] = {
            str(k): int(v) for k, v in pred_counts.items()
        }
        
        return metrics
    
    @staticmethod
    def _annotate_confusion_matrix(cm: np.ndarray, labels: List[str]) -> List[Dict]:
        """Generate annotations for confusion matrix."""
        annotations = []
        n = len(labels)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    annotation_type = 'true_positive' if cm[i, j] > 0 else 'true_negative'
                else:
                    annotation_type = 'false_positive' if cm[i, j] > 0 else 'correct_rejection'
                
                annotations.append({
                    'actual': labels[i],
                    'predicted': labels[j],
                    'count': int(cm[i, j]),
                    'type': annotation_type
                })
        
        return annotations
    
    @staticmethod
    def _get_confused_pairs(y_true: np.ndarray, y_pred: np.ndarray, 
                           class_names: List[str], top_n: int = 5) -> List[Dict]:
        """Get most commonly confused class pairs."""
        confused = []
        classes = np.unique(y_true)
        
        for i, cls_a in enumerate(classes):
            for cls_b in classes[i+1:]:
                # A predicted as B
                mask_ab = (y_true == cls_a) & (y_pred == cls_b)
                count_ab = np.sum(mask_ab)
                
                # B predicted as A
                mask_ba = (y_true == cls_b) & (y_pred == cls_a)
                count_ba = np.sum(mask_ba)
                
                if count_ab > 0 or count_ba > 0:
                    confused.append({
                        'class_a': str(cls_a),
                        'class_b': str(cls_b),
                        'a_as_b': int(count_ab),
                        'b_as_a': int(count_ba),
                        'total': int(count_ab + count_ba)
                    })
        
        return sorted(confused, key=lambda x: x['total'], reverse=True)[:top_n]
    
    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive regression metrics.
        
        Returns:
        - Core metrics: R², MSE, RMSE, MAE, MAPE
        - Residual analysis
        - Distribution statistics
        - Scatter plot data
        """
        from sklearn.metrics import (
            r2_score, mean_squared_error, mean_absolute_error,
            median_absolute_error, explained_variance_score,
            max_error, mean_absolute_percentage_error
        )
        
        y_true = np.array(y_true).astype(float)
        y_pred = np.array(y_pred).astype(float)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        metrics = {
            'task_type': 'regression',
            'n_samples': len(y_true)
        }
        
        # Core metrics
        mse = mean_squared_error(y_true, y_pred)
        metrics['core'] = {
            'r2_score': round(r2_score(y_true, y_pred), 4),
            'adjusted_r2': round(1 - (1 - r2_score(y_true, y_pred)) * (len(y_true) - 1) / (len(y_true) - 2), 4),
            'explained_variance': round(explained_variance_score(y_true, y_pred), 4),
            'mse': round(mse, 4),
            'rmse': round(np.sqrt(mse), 4),
            'mae': round(mean_absolute_error(y_true, y_pred), 4),
            'median_ae': round(median_absolute_error(y_true, y_pred), 4),
            'max_error': round(max_error(y_true, y_pred), 4)
        }
        
        # MAPE (avoid division by zero)
        try:
            non_zero_mask = y_true != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
                metrics['core']['mape'] = round(mape, 2)
        except:
            pass
        
        # Residual analysis
        residuals = y_true - y_pred
        metrics['residuals'] = {
            'mean': round(float(np.mean(residuals)), 4),
            'std': round(float(np.std(residuals)), 4),
            'min': round(float(np.min(residuals)), 4),
            'max': round(float(np.max(residuals)), 4),
            'median': round(float(np.median(residuals)), 4),
            'skewness': round(float(pd.Series(residuals).skew()), 4),
            'kurtosis': round(float(pd.Series(residuals).kurtosis()), 4)
        }
        
        # Percentile errors
        abs_errors = np.abs(residuals)
        metrics['error_percentiles'] = {
            'p50': round(float(np.percentile(abs_errors, 50)), 4),
            'p75': round(float(np.percentile(abs_errors, 75)), 4),
            'p90': round(float(np.percentile(abs_errors, 90)), 4),
            'p95': round(float(np.percentile(abs_errors, 95)), 4),
            'p99': round(float(np.percentile(abs_errors, 99)), 4)
        }
        
        # Distribution statistics
        metrics['target_distribution'] = {
            'actual_mean': round(float(np.mean(y_true)), 4),
            'actual_std': round(float(np.std(y_true)), 4),
            'actual_min': round(float(np.min(y_true)), 4),
            'actual_max': round(float(np.max(y_true)), 4),
            'predicted_mean': round(float(np.mean(y_pred)), 4),
            'predicted_std': round(float(np.std(y_pred)), 4),
            'predicted_min': round(float(np.min(y_pred)), 4),
            'predicted_max': round(float(np.max(y_pred)), 4)
        }
        
        # Scatter plot data (sampled for large datasets)
        if len(y_true) > 1000:
            indices = np.random.choice(len(y_true), 1000, replace=False)
            scatter_true = y_true[indices]
            scatter_pred = y_pred[indices]
            scatter_res = residuals[indices]
        else:
            scatter_true = y_true
            scatter_pred = y_pred
            scatter_res = residuals
        
        metrics['scatter_data'] = [
            {'actual': float(a), 'predicted': float(p), 'residual': float(r)}
            for a, p, r in zip(scatter_true, scatter_pred, scatter_res)
        ]
        
        # Residual histogram bins
        hist, bin_edges = np.histogram(residuals, bins=30)
        metrics['residual_histogram'] = {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }
        
        return metrics
    
    @staticmethod
    def calculate_cv_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        iou_scores: Optional[np.ndarray] = None,
        confidence_scores: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Calculate computer vision specific metrics."""
        # Start with classification metrics
        metrics = MetricsCalculator.calculate_classification_metrics(
            y_true, y_pred, 
            y_prob=confidence_scores if confidence_scores is not None else None
        )
        
        metrics['task_type'] = 'computer_vision'
        
        # Add IoU metrics if available
        if iou_scores is not None:
            iou_scores = np.array(iou_scores)
            metrics['iou_metrics'] = {
                'mean_iou': round(float(np.mean(iou_scores)), 4),
                'median_iou': round(float(np.median(iou_scores)), 4),
                'std_iou': round(float(np.std(iou_scores)), 4),
                'iou_above_50': round(float(np.mean(iou_scores > 0.5)) * 100, 2),
                'iou_above_75': round(float(np.mean(iou_scores > 0.75)) * 100, 2)
            }
            
            # IoU distribution
            hist, bin_edges = np.histogram(iou_scores, bins=20, range=(0, 1))
            metrics['iou_distribution'] = {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
        
        # Confidence distribution
        if confidence_scores is not None:
            conf = np.array(confidence_scores)
            metrics['confidence_metrics'] = {
                'mean_confidence': round(float(np.mean(conf)), 4),
                'median_confidence': round(float(np.median(conf)), 4),
                'high_confidence_rate': round(float(np.mean(conf > 0.8)) * 100, 2)
            }
        
        return metrics


class BusinessMetricsCalculator:
    """Calculate business-specific KPIs and metrics."""
    
    @staticmethod
    def calculate_financial_metrics(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate financial/business metrics from data."""
        metrics = {'type': 'financial_metrics'}
        
        # Try to find revenue/sales columns
        revenue_cols = [c for c in df.columns if any(
            kw in c.lower() for kw in ['revenue', 'sales', 'amount', 'total', 'value', 'price']
        ) and pd.api.types.is_numeric_dtype(df[c])]
        
        if revenue_cols:
            col = revenue_cols[0]
            metrics['revenue'] = {
                'total': round(float(df[col].sum()), 2),
                'average': round(float(df[col].mean()), 2),
                'median': round(float(df[col].median()), 2),
                'min': round(float(df[col].min()), 2),
                'max': round(float(df[col].max()), 2),
                'std': round(float(df[col].std()), 2)
            }
        
        # Try to find cost/expense columns
        cost_cols = [c for c in df.columns if any(
            kw in c.lower() for kw in ['cost', 'expense', 'spend', 'fee']
        ) and pd.api.types.is_numeric_dtype(df[c])]
        
        if cost_cols:
            col = cost_cols[0]
            metrics['costs'] = {
                'total': round(float(df[col].sum()), 2),
                'average': round(float(df[col].mean()), 2)
            }
        
        # Calculate profit if both exist
        if revenue_cols and cost_cols:
            total_revenue = df[revenue_cols[0]].sum()
            total_cost = df[cost_cols[0]].sum()
            profit = total_revenue - total_cost
            margin = (profit / total_revenue * 100) if total_revenue > 0 else 0
            
            metrics['profitability'] = {
                'gross_profit': round(float(profit), 2),
                'profit_margin': round(float(margin), 2),
                'cost_ratio': round(float(total_cost / total_revenue * 100) if total_revenue > 0 else 0, 2)
            }
        
        # Count/quantity metrics
        quantity_cols = [c for c in df.columns if any(
            kw in c.lower() for kw in ['quantity', 'count', 'units', 'orders', 'transactions']
        ) and pd.api.types.is_numeric_dtype(df[c])]
        
        if quantity_cols:
            col = quantity_cols[0]
            metrics['volume'] = {
                'total': int(df[col].sum()),
                'average': round(float(df[col].mean()), 2)
            }
        
        # Date-based metrics
        date_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist()
        if not date_cols:
            for col in df.columns:
                try:
                    pd.to_datetime(df[col], errors='raise')
                    date_cols.append(col)
                    break
                except:
                    pass
        
        if date_cols and revenue_cols:
            df_temp = df.copy()
            df_temp['_date'] = pd.to_datetime(df_temp[date_cols[0]], errors='coerce')
            df_temp = df_temp.dropna(subset=['_date'])
            
            if len(df_temp) > 0:
                df_temp['_month'] = df_temp['_date'].dt.to_period('M')
                monthly = df_temp.groupby('_month')[revenue_cols[0]].sum()
                
                if len(monthly) >= 2:
                    growth = ((monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2] * 100) if monthly.iloc[-2] != 0 else 0
                    metrics['growth'] = {
                        'monthly_growth_rate': round(float(growth), 2),
                        'trend': 'increasing' if growth > 0 else 'decreasing' if growth < 0 else 'stable'
                    }
        
        return metrics


class StatisticalAnalyzer:
    """Perform statistical analysis on data."""
    
    @staticmethod
    def analyze_distribution(series: pd.Series) -> Dict[str, Any]:
        """Analyze distribution of a numeric series."""
        from scipy import stats
        
        clean_series = series.dropna()
        
        analysis = {
            'count': len(clean_series),
            'missing': int(series.isna().sum()),
            'missing_pct': round(series.isna().mean() * 100, 2)
        }
        
        if len(clean_series) == 0:
            return analysis
        
        # Basic stats
        analysis['basic'] = {
            'mean': round(float(clean_series.mean()), 4),
            'median': round(float(clean_series.median()), 4),
            'mode': float(clean_series.mode().iloc[0]) if len(clean_series.mode()) > 0 else None,
            'std': round(float(clean_series.std()), 4),
            'variance': round(float(clean_series.var()), 4),
            'min': round(float(clean_series.min()), 4),
            'max': round(float(clean_series.max()), 4),
            'range': round(float(clean_series.max() - clean_series.min()), 4)
        }
        
        # Percentiles
        analysis['percentiles'] = {
            'p1': round(float(clean_series.quantile(0.01)), 4),
            'p5': round(float(clean_series.quantile(0.05)), 4),
            'p25': round(float(clean_series.quantile(0.25)), 4),
            'p50': round(float(clean_series.quantile(0.50)), 4),
            'p75': round(float(clean_series.quantile(0.75)), 4),
            'p95': round(float(clean_series.quantile(0.95)), 4),
            'p99': round(float(clean_series.quantile(0.99)), 4),
            'iqr': round(float(clean_series.quantile(0.75) - clean_series.quantile(0.25)), 4)
        }
        
        # Shape metrics
        analysis['shape'] = {
            'skewness': round(float(clean_series.skew()), 4),
            'kurtosis': round(float(clean_series.kurtosis()), 4)
        }
        
        # Interpret skewness
        skew = clean_series.skew()
        if abs(skew) < 0.5:
            analysis['shape']['skew_interpretation'] = 'approximately symmetric'
        elif skew > 0:
            analysis['shape']['skew_interpretation'] = 'positively skewed (right tail)'
        else:
            analysis['shape']['skew_interpretation'] = 'negatively skewed (left tail)'
        
        # Outlier detection (IQR method)
        q1 = clean_series.quantile(0.25)
        q3 = clean_series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
        analysis['outliers'] = {
            'count': len(outliers),
            'percentage': round(len(outliers) / len(clean_series) * 100, 2),
            'lower_bound': round(float(lower_bound), 4),
            'upper_bound': round(float(upper_bound), 4)
        }
        
        # Normality test
        if len(clean_series) >= 8:
            try:
                stat, p_value = stats.shapiro(clean_series.sample(min(5000, len(clean_series))))
                analysis['normality_test'] = {
                    'test': 'Shapiro-Wilk',
                    'statistic': round(float(stat), 4),
                    'p_value': round(float(p_value), 4),
                    'is_normal': p_value > 0.05,
                    'interpretation': 'Data appears normally distributed' if p_value > 0.05 
                                    else 'Data does not appear normally distributed'
                }
            except:
                pass
        
        # Histogram data
        hist, bin_edges = np.histogram(clean_series, bins='auto')
        analysis['histogram'] = {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }
        
        return analysis
    
    @staticmethod
    def correlation_analysis(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return {'error': 'Need at least 2 numeric columns for correlation analysis'}
        
        # Limit columns for large datasets
        if numeric_df.shape[1] > 20:
            numeric_df = numeric_df.iloc[:, :20]
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_correlations.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': round(corr_val, 4),
                        'strength': 'strong' if abs(corr_val) > 0.7 else 'moderate'
                    })
        
        return {
            'matrix': corr_matrix.round(4).to_dict(),
            'columns': corr_matrix.columns.tolist(),
            'strong_correlations': sorted(strong_correlations, 
                                         key=lambda x: abs(x['correlation']), 
                                         reverse=True)[:10]
        }


# Export calculator instances
metrics_calculator = MetricsCalculator()
business_calculator = BusinessMetricsCalculator()
statistical_analyzer = StatisticalAnalyzer()
column_detector = SmartColumnDetector()

# Enhanced Model Evaluation Routes
# Smart column detection and comprehensive metrics calculation

from fastapi import APIRouter, HTTPException, Header, UploadFile, File, Body
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import logging
import io

from app.services import state
from app.layers.l7_analytics.metrics_calculator import (
    SmartColumnDetector, MetricsCalculator, column_detector, metrics_calculator,
    TaskType, ColumnMapping
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluate", tags=["evaluation"])


@router.post("/predictions")
async def evaluate_predictions_smart(
    file: UploadFile = File(...),
    actual_col: Optional[str] = None,
    predicted_col: Optional[str] = None,
    probability_col: Optional[str] = None,
    x_session_id: str = Header("default_session")
):
    """
    Smart evaluation of predictions from a CSV file.
    
    Features:
    - Auto-detects actual/predicted columns if not specified
    - Detects task type (binary/multiclass classification, regression)
    - Calculates comprehensive metrics (not predefined)
    - Returns detailed analysis including confusion matrix, ROC data, error analysis
    
    File can have any column names - the system will try to auto-detect:
    - Actual/True/Label/Target/y_true/ground_truth columns
    - Predicted/Pred/Prediction/y_pred/output columns
    - Probability/Confidence/Score columns
    
    Or specify exact column names using query parameters.
    """
    # Read the file
    content = await file.read()
    
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Failed to parse file: {str(e)}")
    
    logger.info(f"Evaluating predictions file: {file.filename}, shape: {df.shape}")
    logger.info(f"Columns detected: {df.columns.tolist()}")
    
    # Use smart column detection
    mapping = column_detector.detect_columns(df)
    
    # Override with user-specified columns if provided
    if actual_col:
        if actual_col not in df.columns:
            raise HTTPException(400, f"Specified actual column '{actual_col}' not found")
        mapping.actual_col = actual_col
    
    if predicted_col:
        if predicted_col not in df.columns:
            raise HTTPException(400, f"Specified predicted column '{predicted_col}' not found")
        mapping.predicted_col = predicted_col
    
    if probability_col:
        if probability_col not in df.columns:
            raise HTTPException(400, f"Specified probability column '{probability_col}' not found")
        mapping.probability_cols = [probability_col]
    
    # Validate we found the required columns
    if not mapping.actual_col:
        raise HTTPException(
            400, 
            f"Could not auto-detect actual/true column. Columns found: {df.columns.tolist()}. "
            f"Please specify 'actual_col' parameter or rename your column to one of: "
            f"actual, true, y_true, label, target, ground_truth"
        )
    
    if not mapping.predicted_col:
        raise HTTPException(
            400, 
            f"Could not auto-detect predicted column. Columns found: {df.columns.tolist()}. "
            f"Please specify 'predicted_col' parameter or rename your column to one of: "
            f"predicted, pred, y_pred, prediction, output"
        )
    
    # Get the data
    y_true = df[mapping.actual_col].values
    y_pred = df[mapping.predicted_col].values
    
    # Get probability data if available
    y_prob = None
    if mapping.probability_cols:
        prob_cols = [c for c in mapping.probability_cols if c in df.columns]
        if prob_cols:
            y_prob = df[prob_cols].values
            if y_prob.shape[1] == 1:
                y_prob = y_prob.flatten()
    
    # Detect task type
    task_type = column_detector.detect_task_type(df, mapping)
    
    # Calculate metrics based on task type
    if task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
        metrics = metrics_calculator.calculate_classification_metrics(
            y_true, y_pred, y_prob
        )
    elif task_type == TaskType.REGRESSION:
        metrics = metrics_calculator.calculate_regression_metrics(y_true, y_pred)
    else:
        # Try to determine from data characteristics
        unique_values = np.unique(y_true)
        if len(unique_values) <= 20:
            metrics = metrics_calculator.calculate_classification_metrics(y_true, y_pred, y_prob)
        else:
            metrics = metrics_calculator.calculate_regression_metrics(y_true, y_pred)
    
    # Add file metadata
    metrics['file_info'] = {
        'filename': file.filename,
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': df.columns.tolist()
    }
    
    metrics['column_mapping'] = {
        'actual_col': mapping.actual_col,
        'predicted_col': mapping.predicted_col,
        'probability_cols': mapping.probability_cols,
        'auto_detected': actual_col is None and predicted_col is None
    }
    
    # Store in session
    state.store_evaluation(x_session_id, metrics)
    
    logger.info(f"Evaluation complete: {metrics.get('task_type')}, samples: {metrics.get('n_samples')}")
    
    return metrics


@router.post("/comprehensive")
async def comprehensive_evaluation(
    file: UploadFile = File(...),
    x_session_id: str = Header("default_session")
):
    """
    Perform comprehensive evaluation with all possible metrics.
    
    This endpoint extracts maximum insights from prediction data:
    - All classification metrics (accuracy, precision, recall, F1, MCC, etc.)
    - Per-class analysis
    - Confusion matrix with annotations
    - Error analysis (most confused pairs)
    - ROC/PR curves (if probabilities available)
    - Statistical analysis of predictions
    
    For regression:
    - R², MSE, RMSE, MAE, MAPE
    - Residual analysis (mean, std, skewness)
    - Percentile errors
    - Scatter plot data
    """
    content = await file.read()
    
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Failed to parse file: {str(e)}")
    
    # Smart column detection
    mapping = column_detector.detect_columns(df)
    
    if not mapping.actual_col or not mapping.predicted_col:
        # List what we found to help user
        col_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample = df[col].head(3).tolist()
            col_info.append(f"  - {col} ({dtype}): {sample}")
        
        raise HTTPException(
            400,
            f"Could not detect actual/predicted columns.\n\n"
            f"Columns in file:\n" + "\n".join(col_info) + "\n\n"
            f"Tip: Rename columns to 'actual' and 'predicted' or similar."
        )
    
    y_true = df[mapping.actual_col].values
    y_pred = df[mapping.predicted_col].values
    
    # Get probabilities
    y_prob = None
    if mapping.probability_cols:
        prob_cols = [c for c in mapping.probability_cols if c in df.columns]
        if prob_cols:
            y_prob = df[prob_cols].values
            if y_prob.shape[1] == 1:
                y_prob = y_prob.flatten()
    
    # Check for CV data
    is_cv, cv_type = column_detector.is_cv_data(df)
    
    if is_cv:
        # Get IoU if available
        iou_scores = None
        for col in df.columns:
            if 'iou' in col.lower():
                iou_scores = df[col].values
                break
        
        # Get confidence scores
        conf_scores = None
        for col in df.columns:
            if any(kw in col.lower() for kw in ['confidence', 'score', 'prob']):
                conf_scores = df[col].values
                break
        
        metrics = metrics_calculator.calculate_cv_metrics(
            y_true, y_pred, iou_scores, conf_scores
        )
    else:
        # Detect task type
        task_type = column_detector.detect_task_type(df, mapping)
        
        if task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
            metrics = metrics_calculator.calculate_classification_metrics(
                y_true, y_pred, y_prob
            )
        else:
            metrics = metrics_calculator.calculate_regression_metrics(y_true, y_pred)
    
    # Add comprehensive file metadata
    metrics['file_info'] = {
        'filename': file.filename,
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': df.columns.tolist(),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
    }
    
    metrics['column_mapping'] = {
        'actual_col': mapping.actual_col,
        'predicted_col': mapping.predicted_col,
        'probability_cols': mapping.probability_cols,
        'feature_cols': mapping.feature_cols,
        'id_col': mapping.id_col
    }
    
    if is_cv:
        metrics['is_computer_vision'] = True
        metrics['cv_type'] = cv_type
    
    # Store in session
    state.store_evaluation(x_session_id, metrics)
    
    return metrics


@router.post("/multi-model")
async def compare_models(
    body: Dict[str, Any] = Body(...),
    x_session_id: str = Header("default_session")
):
    """
    Compare multiple models' predictions.
    
    Body:
    {
        "actual": [list of actual values],
        "models": {
            "model_1": {"predictions": [...]},
            "model_2": {"predictions": [...], "probabilities": [...]},
            ...
        }
    }
    
    Returns comparative metrics for all models.
    """
    actual = body.get('actual')
    models = body.get('models', {})
    
    if not actual:
        raise HTTPException(400, "actual values are required")
    
    if not models:
        raise HTTPException(400, "At least one model's predictions required")
    
    y_true = np.array(actual)
    
    # Determine if classification or regression
    unique_values = np.unique(y_true)
    is_classification = len(unique_values) <= 20
    
    results = {
        'n_samples': len(y_true),
        'task_type': 'classification' if is_classification else 'regression',
        'models': {}
    }
    
    for model_name, model_data in models.items():
        y_pred = np.array(model_data.get('predictions', []))
        y_prob = model_data.get('probabilities')
        
        if len(y_pred) != len(y_true):
            results['models'][model_name] = {
                'error': f'Length mismatch: {len(y_pred)} predictions vs {len(y_true)} actual'
            }
            continue
        
        if y_prob is not None:
            y_prob = np.array(y_prob)
        
        if is_classification:
            metrics = metrics_calculator.calculate_classification_metrics(
                y_true, y_pred, y_prob
            )
        else:
            metrics = metrics_calculator.calculate_regression_metrics(y_true, y_pred)
        
        results['models'][model_name] = metrics
    
    # Add comparison summary
    if len(results['models']) > 1 and all('error' not in m for m in results['models'].values()):
        results['comparison'] = _compare_models_summary(results['models'], is_classification)
    
    return results


def _compare_models_summary(models: Dict[str, Any], is_classification: bool) -> Dict[str, Any]:
    """Generate comparison summary across models."""
    summary = {'rankings': {}}
    
    if is_classification:
        # Rank by accuracy, f1, etc.
        for metric in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']:
            scores = []
            for name, m in models.items():
                overall = m.get('overall', {})
                if metric in overall:
                    scores.append((name, overall[metric]))
            
            if scores:
                scores.sort(key=lambda x: x[1], reverse=True)
                summary['rankings'][metric] = [{'model': n, 'score': s} for n, s in scores]
        
        # Best overall model (by F1)
        if 'f1_macro' in summary['rankings']:
            summary['best_model'] = summary['rankings']['f1_macro'][0]['model']
    else:
        # Rank by R², MAE, RMSE
        for metric, reverse in [('r2_score', True), ('mae', False), ('rmse', False)]:
            scores = []
            for name, m in models.items():
                core = m.get('core', {})
                if metric in core:
                    scores.append((name, core[metric]))
            
            if scores:
                scores.sort(key=lambda x: x[1], reverse=reverse)
                summary['rankings'][metric] = [{'model': n, 'score': s} for n, s in scores]
        
        # Best overall (by R²)
        if 'r2_score' in summary['rankings']:
            summary['best_model'] = summary['rankings']['r2_score'][0]['model']
    
    return summary


@router.get("/column-hints")
async def get_column_hints():
    """
    Get list of column name patterns the system can auto-detect.
    
    Useful for users to know how to name their columns for auto-detection.
    """
    return {
        'actual_patterns': SmartColumnDetector.ACTUAL_PATTERNS,
        'predicted_patterns': SmartColumnDetector.PREDICTED_PATTERNS,
        'probability_patterns': SmartColumnDetector.PROBABILITY_PATTERNS,
        'cv_bbox_patterns': SmartColumnDetector.CV_PATTERNS['bbox'],
        'cv_class_patterns': SmartColumnDetector.CV_PATTERNS['class'],
        'examples': {
            'classification': {
                'columns': ['id', 'actual', 'predicted', 'probability'],
                'description': 'For binary/multiclass classification evaluation'
            },
            'regression': {
                'columns': ['id', 'actual', 'predicted'],
                'description': 'For regression evaluation'
            },
            'object_detection': {
                'columns': ['image_id', 'class', 'predicted_class', 'confidence', 'iou', 'x1', 'y1', 'x2', 'y2'],
                'description': 'For object detection evaluation'
            }
        }
    }


@router.post("/detect-columns")
async def detect_columns(
    file: UploadFile = File(...),
):
    """
    Analyze a file and detect which columns could be actual/predicted.
    
    Useful for previewing before running evaluation.
    """
    content = await file.read()
    
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content), nrows=100)  # Just read sample
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content), nrows=100)
        else:
            df = pd.read_csv(io.BytesIO(content), nrows=100)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse file: {str(e)}")
    
    # Detect columns
    mapping = column_detector.detect_columns(df)
    
    # Check for CV data
    is_cv, cv_type = column_detector.is_cv_data(df)
    
    # Get column info
    columns_info = []
    for col in df.columns:
        info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'n_unique': int(df[col].nunique()),
            'sample_values': df[col].head(5).tolist(),
            'role': None
        }
        
        if col == mapping.actual_col:
            info['role'] = 'actual'
        elif col == mapping.predicted_col:
            info['role'] = 'predicted'
        elif col in mapping.probability_cols:
            info['role'] = 'probability'
        elif col == mapping.id_col:
            info['role'] = 'id'
        elif col in mapping.feature_cols:
            info['role'] = 'feature'
        
        columns_info.append(info)
    
    return {
        'filename': file.filename,
        'n_columns': len(df.columns),
        'detected_mapping': {
            'actual_col': mapping.actual_col,
            'predicted_col': mapping.predicted_col,
            'probability_cols': mapping.probability_cols,
            'feature_cols': mapping.feature_cols[:5],  # Limit for response
            'id_col': mapping.id_col
        },
        'is_cv_data': is_cv,
        'cv_type': cv_type if is_cv else None,
        'columns': columns_info,
        'recommendation': _get_recommendation(mapping, is_cv)
    }


def _get_recommendation(mapping: ColumnMapping, is_cv: bool) -> str:
    """Generate recommendation based on detected columns."""
    if mapping.actual_col and mapping.predicted_col:
        if is_cv:
            return "✅ Ready for Computer Vision evaluation. Actual and predicted columns detected."
        return "✅ Ready for evaluation. Both actual and predicted columns detected."
    elif mapping.actual_col:
        return "⚠️ Only actual column detected. Add a 'predicted' or 'prediction' column."
    elif mapping.predicted_col:
        return "⚠️ Only predicted column detected. Add an 'actual' or 'label' column."
    else:
        return "❌ Could not detect evaluation columns. Rename columns to 'actual' and 'predicted'."

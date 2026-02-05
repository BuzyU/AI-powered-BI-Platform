# ML Model Routes - Load, Evaluate, Predict endpoints
# Provides Streamlit-like model evaluation functionality
from fastapi import APIRouter, HTTPException, Body, Header, UploadFile, File
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import pandas as pd
import numpy as np
import logging
import json

from app.services.model_service import model_service
from app.services import state

logger = logging.getLogger(__name__)
router = APIRouter()


class PredictRequest(BaseModel):
    """Request body for prediction."""
    data: List[List[float]]  # 2D array of features
    feature_names: Optional[List[str]] = None


class EvaluateRequest(BaseModel):
    """Request body for model evaluation."""
    X: List[List[float]]  # Features
    y: List[Any]  # True labels/values
    feature_names: Optional[List[str]] = None


# ============== MODEL ROUTES ==============

@router.get("/models")
async def list_models(x_session_id: str = Header("default_session")):
    """
    List all models in the session.
    Returns both uploaded model files and loaded models ready for inference.
    """
    # Get uploaded model files from session
    datasets = state.get_all_datasets(x_session_id)
    model_files = [d for d in datasets if d.get('metadata', {}).get('is_model', False)]
    
    # Get loaded models (ready for inference)
    loaded_models = model_service.list_models()
    
    return {
        'session_id': x_session_id,
        'uploaded_models': model_files,
        'loaded_models': loaded_models
    }


@router.post("/models/{model_id}/load")
async def load_model(model_id: str, x_session_id: str = Header("default_session")):
    """
    Load a model file into memory for inference.
    The model must have been previously uploaded to the session.
    """
    # Get model file info from session
    dataset = state.get_dataset(x_session_id, model_id)
    
    if not dataset:
        raise HTTPException(404, f"Model file {model_id} not found in session")
    
    if not dataset.get('metadata', {}).get('is_model', False):
        raise HTTPException(400, f"File {model_id} is not a model file")
    
    file_path = dataset.get('file_path')
    if not file_path:
        raise HTTPException(400, "Model file path not found")
    
    # Load the model
    result = model_service.load_model(file_path, model_id)
    
    if result.get('status') == 'error':
        raise HTTPException(500, result.get('error', 'Failed to load model'))
    
    return result


@router.get("/models/{model_id}")
async def get_model_info(model_id: str, x_session_id: str = Header("default_session")):
    """
    Get information about a loaded model.
    Includes architecture, parameters, features, and task type.
    """
    info = model_service.get_model_info(model_id)
    
    if not info:
        # Try to get from session uploads
        dataset = state.get_dataset(x_session_id, model_id)
        if dataset and dataset.get('metadata', {}).get('is_model'):
            return {
                'id': model_id,
                'filename': dataset.get('filename'),
                'status': 'not_loaded',
                'message': 'Model not loaded. Call POST /api/models/{model_id}/load first.'
            }
        raise HTTPException(404, f"Model {model_id} not found")
    
    return info


@router.post("/models/{model_id}/predict")
async def predict(
    model_id: str,
    request: PredictRequest,
    x_session_id: str = Header("default_session")
):
    """
    Run predictions using a loaded model.
    
    Request body:
    - data: 2D array of feature values [[f1, f2, ...], [f1, f2, ...]]
    - feature_names: Optional list of feature names
    
    Returns predictions, probabilities (for classifiers), and confidence scores.
    """
    # Check if model is loaded
    info = model_service.get_model_info(model_id)
    if not info:
        raise HTTPException(400, f"Model {model_id} not loaded. Call POST /api/models/{model_id}/load first.")
    
    # Convert to numpy/DataFrame
    if request.feature_names:
        data = pd.DataFrame(request.data, columns=request.feature_names)
    else:
        data = np.array(request.data)
    
    # Run prediction
    result = model_service.predict(model_id, data)
    
    if 'error' in result:
        err_msg = result['error']
        if "Feature mismatch" in err_msg or "Missing features" in err_msg:
             raise HTTPException(400, err_msg)
        raise HTTPException(500, err_msg)
    
    return result


@router.post("/models/{model_id}/predict-from-dataset")
async def predict_from_dataset(
    model_id: str,
    body: Dict[str, Any] = Body(...),
    x_session_id: str = Header("default_session")
):
    """
    Run predictions using data from an uploaded dataset.
    
    Request body:
    - dataset_id: ID of the uploaded dataset to use
    - feature_columns: Optional list of column names to use as features
    - limit: Optional max rows to predict (default: all)
    
    This is useful for batch predictions on uploaded CSV/Excel files.
    """
    dataset_id = body.get('dataset_id')
    feature_columns = body.get('feature_columns')
    limit = body.get('limit')
    
    if not dataset_id:
        raise HTTPException(400, "dataset_id is required")
    
    # Get the dataset
    df = state.get_dataset_df(x_session_id, dataset_id)
    if df is None:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    
    # Select features
    if feature_columns:
        try:
            X = df[feature_columns]
        except KeyError as e:
            raise HTTPException(400, f"Column not found: {e}")
    else:
        # Use all numeric columns
        X = df.select_dtypes(include=[np.number])
    
    if limit:
        X = X.head(limit)
    
    # Check if model is loaded
    info = model_service.get_model_info(model_id)
    if not info:
        raise HTTPException(400, f"Model {model_id} not loaded. Call POST /api/models/{model_id}/load first.")
    
    # Run prediction
    result = model_service.predict(model_id, X)
    
    if 'error' in result:
        err_msg = result['error']
        if "Feature mismatch" in err_msg or "Missing features" in err_msg:
             raise HTTPException(400, err_msg)
        raise HTTPException(500, err_msg)
    
    result['dataset_id'] = dataset_id
    result['feature_columns'] = X.columns.tolist()
    
    return result


@router.post("/models/{model_id}/evaluate")
async def evaluate_model(
    model_id: str,
    request: EvaluateRequest,
    x_session_id: str = Header("default_session")
):
    """
    Evaluate model performance on test data.
    
    Request body:
    - X: 2D array of features
    - y: Array of true labels/values
    
    Returns metrics like:
    - Classification: accuracy, precision, recall, F1, confusion matrix
    - Regression: MSE, RMSE, MAE, R²
    """
    info = model_service.get_model_info(model_id)
    if not info:
        raise HTTPException(400, f"Model {model_id} not loaded. Call POST /api/models/{model_id}/load first.")
    
    X = np.array(request.X)
    y = np.array(request.y)
    
    metrics = model_service.evaluate(model_id, X, y)
    
    if 'error' in metrics:
        raise HTTPException(500, metrics['error'])
    
    return metrics


@router.post("/models/{model_id}/evaluate-from-dataset")
async def evaluate_from_dataset(
    model_id: str,
    body: Dict[str, Any] = Body(...),
    x_session_id: str = Header("default_session")
):
    """
    Evaluate model using data from an uploaded dataset.
    
    Request body:
    - dataset_id: ID of the uploaded dataset
    - feature_columns: List of column names for features
    - target_column: Column name for true labels/values
    - test_split: Optional fraction for test split (default: uses all data)
    
    This provides Streamlit-like model evaluation on uploaded data.
    """
    dataset_id = body.get('dataset_id')
    feature_columns = body.get('feature_columns')
    target_column = body.get('target_column')
    test_split = body.get('test_split', 0.2)
    
    if not dataset_id:
        raise HTTPException(400, "dataset_id is required")
    if not target_column:
        raise HTTPException(400, "target_column is required")
    
    # Get dataset
    df = state.get_dataset_df(x_session_id, dataset_id)
    if df is None:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    
    # Check target column exists
    if target_column not in df.columns:
        raise HTTPException(400, f"Target column '{target_column}' not found in dataset")
    
    # Select features
    if feature_columns:
        X = df[feature_columns]
    else:
        # All columns except target
        X = df.drop(columns=[target_column]).select_dtypes(include=[np.number])
    
    y = df[target_column]
    
    # Optional train/test split
    if test_split and test_split > 0:
        try:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_split, random_state=42
            )
            X = X_test
            y = y_test
        except ImportError:
            pass  # Use all data if sklearn not available
    
    # Check if model loaded
    info = model_service.get_model_info(model_id)
    if not info:
        raise HTTPException(400, f"Model {model_id} not loaded")
    
    # Evaluate
    metrics = model_service.evaluate(model_id, X.values, y.values)
    
    if 'error' in metrics:
        raise HTTPException(500, metrics['error'])
    
    metrics['dataset_id'] = dataset_id
    metrics['feature_columns'] = X.columns.tolist()
    metrics['target_column'] = target_column
    metrics['n_test_samples'] = len(X)
    
    return metrics


@router.delete("/models/{model_id}")
async def unload_model(model_id: str, x_session_id: str = Header("default_session")):
    """
    Unload a model from memory.
    The model file remains in the session - only the in-memory model is removed.
    """
    success = model_service.unload_model(model_id)
    
    if not success:
        raise HTTPException(404, f"Model {model_id} not loaded")
    
    return {'success': True, 'message': f'Model {model_id} unloaded'}


# ============== MODEL EVALUATION FROM PREDICTIONS FILE ==============

@router.post("/models/evaluate-predictions")
async def evaluate_predictions_file(
    file: UploadFile = File(...),
    x_session_id: str = Header("default_session")
):
    """
    Evaluate model performance from a CSV with pre-computed predictions.
    
    Upload a CSV file with columns:
    - 'actual' or 'y_true' or 'true' or 'label': The true values
    - 'predicted' or 'y_pred' or 'pred' or 'prediction': The predicted values
    - Optional: 'probability' or 'confidence': Prediction confidence
    
    Returns metrics for classification (accuracy, F1, confusion matrix) 
    or regression (MSE, R², MAE) based on detected task type.
    """
    import io
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, r2_score, mean_absolute_error,
        confusion_matrix, classification_report, roc_auc_score
    )
    
    # Read the file
    content = await file.read()
    
    try:
        # Try CSV first
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            # Try as CSV anyway
            df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Failed to parse file: {str(e)}")
    
    # Find actual column
    actual_cols = ['actual', 'y_true', 'true', 'label', 'target', 'ground_truth']
    actual_col = None
    for col in actual_cols:
        if col in df.columns.str.lower().tolist():
            actual_col = df.columns[df.columns.str.lower() == col][0]
            break
    
    if actual_col is None:
        raise HTTPException(400, f"Could not find actual/true column. Expected one of: {actual_cols}")
    
    # Find predicted column
    pred_cols = ['predicted', 'y_pred', 'pred', 'prediction', 'predictions', 'output']
    pred_col = None
    for col in pred_cols:
        if col in df.columns.str.lower().tolist():
            pred_col = df.columns[df.columns.str.lower() == col][0]
            break
    
    if pred_col is None:
        raise HTTPException(400, f"Could not find predicted column. Expected one of: {pred_cols}")
    
    y_true = df[actual_col].values
    y_pred = df[pred_col].values
    
    # Detect task type (classification vs regression)
    unique_actual = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    
    # If few unique values and integers, likely classification
    is_classification = (
        len(unique_actual) <= 20 and 
        (np.issubdtype(y_true.dtype, np.integer) or 
         all(isinstance(v, (int, str, bool)) or (isinstance(v, float) and v.is_integer()) for v in unique_actual[:100]))
    )
    
    metrics = {
        'n_samples': len(y_true),
        'filename': file.filename,
        'actual_column': actual_col,
        'predicted_column': pred_col
    }
    
    if is_classification:
        metrics['task'] = 'classification'
        
        # Convert to same type for comparison
        if y_true.dtype != y_pred.dtype:
            y_true = y_true.astype(str)
            y_pred = y_pred.astype(str)
        
        metrics['accuracy'] = round(float(accuracy_score(y_true, y_pred)) * 100, 2)
        metrics['precision'] = round(float(precision_score(y_true, y_pred, average='weighted', zero_division=0)) * 100, 2)
        metrics['recall'] = round(float(recall_score(y_true, y_pred, average='weighted', zero_division=0)) * 100, 2)
        metrics['f1_score'] = round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)) * 100, 2)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['class_labels'] = sorted([str(c) for c in unique_actual])
        metrics['n_classes'] = len(unique_actual)
        
        # Per-class metrics
        try:
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            metrics['per_class'] = {k: v for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']}
        except:
            pass
        
        # Prediction distribution
        pred_counts = pd.Series(y_pred).value_counts().to_dict()
        metrics['prediction_distribution'] = {str(k): int(v) for k, v in pred_counts.items()}
        
    else:
        metrics['task'] = 'regression'
        
        # Convert to float
        y_true = y_true.astype(float)
        y_pred = y_pred.astype(float)
        
        metrics['mse'] = round(float(mean_squared_error(y_true, y_pred)), 4)
        metrics['rmse'] = round(float(np.sqrt(metrics['mse'])), 4)
        metrics['mae'] = round(float(mean_absolute_error(y_true, y_pred)), 4)
        metrics['r2'] = round(float(r2_score(y_true, y_pred)), 4)
        
        # Scatter plot data (sample for large datasets)
        if len(y_true) > 500:
            indices = np.random.choice(len(y_true), 500, replace=False)
            scatter_actual = y_true[indices]
            scatter_pred = y_pred[indices]
        else:
            scatter_actual = y_true
            scatter_pred = y_pred
        
        metrics['scatter_data'] = [
            {'actual': float(a), 'predicted': float(p)} 
            for a, p in zip(scatter_actual, scatter_pred)
        ]
        
        # Residuals distribution
        residuals = y_true - y_pred
        metrics['residual_stats'] = {
            'mean': round(float(residuals.mean()), 4),
            'std': round(float(residuals.std()), 4),
            'min': round(float(residuals.min()), 4),
            'max': round(float(residuals.max()), 4)
        }
    
    # Store in session for dashboard
    state.store_evaluation(x_session_id, metrics)
    
    return metrics


@router.post("/models/{model_id}/evaluate-with-file")
async def evaluate_model_with_file(
    model_id: str,
    file: UploadFile = File(...),
    x_session_id: str = Header("default_session")
):
    """
    Upload a test dataset and evaluate the loaded model.
    
    The CSV/Excel should have feature columns matching the model's expected input.
    If a 'target' or 'label' column exists, it will be used for evaluation.
    Otherwise, only predictions are returned.
    
    This enables running inference on new data directly.
    """
    import io
    
    # Check if model is loaded
    info = model_service.get_model_info(model_id)
    if not info:
        # Try to load it first
        dataset = state.get_dataset(x_session_id, model_id)
        if not dataset:
            raise HTTPException(404, f"Model {model_id} not found in session")
        
        if dataset.get('metadata', {}).get('is_model'):
            file_path = dataset.get('file_path')
            load_result = model_service.load_model(file_path, model_id)
            
            if load_result.get('status') == 'error':
                 raise HTTPException(400, load_result.get('error', f"Failed to load model {model_id}"))
            
            info = model_service.get_model_info(model_id)
        
        if not info:
            raise HTTPException(400, f"Failed to load model {model_id} - unknown error")
    
    # Read test data
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
    
    # Find target column if exists
    target_cols = ['target', 'label', 'y', 'actual', 'class', 'output']
    target_col = None
    for col in target_cols:
        if col in df.columns.str.lower().tolist():
            target_col = df.columns[df.columns.str.lower() == col][0]
            break
    
    # Get features (all numeric columns except target)
    if target_col:
        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
        y_true = df[target_col].values
    else:
        X = df.select_dtypes(include=[np.number])
        y_true = None
    
    if X.empty:
        raise HTTPException(400, "No numeric feature columns found in the file")
    
    # Run predictions
    result = model_service.predict(model_id, X)
    
    if 'error' in result:
        err_msg = result['error']
        if "Feature mismatch" in err_msg or "Missing features" in err_msg:
             raise HTTPException(400, err_msg)
        raise HTTPException(500, err_msg)
    
    # If we have true labels, compute metrics
    if y_true is not None:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            mean_squared_error, r2_score, mean_absolute_error, confusion_matrix
        )
        
        y_pred = np.array(result['predictions'])
        task = info.get('task', 'classification')
        
        result['has_evaluation'] = True
        result['n_test_samples'] = len(y_true)
        
        if task == 'classification':
            result['accuracy'] = round(float(accuracy_score(y_true, y_pred)) * 100, 2)
            result['precision'] = round(float(precision_score(y_true, y_pred, average='weighted', zero_division=0)) * 100, 2)
            result['recall'] = round(float(recall_score(y_true, y_pred, average='weighted', zero_division=0)) * 100, 2)
            result['f1_score'] = round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)) * 100, 2)
            result['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        else:
            y_true_float = y_true.astype(float)
            y_pred_float = np.array(y_pred).astype(float)
            result['mse'] = round(float(mean_squared_error(y_true_float, y_pred_float)), 4)
            result['rmse'] = round(float(np.sqrt(result['mse'])), 4)
            result['mae'] = round(float(mean_absolute_error(y_true_float, y_pred_float)), 4)
            result['r2'] = round(float(r2_score(y_true_float, y_pred_float)), 4)
    else:
        result['has_evaluation'] = False
        result['message'] = 'No target column found. Showing predictions only.'
    
    result['feature_columns'] = X.columns.tolist()
    result['filename'] = file.filename
    
    # Store in session for dashboard
    state.store_evaluation(x_session_id, result)
    
    return result


# ============== QUICK INFERENCE ENDPOINT ==============

@router.post("/models/quick-predict")
async def quick_predict(
    file: UploadFile = File(...),
    body: str = Body(None),
    x_session_id: str = Header("default_session")
):
    """
    Quick prediction endpoint - upload a model and data together.
    
    This is a convenience endpoint for one-off predictions.
    For repeated predictions, use the load/predict workflow.
    """
    import tempfile
    import os
    from uuid import uuid4
    
    # Save model temporarily
    temp_id = str(uuid4())
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else 'pkl'
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Load model
        result = model_service.load_model(tmp_path, temp_id)
        
        if result.get('status') == 'error':
            raise HTTPException(500, result.get('error'))
        
        # Parse data from body if provided
        predictions = None
        if body:
            data = json.loads(body)
            if 'data' in data:
                predictions = model_service.predict(temp_id, np.array(data['data']))
        
        response = {
            'model_info': result,
            'predictions': predictions
        }
        
        return response
        
    finally:
        # Cleanup
        model_service.unload_model(temp_id)
        os.unlink(tmp_path)

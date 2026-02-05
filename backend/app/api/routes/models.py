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
        raise HTTPException(500, result['error'])
    
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
        raise HTTPException(500, result['error'])
    
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

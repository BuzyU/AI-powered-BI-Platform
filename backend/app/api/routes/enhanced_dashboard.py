# Enhanced Dashboard API Routes
# Comprehensive persona-aware dashboard generation

from fastapi import APIRouter, Header, HTTPException, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging

from app.services import state
from app.services.model_service import model_service
from app.services.dashboard_template_service import dashboard_template_service
from app.layers.l2_classification.enhanced_persona_detector import (
    enhanced_persona_detector, UserPersona, DashboardType
)
from app.layers.l7_analytics.metrics_calculator import (
    column_detector, metrics_calculator, statistical_analyzer, business_calculator
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/dashboard", tags=["dashboard"])


class RefreshRequest(BaseModel):
    """Request to refresh dashboard with specific options."""
    force_regenerate: bool = False
    include_evaluation: bool = True


@router.get("/adaptive")
def get_adaptive_dashboard(
    force_refresh: bool = Query(False, description="Force regenerate dashboard"),
    dashboard_type: Optional[str] = Query(None, description="Manually selected dashboard type"),
    x_session_id: str = Header("default_session")
):
    """
    Generate a persona-aware adaptive dashboard.
    """
    # Get session data
    datasets = state.get_all_datasets(x_session_id)
    if not datasets:
        raise HTTPException(400, "No datasets in session. Upload data first.")
    
    # Check cache (unless force refresh or manual override)
    if not force_refresh and not dashboard_type:
        cached = state.get_dashboard_config(x_session_id)
        if cached and cached.get('version') == '2.0':  # New enhanced version
            return cached
    
    # Prepare datasets with dataframes
    datasets_with_df = []
    model_files = []
    
    for ds in datasets:
        df = state.get_dataset_df(x_session_id, ds['id'])
        is_model = ds.get('metadata', {}).get('is_model', False)
        
        if is_model:
            model_files.append(ds)
            continue  # Skip models for persona detection
        
        if df is not None:
            datasets_with_df.append({
                'id': ds['id'],
                'filename': ds.get('filename', ''),
                'df': df,
                'metadata': ds.get('metadata', {})
            })
    
    # Get model info if available
    model_info = None
    if model_files:
        model_info = {
            'is_model': True,
            'id': model_files[0]['id'],
            'filename': model_files[0].get('filename', ''),
            'model_type': model_files[0].get('metadata', {}).get('model_type', 'Unknown'),
            'file_type': model_files[0].get('file_type', '')
        }
    
    # If only model, no data
    if not datasets_with_df and model_files:
        return _generate_model_only_dashboard(model_info)
    
    # Detect persona using enhanced detector
    persona_result = enhanced_persona_detector.detect_persona(
        datasets_with_df,
        model_info=model_info
    )
    
    # MANUAL OVERRIDE: If dashboard_type is provided, override the result
    if dashboard_type:
        logger.info(f"Manual dashboard override: {dashboard_type}")
        persona_result.dashboard_type = DashboardType(dashboard_type)
        # Map dashboard type back to a plausible persona for consistency
        type_to_persona = {
            'power_bi_style': UserPersona.BUSINESS,
            'eda_analytics': UserPersona.ANALYTICS,
            'ml_metrics': UserPersona.ML_ENGINEER,
            'data_science': UserPersona.DATA_SCIENTIST,
            'developer_dashboard': UserPersona.DEVELOPER,
            'cv_dashboard': UserPersona.COMPUTER_VISION if hasattr(UserPersona, 'COMPUTER_VISION') else UserPersona.ML_ENGINEER
        }
        if dashboard_type in type_to_persona:
            persona_result.persona = type_to_persona[dashboard_type]
            persona_result.summary = f"Manually switched to {dashboard_type.replace('_', ' ').title()} view."
            
    
    logger.info(f"Detected persona: {persona_result.persona.value}, "
                f"confidence: {persona_result.confidence:.2f}, "
                f"dashboard type: {persona_result.dashboard_type.value if hasattr(persona_result.dashboard_type, 'value') else persona_result.dashboard_type}")
    
    # Get evaluation results if available
    evaluation = state.get_evaluation(x_session_id)
    
    # Generate dashboard using template service
    dashboard = dashboard_template_service.generate_dashboard(
        persona_result=persona_result.to_dict(),
        datasets=datasets_with_df,
        evaluation_result=evaluation,
        model_info=model_info
    )
    
    # Add metadata
    dashboard['version'] = '2.0'
    dashboard['session_id'] = x_session_id
    dashboard['persona_detection'] = persona_result.to_dict()
    
    # Add model info
    dashboard['model_info'] = model_info
    dashboard['has_model'] = model_info is not None
    
    # Add evaluation info
    dashboard['has_evaluation'] = evaluation is not None
    if evaluation:
        dashboard['evaluation_summary'] = {
            'task_type': evaluation.get('task_type'),
            'n_samples': evaluation.get('n_samples'),
            'accuracy': evaluation.get('overall', {}).get('accuracy') or evaluation.get('accuracy'),
            'f1_score': evaluation.get('overall', {}).get('f1_weighted') or evaluation.get('f1_score'),
            'r2': evaluation.get('core', {}).get('r2_score') or evaluation.get('r2')
        }
    
    # Cache it
    state.store_dashboard_config(x_session_id, dashboard)
    
    return dashboard


def _generate_model_only_dashboard(model_info: Dict) -> Dict[str, Any]:
    """Generate dashboard when only a model is uploaded."""
    return {
        'version': '2.0',
        'type': 'model_only',
        'theme': 'technical',
        'layout': 'ml',
        'title': 'Model Dashboard',
        'subtitle': 'Upload prediction data to see model performance metrics',
        'model_info': model_info,
        'has_model': True,
        'has_evaluation': False,
        'sections': [
            {
                'id': 'model_section',
                'title': 'Uploaded Model',
                'type': 'model_info',
                'data': model_info
            },
            {
                'id': 'evaluation_prompt',
                'title': 'Evaluate Model',
                'type': 'prompt',
                'content': {
                    'message': 'Upload a CSV with actual/predicted columns to evaluate model performance.',
                    'actions': [
                        {
                            'label': 'Upload Predictions CSV',
                            'endpoint': '/api/evaluate/predictions',
                            'method': 'POST'
                        },
                        {
                            'label': 'Run Model on Test Data',
                            'endpoint': f'/api/models/{model_info["id"]}/evaluate-with-file',
                            'method': 'POST'
                        }
                    ]
                }
            }
        ],
        'persona_detection': {
            'persona': 'ml_engineer',
            'confidence': 0.9,
            'detected_domains': ['Machine Learning'],
            'dashboard_type': 'ml_metrics'
        }
    }


@router.get("/detect-persona")
def detect_persona(x_session_id: str = Header("default_session")):
    """
    Detect the user persona based on uploaded data without generating full dashboard.
    
    Useful for:
    - Understanding what type of data was uploaded
    - Getting recommendations for analysis
    - Debugging persona detection
    """
    datasets = state.get_all_datasets(x_session_id)
    if not datasets:
        raise HTTPException(400, "No datasets in session. Upload data first.")
    
    datasets_with_df = []
    model_files = []
    
    for ds in datasets:
        df = state.get_dataset_df(x_session_id, ds['id'])
        is_model = ds.get('metadata', {}).get('is_model', False)
        
        if is_model:
            model_files.append(ds)
            continue
        
        if df is not None:
            datasets_with_df.append({
                'id': ds['id'],
                'filename': ds.get('filename', ''),
                'df': df,
                'metadata': ds.get('metadata', {})
            })
    
    model_info = None
    if model_files:
        model_info = {
            'is_model': True,
            'model_type': model_files[0].get('metadata', {}).get('model_type', 'Unknown')
        }
    
    persona_result = enhanced_persona_detector.detect_persona(
        datasets_with_df,
        model_info=model_info
    )
    
    return {
        **persona_result.to_dict(),
        'has_model': model_info is not None,
        'model_info': model_info,
        'datasets_analyzed': len(datasets_with_df)
    }


@router.get("/statistics")
def get_data_statistics(
    dataset_id: Optional[str] = Query(None, description="Specific dataset ID"),
    x_session_id: str = Header("default_session")
):
    """
    Get comprehensive statistical analysis of the data.
    
    Returns calculated statistics including:
    - Distribution analysis (mean, median, std, skewness, kurtosis)
    - Outlier detection (IQR method)
    - Missing data analysis
    - Correlation analysis
    - Normality tests
    
    All values are calculated from actual data, not predefined.
    """
    datasets = state.get_all_datasets(x_session_id)
    if not datasets:
        raise HTTPException(400, "No datasets in session. Upload data first.")
    
    # Get specific dataset or first one
    if dataset_id:
        df = state.get_dataset_df(x_session_id, dataset_id)
        if df is None:
            raise HTTPException(404, f"Dataset {dataset_id} not found")
    else:
        # Get first non-model dataset
        df = None
        for ds in datasets:
            if not ds.get('metadata', {}).get('is_model'):
                df = state.get_dataset_df(x_session_id, ds['id'])
                if df is not None:
                    dataset_id = ds['id']
                    break
    
    if df is None:
        raise HTTPException(400, "No valid dataset found for analysis")
    
    # Calculate comprehensive statistics
    result = {
        'dataset_id': dataset_id,
        'shape': {'rows': len(df), 'columns': len(df.columns)},
        'columns': {}
    }
    
    # Analyze each column
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    for col in numeric_cols[:20]:  # Limit for performance
        analysis = statistical_analyzer.analyze_distribution(df[col])
        result['columns'][col] = {
            'dtype': str(df[col].dtype),
            **analysis
        }
    
    # Correlation analysis
    if len(numeric_cols) >= 2:
        result['correlations'] = statistical_analyzer.correlation_analysis(df)
    
    # Overall data quality
    total_missing = df.isna().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    
    result['data_quality'] = {
        'missing_values': int(total_missing),
        'missing_percentage': round(total_missing / total_cells * 100, 2),
        'duplicate_rows': int(df.duplicated().sum()),
        'unique_columns': {col: int(df[col].nunique()) for col in df.columns[:20]}
    }
    
    return result


@router.get("/business-metrics")
def get_business_metrics(
    dataset_id: Optional[str] = Query(None),
    x_session_id: str = Header("default_session")
):
    """
    Get business-specific metrics (for Power BI style dashboard).
    
    Automatically detects and calculates:
    - Revenue/Sales totals and averages
    - Profit margins and cost ratios
    - Volume metrics
    - Growth trends
    
    All metrics are calculated from actual data columns.
    """
    datasets = state.get_all_datasets(x_session_id)
    if not datasets:
        raise HTTPException(400, "No datasets in session. Upload data first.")
    
    # Get dataset
    df = None
    for ds in datasets:
        if not ds.get('metadata', {}).get('is_model'):
            if dataset_id and ds['id'] != dataset_id:
                continue
            df = state.get_dataset_df(x_session_id, ds['id'])
            if df is not None:
                dataset_id = ds['id']
                break
    
    if df is None:
        raise HTTPException(400, "No valid dataset found")
    
    # Calculate business metrics
    metrics = business_calculator.calculate_financial_metrics(df)
    metrics['dataset_id'] = dataset_id
    metrics['columns_detected'] = {
        'revenue': [c for c in df.columns if any(
            kw in c.lower() for kw in ['revenue', 'sales', 'amount']
        )],
        'cost': [c for c in df.columns if any(
            kw in c.lower() for kw in ['cost', 'expense']
        )],
        'quantity': [c for c in df.columns if any(
            kw in c.lower() for kw in ['quantity', 'count', 'units']
        )]
    }
    
    return metrics


@router.post("/refresh")
def refresh_dashboard(
    request: RefreshRequest,
    x_session_id: str = Header("default_session")
):
    """
    Refresh the dashboard with new analysis.
    
    Options:
    - force_regenerate: Clear cache and regenerate
    - include_evaluation: Include ML evaluation if available
    """
    if request.force_regenerate:
        state.clear_dashboard_config(x_session_id)
    
    # Redirect to main endpoint
    return get_adaptive_dashboard(
        force_refresh=request.force_regenerate,
        x_session_id=x_session_id
    )


@router.get("/types")
async def get_dashboard_types():
    """
    List all available dashboard types and their descriptions.
    
    Helps users understand what kind of dashboard they'll get.
    """
    return {
        'dashboard_types': [
            {
                'type': 'power_bi_style',
                'name': 'Business Intelligence',
                'description': 'Power BI style dashboard with KPIs, revenue charts, profit analysis',
                'best_for': 'Sales data, financial data, customer data',
                'features': ['KPI cards', 'Revenue trends', 'Profit analysis', 'Category breakdowns']
            },
            {
                'type': 'eda_analytics',
                'name': 'Exploratory Data Analysis',
                'description': 'Statistical analysis dashboard with distributions and correlations',
                'best_for': 'Research data, survey data, general analytics',
                'features': ['Distribution plots', 'Correlation matrix', 'Outlier detection', 'Missing data analysis']
            },
            {
                'type': 'ml_metrics',
                'name': 'ML Performance',
                'description': 'Machine learning model evaluation dashboard',
                'best_for': 'Model predictions, classification/regression results',
                'features': ['Accuracy metrics', 'Confusion matrix', 'ROC/PR curves', 'Error analysis']
            },
            {
                'type': 'cv_dashboard',
                'name': 'Computer Vision',
                'description': 'Computer vision model evaluation dashboard',
                'best_for': 'Object detection, image classification results',
                'features': ['mAP/IoU metrics', 'Class-wise performance', 'Confidence distribution']
            },
            {
                'type': 'data_science',
                'name': 'Data Science',
                'description': 'Combined EDA and ML preparation dashboard',
                'best_for': 'Data exploration before modeling',
                'features': ['Data profiling', 'Feature analysis', 'ML readiness score']
            }
        ],
        'personas': [
            {'id': 'business', 'name': 'Business User', 'default_dashboard': 'power_bi_style'},
            {'id': 'analytics', 'name': 'Data Analyst', 'default_dashboard': 'eda_analytics'},
            {'id': 'ml_engineer', 'name': 'ML Engineer', 'default_dashboard': 'ml_metrics'},
            {'id': 'computer_vision', 'name': 'CV Engineer', 'default_dashboard': 'cv_dashboard'},
            {'id': 'data_scientist', 'name': 'Data Scientist', 'default_dashboard': 'data_science'}
        ]
    }

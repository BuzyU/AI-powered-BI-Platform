# Session-Based Dashboard API Routes
# Smart Dashboards Generated from Content Classification

from fastapi import APIRouter, Header, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import numpy as np
from pathlib import Path

from app.services import state
from app.services.model_service import model_service
from app.layers.l2_classification.persona_detector import persona_detector
from app.layers.l7_analytics.adaptive_dashboard import adaptive_dashboard_generator
from app.services.analyzer import DataAnalyzer

router = APIRouter()
analyzer = DataAnalyzer()


def _render_model_info(session_id: str, spec: Dict) -> Dict:
    """Render model information chart."""
    datasets = state.get_all_datasets(session_id)
    model_files = [d for d in datasets if d.get('metadata', {}).get('is_model')]
    
    if not model_files:
        return {'chart_id': spec['id'], 'type': 'kpi', 'value': 'No model', 'formatted_value': 'No model found'}
    
    model = model_files[0]
    model_type = model.get('metadata', {}).get('model_type', 'Unknown')
    filename = model.get('filename', 'Unknown')
    
    return {
        'chart_id': spec['id'],
        'title': spec.get('title', 'Model Type'),
        'type': 'kpi',
        'value': model_type,
        'formatted_value': model_type,
        'subtitle': filename,
        'icon': 'cpu'
    }


def _render_model_only_chart(session_id: str, datasets: List[Dict], spec: Dict) -> Dict:
    """Render charts for model-only sessions."""
    chart_type = spec.get('type', 'kpi')
    chart_id = spec.get('id')
    title = spec.get('title', '')
    agg = spec.get('aggregation', '')
    
    model_files = [d for d in datasets if d.get('metadata', {}).get('is_model')]
    
    if chart_type == 'kpi':
        if agg == 'model_info' or 'model' in chart_id.lower():
            model = model_files[0] if model_files else {}
            model_type = model.get('metadata', {}).get('model_type', 'Unknown')
            return {
                'chart_id': chart_id,
                'title': title,
                'type': 'kpi',
                'value': model_type,
                'formatted_value': model_type,
                'icon': 'cpu'
            }
        elif 'sample' in chart_id.lower() or 'count' in agg:
            return {
                'chart_id': chart_id,
                'title': title,
                'type': 'kpi',
                'value': 0,
                'formatted_value': 'No data',
                'subtitle': 'Upload a dataset to evaluate',
                'icon': 'database'
            }
        else:
            return {
                'chart_id': chart_id,
                'title': title,
                'type': 'kpi',
                'value': 'N/A',
                'formatted_value': 'N/A',
                'subtitle': 'Upload data to calculate',
                'icon': 'help-circle'
            }
    
    elif chart_type == 'gauge':
        return {
            'chart_id': chart_id,
            'title': title,
            'type': 'gauge',
            'value': 0,
            'max': 100,
            'message': 'Upload evaluation data to calculate accuracy',
            'thresholds': [60, 80, 90]
        }
    
    elif chart_type == 'heatmap':
        return {
            'chart_id': chart_id,
            'title': title,
            'type': 'heatmap',
            'data': [],
            'message': 'Upload prediction data to see confusion matrix'
        }
    
    elif chart_type in ['bar', 'histogram']:
        return {
            'chart_id': chart_id,
            'title': title,
            'type': chart_type,
            'data': [],
            'message': 'Upload data to see distribution'
        }
    
    elif chart_type == 'table':
        return {
            'chart_id': chart_id,
            'title': title,
            'type': 'table',
            'columns': [],
            'data': [],
            'message': 'Upload data to see predictions'
        }
    
    # Default response
    return {
        'chart_id': chart_id,
        'title': title,
        'type': chart_type,
        'data': [],
        'message': 'Upload a dataset to see this visualization'
    }


class ChartDataRequest(BaseModel):
    chart_id: str
    tab_id: str
    filters: Optional[Dict[str, Any]] = None


@router.get("/session-dashboard")
async def get_session_dashboard(x_session_id: str = Header("default_session")):
    """
    Generate a smart adaptive dashboard based on WHO is using it and WHAT they uploaded.
    
    - Business User → Revenue, Customers, Products KPIs
    - Analytics User → EDA, Distributions, Correlations  
    - ML Engineer → Model Metrics, Confusion Matrix
    """
    # Get session data
    datasets = state.get_all_datasets(x_session_id)
    if not datasets:
        raise HTTPException(400, "No datasets in session. Upload data first.")
    
    # Check for cached dashboard config
    cached_config = state.get_dashboard_config(x_session_id)
    if cached_config:
        return cached_config
    
    # Prepare datasets with dataframes
    datasets_with_df = []
    model_files = []
    
    for ds in datasets:
        df = state.get_dataset_df(x_session_id, ds['id'])
        is_model = ds.get('metadata', {}).get('is_model', False)
        
        if is_model:
            model_files.append(ds)
        
        datasets_with_df.append({
            'id': ds['id'],
            'filename': ds.get('filename', ''),
            'df': df,
            'metadata': ds.get('metadata', {})
        })
    
    # Detect persona
    persona_result = persona_detector.detect_persona(datasets_with_df)
    
    # Generate adaptive dashboard
    dashboard = adaptive_dashboard_generator.generate(datasets_with_df, persona_result)
    config_dict = adaptive_dashboard_generator.to_dict(dashboard)
    
    # Add persona detection info
    config_dict['persona_detection'] = {
        'persona': persona_result.persona.value,
        'confidence': persona_result.confidence,
        'detected_domains': persona_result.detected_domains,
        'recommended_analysis': persona_result.recommended_analysis,
        'summary': persona_result.summary
    }
    
    # Add model info if models are present
    if model_files:
        config_dict['has_model'] = True
        config_dict['model_id'] = model_files[0]['id']
        config_dict['model_info'] = {
            'id': model_files[0]['id'],
            'filename': model_files[0].get('filename', ''),
            'model_type': model_files[0].get('metadata', {}).get('model_type', 'Unknown'),
            'file_type': model_files[0].get('file_type', '')
        }
    else:
        config_dict['has_model'] = False
        config_dict['model_id'] = None
        config_dict['model_info'] = None
    
    # Add evaluation results if available
    evaluation = state.get_evaluation(x_session_id)
    if evaluation:
        config_dict['has_evaluation'] = True
        config_dict['evaluation_summary'] = {
            'task': evaluation.get('task'),
            'accuracy': evaluation.get('accuracy'),
            'f1_score': evaluation.get('f1_score'),
            'r2': evaluation.get('r2'),
            'n_samples': evaluation.get('n_samples')
        }
    else:
        config_dict['has_evaluation'] = False
    
    # Cache it
    state.store_dashboard_config(x_session_id, config_dict)
    
    return config_dict


@router.post("/session-dashboard/chart-data")
async def get_chart_data(
    request: ChartDataRequest,
    x_session_id: str = Header("default_session")
):
    """
    Get actual data for a specific chart.
    """
    # Get dashboard config
    config = state.get_dashboard_config(x_session_id)
    if not config:
        raise HTTPException(400, "Dashboard not generated. Call GET /session-dashboard first.")
    
    # Find the chart spec
    chart_spec = None
    for tab in config.get('tabs', []):
        if tab['id'] == request.tab_id:
            for chart in tab.get('charts', []):
                if chart['id'] == request.chart_id:
                    chart_spec = chart
                    break
    
    if not chart_spec:
        raise HTTPException(404, f"Chart {request.chart_id} not found in tab {request.tab_id}")
    
    # Handle model-only sessions - return model info charts
    if chart_spec.get('aggregation') == 'model_info':
        return _render_model_info(x_session_id, chart_spec)
    
    # Get dataset(s)
    datasets = state.get_all_datasets(x_session_id)
    if not datasets:
        raise HTTPException(400, "No datasets available")
    
    # Check if it's model-only session
    has_data = any(not ds.get('metadata', {}).get('is_model') for ds in datasets)
    
    # Get first non-model dataset
    df = None
    for ds in datasets:
        if not ds.get('metadata', {}).get('is_model'):
            df = state.get_dataset_df(x_session_id, ds['id'])
            if df is not None:
                break
    
    if df is None and not has_data:
        # Model-only session - return placeholder data
        return _render_model_only_chart(x_session_id, datasets, chart_spec)
    
    # Apply filters
    if request.filters:
        for col, val in request.filters.items():
            if col in df.columns:
                if isinstance(val, list):
                    df = df[df[col].isin(val)]
                else:
                    df = df[df[col] == val]
    
    # Generate chart data based on type
    try:
        return _render_chart(df, chart_spec, config.get('dashboard_type', 'general'))
    except Exception as e:
        return {
            'chart_id': chart_spec['id'],
            'error': str(e),
            'data': []
        }


def _render_chart(df, spec: Dict, dashboard_type: str) -> Dict:
    """Render chart data based on spec."""
    chart_type = spec.get('type', 'bar')
    chart_id = spec['id']
    title = spec['title']
    x_col = spec.get('x_column')
    y_col = spec.get('y_column')
    agg = spec.get('aggregation', 'sum')
    config = spec.get('config', {})
    
    # KPI Charts
    if chart_type == 'kpi':
        return _render_kpi(df, spec)
    
    # Table
    if chart_type == 'table':
        return _render_table(df, spec)
    
    # Histogram
    if chart_type == 'histogram':
        return _render_histogram(df, x_col, spec)
    
    # Box Plot
    if chart_type == 'boxplot':
        return _render_boxplot(df, x_col, spec)
    
    # Heatmap (Correlation Matrix or Confusion Matrix)
    if chart_type == 'heatmap':
        if agg == 'confusion':
            return _render_confusion_matrix(df, x_col, y_col, spec)
        return _render_correlation_matrix(df, spec)
    
    # Scatter Plot
    if chart_type == 'scatter':
        return _render_scatter(df, x_col, y_col, spec)
    
    # Gauge
    if chart_type == 'gauge':
        return _render_gauge(df, x_col, y_col, spec)
    
    # Bar/Line/Pie/Area charts
    return _render_aggregated_chart(df, chart_type, x_col, y_col, agg, spec)


def _render_kpi(df, spec: Dict) -> Dict:
    """Render KPI value."""
    agg = spec.get('aggregation', 'count')
    y_col = spec.get('y_column')
    x_col = spec.get('x_column')
    config = spec.get('config', {})
    
    value = 0
    
    if agg == 'count':
        value = len(df)
    elif agg == 'count_distinct' and x_col:
        value = df[x_col].nunique() if x_col in df.columns else 0
    elif agg == 'column_count':
        value = len(df.columns)
    elif agg == 'completeness':
        value = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    elif agg == 'sum' and y_col and y_col in df.columns:
        value = df[y_col].sum()
    elif agg == 'avg' and y_col and y_col in df.columns:
        value = df[y_col].mean()
    elif agg in ['accuracy', 'precision', 'recall', 'f1'] and x_col and y_col:
        value = _calculate_metric(df, x_col, y_col, agg)
    elif agg == 'error_rate' and x_col and y_col:
        value = _calculate_metric(df, x_col, y_col, 'error_rate')
    
    # Format value
    fmt = config.get('format', 'number')
    if fmt == 'currency':
        if value >= 1_000_000:
            formatted = f"${value/1_000_000:.1f}M"
        elif value >= 1_000:
            formatted = f"${value/1_000:.1f}K"
        else:
            formatted = f"${value:.2f}"
    elif fmt == 'percent':
        formatted = f"{value:.1f}%"
    else:
        formatted = f"{value:,.0f}" if isinstance(value, (int, float)) else str(value)
    
    return {
        'chart_id': spec['id'],
        'title': spec['title'],
        'type': 'kpi',
        'value': float(value) if isinstance(value, (int, float, np.number)) else 0,
        'formatted_value': formatted,
        'icon': config.get('icon', 'hash')
    }


def _render_table(df, spec: Dict) -> Dict:
    """Render data table."""
    config = spec.get('config', {})
    limit = config.get('limit', 100)
    
    # Filter errors only if specified
    if config.get('filter') == 'errors':
        x_col = spec.get('x_column')
        y_col = spec.get('y_column')
        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            df = df[df[x_col] != df[y_col]]
    
    sample = df.head(limit)
    
    return {
        'chart_id': spec['id'],
        'title': spec['title'],
        'type': 'table',
        'columns': list(sample.columns),
        'data': sample.to_dict(orient='records'),
        'total_rows': len(df)
    }


def _render_histogram(df, x_col: str, spec: Dict) -> Dict:
    """Render histogram."""
    if not x_col or x_col not in df.columns:
        return {'chart_id': spec['id'], 'type': 'histogram', 'data': [], 'error': 'Column not found'}
    
    values = df[x_col].dropna()
    
    try:
        hist, bin_edges = np.histogram(values, bins=20)
        data = [
            {'bin': f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}", 'count': int(hist[i])}
            for i in range(len(hist))
        ]
    except:
        data = []
    
    return {
        'chart_id': spec['id'],
        'title': spec['title'],
        'type': 'histogram',
        'x_label': x_col,
        'data': data
    }


def _render_boxplot(df, x_col: str, spec: Dict) -> Dict:
    """Render box plot statistics."""
    if not x_col or x_col not in df.columns:
        return {'chart_id': spec['id'], 'type': 'boxplot', 'data': [], 'error': 'Column not found'}
    
    values = df[x_col].dropna()
    
    try:
        q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        data = {
            'min': float(values.min()),
            'q1': float(q1),
            'median': float(median),
            'q3': float(q3),
            'max': float(values.max()),
            'lower_fence': float(max(lower, values.min())),
            'upper_fence': float(min(upper, values.max())),
            'outliers': values[(values < lower) | (values > upper)].tolist()[:50]
        }
    except:
        data = {}
    
    return {
        'chart_id': spec['id'],
        'title': spec['title'],
        'type': 'boxplot',
        'column': x_col,
        'data': data
    }


def _render_correlation_matrix(df, spec: Dict) -> Dict:
    """Render correlation matrix."""
    config = spec.get('config', {})
    cols = config.get('columns', [])
    
    if not cols:
        # Use all numeric columns
        cols = df.select_dtypes(include=[np.number]).columns.tolist()[:10]
    
    valid_cols = [c for c in cols if c in df.columns]
    
    if len(valid_cols) < 2:
        return {'chart_id': spec['id'], 'type': 'heatmap', 'data': [], 'error': 'Need at least 2 numeric columns'}
    
    try:
        corr = df[valid_cols].corr()
        data = []
        for i, row in enumerate(valid_cols):
            for j, col in enumerate(valid_cols):
                data.append({
                    'x': col,
                    'y': row,
                    'value': round(corr.iloc[i, j], 2)
                })
    except:
        data = []
    
    return {
        'chart_id': spec['id'],
        'title': spec['title'],
        'type': 'heatmap',
        'columns': valid_cols,
        'data': data
    }


def _render_confusion_matrix(df, pred_col: str, actual_col: str, spec: Dict) -> Dict:
    """Render confusion matrix."""
    if not pred_col or not actual_col:
        return {'chart_id': spec['id'], 'type': 'heatmap', 'data': [], 'error': 'Need prediction and actual columns'}
    
    if pred_col not in df.columns or actual_col not in df.columns:
        return {'chart_id': spec['id'], 'type': 'heatmap', 'data': [], 'error': 'Columns not found'}
    
    try:
        from sklearn.metrics import confusion_matrix
        labels = sorted(df[actual_col].unique())
        cm = confusion_matrix(df[actual_col], df[pred_col], labels=labels)
        
        data = []
        for i, actual in enumerate(labels):
            for j, pred in enumerate(labels):
                data.append({
                    'x': str(pred),
                    'y': str(actual),
                    'value': int(cm[i, j])
                })
    except:
        # Fallback without sklearn
        data = []
    
    return {
        'chart_id': spec['id'],
        'title': spec['title'],
        'type': 'heatmap',
        'data': data
    }


def _render_scatter(df, x_col: str, y_col: str, spec: Dict) -> Dict:
    """Render scatter plot."""
    if not x_col or not y_col:
        return {'chart_id': spec['id'], 'type': 'scatter', 'data': []}
    
    if x_col not in df.columns or y_col not in df.columns:
        return {'chart_id': spec['id'], 'type': 'scatter', 'data': [], 'error': 'Columns not found'}
    
    sample = df[[x_col, y_col]].dropna().head(500)
    data = [{'x': float(row[x_col]), 'y': float(row[y_col])} for _, row in sample.iterrows()]
    
    return {
        'chart_id': spec['id'],
        'title': spec['title'],
        'type': 'scatter',
        'x_label': x_col,
        'y_label': y_col,
        'data': data
    }


def _render_gauge(df, x_col: str, y_col: str, spec: Dict) -> Dict:
    """Render gauge (for accuracy etc.)."""
    config = spec.get('config', {})
    agg = spec.get('aggregation', 'accuracy')
    
    value = _calculate_metric(df, x_col, y_col, agg)
    
    return {
        'chart_id': spec['id'],
        'title': spec['title'],
        'type': 'gauge',
        'value': round(value, 1),
        'max': config.get('max', 100),
        'thresholds': config.get('thresholds', [60, 80, 90])
    }


def _render_aggregated_chart(df, chart_type: str, x_col: str, y_col: str, agg: str, spec: Dict) -> Dict:
    """Render bar/line/pie/area charts with aggregation."""
    config = spec.get('config', {})
    
    if not x_col or x_col not in df.columns:
        # Fallback: use first categorical column
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        x_col = cat_cols[0] if len(cat_cols) > 0 else df.columns[0]
    
    # Aggregate data
    if y_col and y_col in df.columns:
        if agg == 'sum':
            grouped = df.groupby(x_col)[y_col].sum()
        elif agg == 'avg':
            grouped = df.groupby(x_col)[y_col].mean()
        elif agg == 'min':
            grouped = df.groupby(x_col)[y_col].min()
        elif agg == 'max':
            grouped = df.groupby(x_col)[y_col].max()
        else:
            grouped = df.groupby(x_col)[y_col].sum()
    else:
        grouped = df.groupby(x_col).size()
        y_col = 'Count'
    
    # Sort and limit
    sort_order = config.get('sort', 'desc')
    limit = config.get('limit', 20)
    
    if sort_order == 'desc':
        grouped = grouped.sort_values(ascending=False)
    elif sort_order == 'asc':
        grouped = grouped.sort_values(ascending=True)
    
    grouped = grouped.head(limit)
    
    # Format data
    data = [{'label': str(k), 'value': float(v)} for k, v in grouped.items()]
    
    return {
        'chart_id': spec['id'],
        'title': spec['title'],
        'type': chart_type,
        'x_label': x_col,
        'y_label': y_col,
        'data': data
    }


def _calculate_metric(df, pred_col: str, actual_col: str, metric: str) -> float:
    """Calculate ML metrics."""
    if pred_col not in df.columns or actual_col not in df.columns:
        return 0.0
    
    try:
        y_true = df[actual_col]
        y_pred = df[pred_col]
        
        if metric == 'accuracy':
            return (y_true == y_pred).mean() * 100
        elif metric == 'error_rate':
            return (y_true != y_pred).mean() * 100
        elif metric == 'precision':
            from sklearn.metrics import precision_score
            return precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        elif metric == 'recall':
            from sklearn.metrics import recall_score
            return recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        elif metric == 'f1':
            from sklearn.metrics import f1_score
            return f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    except:
        pass
    
    return 0.0


@router.post("/session-dashboard/regenerate")
async def regenerate_dashboard(x_session_id: str = Header("default_session")):
    """Force regenerate the dashboard config."""
    state.clear_dashboard_config(x_session_id)
    return await get_session_dashboard(x_session_id)


@router.get("/session-dashboard/tabs/{tab_id}")
async def get_tab_with_data(
    tab_id: str,
    x_session_id: str = Header("default_session")
):
    """Get a specific tab with all its chart data pre-loaded."""
    config = state.get_dashboard_config(x_session_id)
    if not config:
        config = await get_session_dashboard(x_session_id)
    
    # Find tab
    target_tab = None
    for tab in config.get('tabs', []):
        if tab['id'] == tab_id:
            target_tab = tab
            break
    
    if not target_tab:
        raise HTTPException(404, f"Tab {tab_id} not found")
    
    # Load data for all charts
    charts_with_data = []
    for chart in target_tab.get('charts', []):
        chart_data = await get_chart_data(
            ChartDataRequest(chart_id=chart['id'], tab_id=tab_id),
            x_session_id
        )
        charts_with_data.append({**chart, 'chart_data': chart_data})
    
    return {
        'tab_id': target_tab['id'],
        'title': target_tab['title'],
        'icon': target_tab['icon'],
        'description': target_tab.get('description', ''),
        'charts': charts_with_data
    }


@router.get("/session-dashboard/persona")
async def get_detected_persona(x_session_id: str = Header("default_session")):
    """Get the detected persona and recommendations."""
    config = state.get_dashboard_config(x_session_id)
    if config and 'persona_detection' in config:
        return config['persona_detection']
    
    # Detect fresh
    datasets = state.get_all_datasets(x_session_id)
    datasets_with_df = []
    for ds in datasets:
        df = state.get_dataset_df(x_session_id, ds['id'])
        datasets_with_df.append({
            'id': ds['id'],
            'filename': ds.get('filename', ''),
            'df': df,
            'metadata': ds.get('metadata', {})
        })
    
    result = persona_detector.detect_persona(datasets_with_df)
    
    return {
        'persona': result.persona.value,
        'confidence': result.confidence,
        'detected_domains': result.detected_domains,
        'recommended_analysis': result.recommended_analysis,
        'dashboard_type': result.dashboard_type,
        'summary': result.summary
    }

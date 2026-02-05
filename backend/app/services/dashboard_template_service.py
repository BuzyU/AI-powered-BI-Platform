# Dashboard Template Service
# Generates dynamic dashboard configurations based on persona and data

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging

from app.layers.l2_classification.enhanced_persona_detector import (
    UserPersona, DataCategory, DashboardType, enhanced_persona_detector
)
from app.layers.l7_analytics.metrics_calculator import (
    BusinessMetricsCalculator, StatisticalAnalyzer, business_calculator, statistical_analyzer
)

logger = logging.getLogger(__name__)


@dataclass
class ChartConfig:
    """Configuration for a single chart."""
    id: str
    type: str  # 'bar', 'line', 'pie', 'scatter', 'heatmap', 'table', 'kpi', etc.
    title: str
    subtitle: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None


class DashboardTemplateService:
    """
    Generates dynamic dashboards based on:
    - User persona (Business, Analytics, ML, CV)
    - Data characteristics
    - Detected domains
    """
    
    def generate_dashboard(
        self,
        persona_result: Dict[str, Any],
        datasets: List[Dict[str, Any]],
        evaluation_result: Optional[Dict[str, Any]] = None,
        model_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete dashboard configuration.
        
        Args:
            persona_result: Result from persona detection
            datasets: List of uploaded datasets
            evaluation_result: Optional ML evaluation results
            model_info: Optional model information
        
        Returns:
            Complete dashboard configuration with charts, KPIs, and layout
        """
        dashboard_type = persona_result.get('dashboard_type', 'general')
        persona = persona_result.get('persona', 'unknown')
        
        # Route to appropriate generator
        if dashboard_type == 'power_bi_style':
            return self._generate_business_dashboard(datasets, persona_result)
        elif dashboard_type == 'eda_analytics':
            return self._generate_analytics_dashboard(datasets, persona_result)
        elif dashboard_type == 'ml_metrics':
            return self._generate_ml_dashboard(datasets, persona_result, evaluation_result, model_info)
        elif dashboard_type == 'cv_dashboard':
            return self._generate_cv_dashboard(datasets, persona_result, evaluation_result, model_info)
        elif dashboard_type == 'data_science':
            return self._generate_datascience_dashboard(datasets, persona_result, evaluation_result)
        else:
            return self._generate_general_dashboard(datasets, persona_result)
    
    def _generate_business_dashboard(
        self, 
        datasets: List[Dict[str, Any]], 
        persona_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate Power BI style business dashboard."""
        dashboard = {
            'type': 'business_intelligence',
            'theme': 'professional',
            'layout': 'business',
            'title': 'Business Intelligence Dashboard',
            'subtitle': persona_result.get('summary', ''),
            'sections': []
        }
        
        # Combine all dataframes
        dfs = [ds.get('df') for ds in datasets if ds.get('df') is not None]
        if not dfs:
            return self._empty_dashboard('No data available')
        
        combined_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        
        # Calculate business metrics
        financial_metrics = business_calculator.calculate_financial_metrics(combined_df)
        
        # KPI Section
        kpi_section = self._generate_business_kpis(combined_df, financial_metrics)
        dashboard['sections'].append(kpi_section)
        
        # Revenue/Sales Charts
        revenue_section = self._generate_revenue_charts(combined_df)
        if revenue_section:
            dashboard['sections'].append(revenue_section)
        
        # Distribution Charts
        dist_section = self._generate_business_distributions(combined_df)
        if dist_section:
            dashboard['sections'].append(dist_section)
        
        # Trend Analysis
        trend_section = self._generate_trend_analysis(combined_df)
        if trend_section:
            dashboard['sections'].append(trend_section)
        
        # Data Table
        table_section = self._generate_data_table(combined_df, title="Recent Transactions")
        dashboard['sections'].append(table_section)
        
        return dashboard
    
    def _generate_analytics_dashboard(
        self, 
        datasets: List[Dict[str, Any]], 
        persona_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate EDA/Analytics dashboard."""
        dashboard = {
            'type': 'exploratory_analytics',
            'theme': 'analytical',
            'layout': 'analytics',
            'title': 'Exploratory Data Analysis Dashboard',
            'subtitle': persona_result.get('summary', ''),
            'sections': []
        }
        
        dfs = [ds.get('df') for ds in datasets if ds.get('df') is not None]
        if not dfs:
            return self._empty_dashboard('No data available')
        
        df = dfs[0]  # Use first dataset for EDA
        
        # Data Overview Section
        overview_section = self._generate_data_overview(df)
        dashboard['sections'].append(overview_section)
        
        # Statistical Summary
        stats_section = self._generate_statistical_summary(df)
        dashboard['sections'].append(stats_section)
        
        # Distribution Analysis
        dist_section = self._generate_distribution_analysis(df)
        if dist_section:
            dashboard['sections'].append(dist_section)
        
        # Correlation Analysis
        corr_section = self._generate_correlation_analysis(df)
        if corr_section:
            dashboard['sections'].append(corr_section)
        
        # Outlier Detection
        outlier_section = self._generate_outlier_analysis(df)
        if outlier_section:
            dashboard['sections'].append(outlier_section)
        
        # Missing Data Analysis
        missing_section = self._generate_missing_data_analysis(df)
        if missing_section:
            dashboard['sections'].append(missing_section)
        
        return dashboard
    
    def _generate_ml_dashboard(
        self, 
        datasets: List[Dict[str, Any]], 
        persona_result: Dict[str, Any],
        evaluation_result: Optional[Dict[str, Any]] = None,
        model_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate ML Performance dashboard."""
        dashboard = {
            'type': 'ml_performance',
            'theme': 'technical',
            'layout': 'ml',
            'title': 'Model Performance Dashboard',
            'subtitle': persona_result.get('summary', ''),
            'sections': []
        }
        
        # Model Info Section
        if model_info:
            model_section = self._generate_model_info_section(model_info)
            dashboard['sections'].append(model_section)
        
        # If we have evaluation results, use them
        if evaluation_result:
            # Metrics Section
            metrics_section = self._generate_ml_metrics_section(evaluation_result)
            dashboard['sections'].append(metrics_section)
            
            # Confusion Matrix (for classification)
            if 'confusion_matrix' in evaluation_result:
                cm_section = self._generate_confusion_matrix_section(evaluation_result)
                dashboard['sections'].append(cm_section)
            
            # ROC/PR Curves
            if 'roc_curve' in evaluation_result or 'pr_curve' in evaluation_result:
                curves_section = self._generate_curves_section(evaluation_result)
                dashboard['sections'].append(curves_section)
            
            # Per-Class Metrics
            if 'per_class' in evaluation_result:
                per_class_section = self._generate_per_class_section(evaluation_result)
                dashboard['sections'].append(per_class_section)
            
            # Error Analysis
            if 'error_analysis' in evaluation_result:
                error_section = self._generate_error_analysis_section(evaluation_result)
                dashboard['sections'].append(error_section)
            
            # Regression-specific
            if 'scatter_data' in evaluation_result:
                scatter_section = self._generate_regression_scatter_section(evaluation_result)
                dashboard['sections'].append(scatter_section)
            
            if 'residuals' in evaluation_result:
                residual_section = self._generate_residual_analysis_section(evaluation_result)
                dashboard['sections'].append(residual_section)
        else:
            # No evaluation yet - show evaluation prompt
            prompt_section = {
                'id': 'evaluation_prompt',
                'title': 'Model Evaluation',
                'type': 'prompt',
                'content': {
                    'message': 'Upload a predictions file to see model performance metrics.',
                    'action': 'Upload Predictions CSV',
                    'endpoint': '/api/evaluate/predictions'
                }
            }
            dashboard['sections'].append(prompt_section)
        
        return dashboard
    
    def _generate_cv_dashboard(
        self, 
        datasets: List[Dict[str, Any]], 
        persona_result: Dict[str, Any],
        evaluation_result: Optional[Dict[str, Any]] = None,
        model_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate Computer Vision dashboard."""
        dashboard = {
            'type': 'computer_vision',
            'theme': 'technical',
            'layout': 'cv',
            'title': 'Computer Vision Model Dashboard',
            'subtitle': persona_result.get('summary', ''),
            'sections': []
        }
        
        # Model Info
        if model_info:
            model_section = self._generate_model_info_section(model_info, cv_specific=True)
            dashboard['sections'].append(model_section)
        
        if evaluation_result:
            # CV-specific metrics
            cv_metrics_section = self._generate_cv_metrics_section(evaluation_result)
            dashboard['sections'].append(cv_metrics_section)
            
            # Confusion Matrix
            if 'confusion_matrix' in evaluation_result:
                cm_section = self._generate_confusion_matrix_section(evaluation_result)
                dashboard['sections'].append(cm_section)
            
            # IoU Distribution
            if 'iou_distribution' in evaluation_result or 'iou_metrics' in evaluation_result:
                iou_section = self._generate_iou_section(evaluation_result)
                dashboard['sections'].append(iou_section)
            
            # Confidence Distribution
            if 'confidence_metrics' in evaluation_result:
                conf_section = self._generate_confidence_section(evaluation_result)
                dashboard['sections'].append(conf_section)
            
            # Per-Class Performance
            if 'per_class' in evaluation_result:
                per_class_section = self._generate_per_class_section(evaluation_result)
                dashboard['sections'].append(per_class_section)
        else:
            # Prompt for evaluation
            prompt_section = {
                'id': 'cv_evaluation_prompt',
                'title': 'Model Evaluation',
                'type': 'prompt',
                'content': {
                    'message': 'Upload detection/classification results to see CV metrics.',
                    'action': 'Upload Predictions',
                    'endpoint': '/api/evaluate/predictions'
                }
            }
            dashboard['sections'].append(prompt_section)
        
        return dashboard
    
    def _generate_datascience_dashboard(
        self, 
        datasets: List[Dict[str, Any]], 
        persona_result: Dict[str, Any],
        evaluation_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate Data Science dashboard (mix of EDA + ML)."""
        dashboard = {
            'type': 'data_science',
            'theme': 'modern',
            'layout': 'data_science',
            'title': 'Data Science Dashboard',
            'subtitle': persona_result.get('summary', ''),
            'sections': []
        }
        
        dfs = [ds.get('df') for ds in datasets if ds.get('df') is not None]
        if not dfs:
            return self._empty_dashboard('No data available')
        
        df = dfs[0]
        
        # Data Profile
        profile_section = self._generate_data_profile(df)
        dashboard['sections'].append(profile_section)
        
        # Feature Analysis
        feature_section = self._generate_feature_analysis(df)
        if feature_section:
            dashboard['sections'].append(feature_section)
        
        # ML Readiness
        readiness_section = self._generate_ml_readiness(df)
        dashboard['sections'].append(readiness_section)
        
        # If evaluation results exist, include them
        if evaluation_result:
            metrics_section = self._generate_ml_metrics_section(evaluation_result)
            dashboard['sections'].append(metrics_section)
        
        return dashboard
    
    def _generate_general_dashboard(
        self, 
        datasets: List[Dict[str, Any]], 
        persona_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate general purpose dashboard."""
        dashboard = {
            'type': 'general',
            'theme': 'default',
            'layout': 'general',
            'title': 'Data Analysis Dashboard',
            'subtitle': persona_result.get('summary', ''),
            'sections': []
        }
        
        dfs = [ds.get('df') for ds in datasets if ds.get('df') is not None]
        if not dfs:
            return self._empty_dashboard('No data available')
        
        df = dfs[0]
        
        # Basic overview
        overview = self._generate_data_overview(df)
        dashboard['sections'].append(overview)
        
        # Simple stats
        stats = self._generate_basic_stats(df)
        dashboard['sections'].append(stats)
        
        # Data table
        table = self._generate_data_table(df)
        dashboard['sections'].append(table)
        
        return dashboard
    
    # ==================== Helper Methods ====================
    
    def _empty_dashboard(self, message: str) -> Dict[str, Any]:
        """Return empty dashboard with message."""
        return {
            'type': 'empty',
            'title': 'No Data',
            'sections': [{
                'id': 'empty_message',
                'type': 'message',
                'content': message
            }]
        }
    
    def _generate_business_kpis(self, df: pd.DataFrame, metrics: Dict) -> Dict[str, Any]:
        """Generate KPI cards for business dashboard."""
        kpis = []
        
        # Revenue KPI
        if 'revenue' in metrics:
            kpis.append({
                'id': 'revenue',
                'label': 'Total Revenue',
                'value': f"${metrics['revenue']['total']:,.2f}",
                'subtitle': f"Avg: ${metrics['revenue']['average']:,.2f}",
                'icon': '💰',
                'trend': metrics.get('growth', {}).get('trend', 'stable')
            })
        
        # Profit KPI
        if 'profitability' in metrics:
            profit = metrics['profitability']['gross_profit']
            margin = metrics['profitability']['profit_margin']
            kpis.append({
                'id': 'profit',
                'label': 'Gross Profit',
                'value': f"${profit:,.2f}",
                'subtitle': f"Margin: {margin:.1f}%",
                'icon': '📈',
                'trend': 'up' if profit > 0 else 'down'
            })
        
        # Volume KPI
        if 'volume' in metrics:
            kpis.append({
                'id': 'volume',
                'label': 'Total Units',
                'value': f"{metrics['volume']['total']:,}",
                'subtitle': f"Avg: {metrics['volume']['average']:.1f}",
                'icon': '📦'
            })
        
        # Add row count if no other KPIs
        if not kpis:
            kpis.append({
                'id': 'records',
                'label': 'Total Records',
                'value': f"{len(df):,}",
                'subtitle': f"Columns: {len(df.columns)}",
                'icon': '📊'
            })
        
        return {
            'id': 'kpi_section',
            'title': 'Key Performance Indicators',
            'type': 'kpi_grid',
            'charts': kpis
        }
    
    def _generate_revenue_charts(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate revenue/sales related charts."""
        revenue_cols = [c for c in df.columns if any(
            kw in c.lower() for kw in ['revenue', 'sales', 'amount', 'total', 'value']
        ) and pd.api.types.is_numeric_dtype(df[c])]
        
        if not revenue_cols:
            return None
        
        col = revenue_cols[0]
        
        # Try to find a category column for breakdown
        category_cols = [c for c in df.columns if any(
            kw in c.lower() for kw in ['category', 'type', 'product', 'region', 'segment']
        ) and df[c].nunique() <= 20]
        
        charts = []
        
        if category_cols:
            cat_col = category_cols[0]
            breakdown = df.groupby(cat_col)[col].sum().sort_values(ascending=False).head(10)
            
            charts.append({
                'id': 'revenue_by_category',
                'type': 'bar',
                'title': f'{col} by {cat_col}',
                'data': {
                    'labels': breakdown.index.tolist(),
                    'values': breakdown.values.tolist()
                }
            })
        
        # Distribution of revenue values
        charts.append({
            'id': 'revenue_distribution',
            'type': 'histogram',
            'title': f'{col} Distribution',
            'data': {
                'values': df[col].dropna().tolist()[:1000]  # Limit for performance
            }
        })
        
        return {
            'id': 'revenue_section',
            'title': 'Revenue Analysis',
            'type': 'chart_grid',
            'charts': charts
        }
    
    def _generate_business_distributions(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate distribution charts for business data."""
        # Find categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        cat_cols = [c for c in cat_cols if df[c].nunique() <= 15][:3]  # Max 3 charts
        
        if not cat_cols:
            return None
        
        charts = []
        for col in cat_cols:
            counts = df[col].value_counts().head(10)
            charts.append({
                'id': f'dist_{col}',
                'type': 'pie',
                'title': f'{col} Distribution',
                'data': {
                    'labels': counts.index.tolist(),
                    'values': counts.values.tolist()
                }
            })
        
        return {
            'id': 'distribution_section',
            'title': 'Category Distributions',
            'type': 'chart_grid',
            'charts': charts
        }
    
    def _generate_trend_analysis(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate time-based trend charts."""
        # Find date columns
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        if not date_cols:
            # Try to parse string columns
            for col in df.columns:
                if any(kw in col.lower() for kw in ['date', 'time', 'period']):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        if df[col].notna().sum() > 0:
                            date_cols.append(col)
                            break
                    except:
                        pass
        
        if not date_cols:
            return None
        
        date_col = date_cols[0]
        
        # Find numeric column to trend
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        value_cols = [c for c in numeric_cols if any(
            kw in c.lower() for kw in ['revenue', 'sales', 'amount', 'count', 'total']
        )]
        
        if not value_cols:
            value_cols = list(numeric_cols[:1])
        
        if not value_cols:
            return None
        
        value_col = value_cols[0]
        
        # Aggregate by date
        df_temp = df[[date_col, value_col]].dropna()
        df_temp = df_temp.set_index(date_col).resample('D').sum().reset_index()
        
        return {
            'id': 'trend_section',
            'title': 'Trend Analysis',
            'type': 'chart_grid',
            'charts': [{
                'id': 'time_trend',
                'type': 'line',
                'title': f'{value_col} Over Time',
                'data': {
                    'labels': df_temp[date_col].dt.strftime('%Y-%m-%d').tolist()[-100:],
                    'values': df_temp[value_col].tolist()[-100:]
                }
            }]
        }
    
    def _generate_data_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data overview section."""
        n_numeric = len(df.select_dtypes(include=[np.number]).columns)
        n_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
        n_datetime = len(df.select_dtypes(include=['datetime64']).columns)
        
        missing_pct = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        
        return {
            'id': 'overview_section',
            'title': 'Data Overview',
            'type': 'stats_grid',
            'stats': [
                {'label': 'Rows', 'value': f"{len(df):,}", 'icon': '📝'},
                {'label': 'Columns', 'value': str(len(df.columns)), 'icon': '📊'},
                {'label': 'Numeric', 'value': str(n_numeric), 'icon': '🔢'},
                {'label': 'Categorical', 'value': str(n_categorical), 'icon': '🏷️'},
                {'label': 'Datetime', 'value': str(n_datetime), 'icon': '📅'},
                {'label': 'Missing %', 'value': f"{missing_pct:.1f}%", 'icon': '❓'},
                {'label': 'Memory', 'value': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB", 'icon': '💾'},
                {'label': 'Duplicates', 'value': f"{df.duplicated().sum():,}", 'icon': '🔄'}
            ]
        }
    
    def _generate_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical summary for numeric columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {
                'id': 'stats_section',
                'title': 'Statistical Summary',
                'type': 'message',
                'content': 'No numeric columns found'
            }
        
        stats = []
        for col in numeric_df.columns[:10]:  # Limit to 10 columns
            analysis = statistical_analyzer.analyze_distribution(numeric_df[col])
            stats.append({
                'column': col,
                'mean': analysis.get('basic', {}).get('mean'),
                'median': analysis.get('basic', {}).get('median'),
                'std': analysis.get('basic', {}).get('std'),
                'min': analysis.get('basic', {}).get('min'),
                'max': analysis.get('basic', {}).get('max'),
                'skewness': analysis.get('shape', {}).get('skewness'),
                'outliers': analysis.get('outliers', {}).get('count', 0)
            })
        
        return {
            'id': 'stats_section',
            'title': 'Statistical Summary',
            'type': 'stats_table',
            'data': stats
        }
    
    def _generate_distribution_analysis(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate distribution charts for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
        
        if len(numeric_cols) == 0:
            return None
        
        charts = []
        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) == 0:
                continue
            
            hist, bin_edges = np.histogram(values, bins='auto')
            charts.append({
                'id': f'dist_{col}',
                'type': 'histogram',
                'title': f'{col}',
                'data': {
                    'counts': hist.tolist(),
                    'bins': bin_edges.tolist()
                },
                'analysis': {
                    'skewness': round(float(values.skew()), 4),
                    'kurtosis': round(float(values.kurtosis()), 4)
                }
            })
        
        return {
            'id': 'distribution_section',
            'title': 'Distribution Analysis',
            'subtitle': 'Histograms of numeric columns with skewness/kurtosis',
            'type': 'chart_grid',
            'charts': charts
        }
    
    def _generate_correlation_analysis(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate correlation matrix."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return None
        
        # Limit columns for readability
        if numeric_df.shape[1] > 15:
            numeric_df = numeric_df.iloc[:, :15]
        
        corr_analysis = statistical_analyzer.correlation_analysis(numeric_df)
        
        return {
            'id': 'correlation_section',
            'title': 'Correlation Analysis',
            'type': 'heatmap',
            'data': {
                'matrix': corr_analysis.get('matrix', {}),
                'columns': corr_analysis.get('columns', [])
            },
            'highlights': corr_analysis.get('strong_correlations', [])[:5]
        }
    
    def _generate_outlier_analysis(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate outlier detection summary."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return None
        
        outlier_info = []
        for col in numeric_df.columns[:10]:
            q1 = numeric_df[col].quantile(0.25)
            q3 = numeric_df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum()
            pct = outliers / len(numeric_df) * 100
            
            if outliers > 0:
                outlier_info.append({
                    'column': col,
                    'count': int(outliers),
                    'percentage': round(pct, 2),
                    'lower_bound': round(lower, 4),
                    'upper_bound': round(upper, 4)
                })
        
        return {
            'id': 'outlier_section',
            'title': 'Outlier Detection (IQR Method)',
            'type': 'outlier_table',
            'data': sorted(outlier_info, key=lambda x: x['count'], reverse=True),
            'total_outlier_columns': len(outlier_info)
        }
    
    def _generate_missing_data_analysis(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate missing data analysis."""
        missing = df.isna().sum()
        missing = missing[missing > 0]
        
        if missing.empty:
            return {
                'id': 'missing_section',
                'title': 'Missing Data',
                'type': 'message',
                'content': '✅ No missing values found in the dataset!'
            }
        
        missing_data = []
        for col in missing.index:
            missing_data.append({
                'column': col,
                'missing_count': int(missing[col]),
                'missing_pct': round(missing[col] / len(df) * 100, 2)
            })
        
        return {
            'id': 'missing_section',
            'title': 'Missing Data Analysis',
            'type': 'missing_table',
            'data': sorted(missing_data, key=lambda x: x['missing_pct'], reverse=True),
            'chart': {
                'type': 'bar',
                'data': {
                    'labels': [d['column'] for d in missing_data[:10]],
                    'values': [d['missing_pct'] for d in missing_data[:10]]
                }
            }
        }
    
    def _generate_ml_metrics_section(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ML metrics section."""
        task_type = evaluation.get('task_type', 'classification')
        
        if task_type in ['binary_classification', 'multiclass_classification', 'classification']:
            overall = evaluation.get('overall', {})
            metrics = [
                {'label': 'Accuracy', 'value': f"{overall.get('accuracy', 0):.1f}%", 'icon': '🎯'},
                {'label': 'Precision', 'value': f"{overall.get('precision_weighted', overall.get('precision', 0)):.1f}%", 'icon': '🔍'},
                {'label': 'Recall', 'value': f"{overall.get('recall_weighted', overall.get('recall', 0)):.1f}%", 'icon': '📈'},
                {'label': 'F1 Score', 'value': f"{overall.get('f1_weighted', overall.get('f1_score', 0)):.1f}%", 'icon': '⚖️'},
            ]
            
            if 'balanced_accuracy' in overall:
                metrics.append({'label': 'Balanced Acc', 'value': f"{overall['balanced_accuracy']:.1f}%", 'icon': '🔄'})
            if 'roc_auc' in overall:
                metrics.append({'label': 'ROC-AUC', 'value': f"{overall['roc_auc']:.1f}%", 'icon': '📊'})
            if 'matthews_corrcoef' in overall:
                metrics.append({'label': 'MCC', 'value': f"{overall['matthews_corrcoef']:.4f}", 'icon': '📐'})
        else:
            core = evaluation.get('core', {})
            metrics = [
                {'label': 'R² Score', 'value': f"{core.get('r2_score', 0):.4f}", 'icon': '📈'},
                {'label': 'RMSE', 'value': f"{core.get('rmse', 0):.4f}", 'icon': '📊'},
                {'label': 'MAE', 'value': f"{core.get('mae', 0):.4f}", 'icon': '📉'},
                {'label': 'MSE', 'value': f"{core.get('mse', 0):.4f}", 'icon': '🔢'},
            ]
            
            if 'mape' in core:
                metrics.append({'label': 'MAPE', 'value': f"{core['mape']:.2f}%", 'icon': '📐'})
            if 'explained_variance' in core:
                metrics.append({'label': 'Explained Var', 'value': f"{core['explained_variance']:.4f}", 'icon': '📊'})
        
        return {
            'id': 'metrics_section',
            'title': 'Model Performance Metrics',
            'subtitle': f"Task: {task_type} | Samples: {evaluation.get('n_samples', 0):,}",
            'type': 'metric_cards',
            'metrics': metrics
        }
    
    def _generate_confusion_matrix_section(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate confusion matrix visualization."""
        cm_data = evaluation.get('confusion_matrix', {})
        
        if isinstance(cm_data, dict):
            matrix = cm_data.get('matrix', [])
            labels = cm_data.get('labels', [])
            normalized = cm_data.get('normalized', [])
        else:
            matrix = cm_data
            labels = evaluation.get('class_labels', evaluation.get('class_names', []))
            normalized = None
        
        return {
            'id': 'confusion_matrix_section',
            'title': 'Confusion Matrix',
            'type': 'confusion_matrix',
            'data': {
                'matrix': matrix,
                'labels': labels,
                'normalized': normalized
            }
        }
    
    def _generate_curves_section(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ROC and PR curve section."""
        charts = []
        
        if 'roc_curve' in evaluation:
            charts.append({
                'id': 'roc_curve',
                'type': 'line',
                'title': 'ROC Curve',
                'data': evaluation['roc_curve']
            })
        
        if 'pr_curve' in evaluation:
            charts.append({
                'id': 'pr_curve',
                'type': 'line',
                'title': 'Precision-Recall Curve',
                'data': evaluation['pr_curve']
            })
        
        return {
            'id': 'curves_section',
            'title': 'Performance Curves',
            'type': 'chart_grid',
            'charts': charts
        }
    
    def _generate_per_class_section(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate per-class metrics table."""
        per_class = evaluation.get('per_class', {})
        
        data = []
        for class_name, metrics in per_class.items():
            data.append({
                'class': class_name,
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', metrics.get('f1-score', 0)),
                'support': metrics.get('support', 0)
            })
        
        return {
            'id': 'per_class_section',
            'title': 'Per-Class Performance',
            'type': 'metrics_table',
            'data': data
        }
    
    def _generate_error_analysis_section(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate error analysis section."""
        error = evaluation.get('error_analysis', {})
        
        return {
            'id': 'error_section',
            'title': 'Error Analysis',
            'type': 'error_analysis',
            'data': {
                'total_errors': error.get('total_errors', 0),
                'error_rate': error.get('error_rate', 0),
                'confused_pairs': error.get('most_confused_pairs', [])
            }
        }
    
    def _generate_regression_scatter_section(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate scatter plot for regression."""
        scatter_data = evaluation.get('scatter_data', [])
        
        return {
            'id': 'scatter_section',
            'title': 'Actual vs Predicted',
            'type': 'scatter',
            'data': scatter_data
        }
    
    def _generate_residual_analysis_section(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate residual analysis section."""
        residuals = evaluation.get('residuals', {})
        histogram = evaluation.get('residual_histogram', {})
        
        return {
            'id': 'residual_section',
            'title': 'Residual Analysis',
            'type': 'residual_analysis',
            'stats': residuals,
            'histogram': histogram
        }
    
    def _generate_cv_metrics_section(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate CV-specific metrics section."""
        overall = evaluation.get('overall', {})
        iou_metrics = evaluation.get('iou_metrics', {})
        
        metrics = [
            {'label': 'Accuracy', 'value': f"{overall.get('accuracy', 0):.1f}%", 'icon': '🎯'},
            {'label': 'Precision', 'value': f"{overall.get('precision_weighted', 0):.1f}%", 'icon': '🔍'},
            {'label': 'Recall', 'value': f"{overall.get('recall_weighted', 0):.1f}%", 'icon': '📈'},
            {'label': 'F1 Score', 'value': f"{overall.get('f1_weighted', 0):.1f}%", 'icon': '⚖️'},
        ]
        
        if iou_metrics:
            metrics.extend([
                {'label': 'Mean IoU', 'value': f"{iou_metrics.get('mean_iou', 0):.4f}", 'icon': '📐'},
                {'label': 'IoU > 0.5', 'value': f"{iou_metrics.get('iou_above_50', 0):.1f}%", 'icon': '✅'},
                {'label': 'IoU > 0.75', 'value': f"{iou_metrics.get('iou_above_75', 0):.1f}%", 'icon': '⭐'},
            ])
        
        return {
            'id': 'cv_metrics_section',
            'title': 'Computer Vision Metrics',
            'type': 'metric_cards',
            'metrics': metrics
        }
    
    def _generate_iou_section(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate IoU distribution section."""
        iou_dist = evaluation.get('iou_distribution', {})
        iou_metrics = evaluation.get('iou_metrics', {})
        
        return {
            'id': 'iou_section',
            'title': 'IoU Distribution',
            'type': 'histogram',
            'data': iou_dist,
            'stats': iou_metrics
        }
    
    def _generate_confidence_section(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate confidence distribution section."""
        conf_metrics = evaluation.get('confidence_metrics', {})
        
        return {
            'id': 'confidence_section',
            'title': 'Confidence Analysis',
            'type': 'confidence_analysis',
            'stats': conf_metrics
        }
    
    def _generate_model_info_section(
        self, 
        model_info: Dict[str, Any], 
        cv_specific: bool = False
    ) -> Dict[str, Any]:
        """Generate model information section."""
        return {
            'id': 'model_info_section',
            'title': 'Model Information',
            'type': 'model_info',
            'data': {
                'filename': model_info.get('filename', 'Unknown'),
                'model_type': model_info.get('model_type', 'Unknown'),
                'framework': model_info.get('framework', 'Unknown'),
                'task': model_info.get('task', 'classification'),
                'input_shape': model_info.get('input_shape'),
                'output_classes': model_info.get('output_classes'),
                'parameters': model_info.get('n_parameters')
            }
        }
    
    def _generate_data_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data profile."""
        profile = {
            'shape': {'rows': len(df), 'columns': len(df.columns)},
            'types': df.dtypes.astype(str).to_dict(),
            'missing': df.isna().sum().to_dict(),
            'unique': df.nunique().to_dict()
        }
        
        return {
            'id': 'profile_section',
            'title': 'Data Profile',
            'type': 'data_profile',
            'data': profile
        }
    
    def _generate_feature_analysis(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Analyze features for ML readiness."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return None
        
        features = []
        for col in numeric_cols[:20]:
            features.append({
                'name': col,
                'dtype': str(df[col].dtype),
                'missing_pct': round(df[col].isna().mean() * 100, 2),
                'unique': int(df[col].nunique()),
                'mean': round(float(df[col].mean()), 4) if not df[col].isna().all() else None,
                'std': round(float(df[col].std()), 4) if not df[col].isna().all() else None
            })
        
        return {
            'id': 'feature_section',
            'title': 'Feature Analysis',
            'type': 'feature_table',
            'data': features
        }
    
    def _generate_ml_readiness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess ML readiness of the dataset."""
        issues = []
        score = 100
        
        # Check missing values
        missing_pct = df.isna().mean().mean() * 100
        if missing_pct > 0:
            score -= min(20, missing_pct)
            if missing_pct > 5:
                issues.append(f"High missing values: {missing_pct:.1f}%")
        
        # Check for high cardinality categoricals
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > len(df) * 0.5:
                score -= 5
                issues.append(f"High cardinality column: {col}")
        
        # Check duplicates
        dup_pct = df.duplicated().mean() * 100
        if dup_pct > 5:
            score -= 10
            issues.append(f"High duplicate rows: {dup_pct:.1f}%")
        
        # Check for target column
        target_cols = [c for c in df.columns if any(
            kw in c.lower() for kw in ['target', 'label', 'class', 'y']
        )]
        
        suggestions = []
        if not target_cols:
            suggestions.append("No obvious target column found. Define your prediction target.")
        
        if missing_pct > 0:
            suggestions.append("Consider imputation strategies for missing values.")
        
        return {
            'id': 'readiness_section',
            'title': 'ML Readiness Score',
            'type': 'readiness_score',
            'score': max(0, score),
            'issues': issues,
            'suggestions': suggestions
        }
    
    def _generate_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic statistics."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {
                'id': 'stats_section',
                'title': 'Basic Statistics',
                'type': 'message',
                'content': 'No numeric columns available'
            }
        
        return {
            'id': 'stats_section',
            'title': 'Basic Statistics',
            'type': 'describe_table',
            'data': numeric_df.describe().round(4).to_dict()
        }
    
    def _generate_data_table(
        self, 
        df: pd.DataFrame, 
        title: str = "Data Preview"
    ) -> Dict[str, Any]:
        """Generate data table preview."""
        sample = df.head(100).fillna('').to_dict('records')
        
        return {
            'id': 'table_section',
            'title': title,
            'type': 'data_table',
            'data': {
                'columns': df.columns.tolist(),
                'rows': sample,
                'total_rows': len(df)
            }
        }


# Singleton
dashboard_template_service = DashboardTemplateService()

# Adaptive Dashboard Generator
# Creates completely different dashboards based on WHO is using and WHAT they uploaded

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from enum import Enum

from app.layers.l2_classification.persona_detector import (
    UserPersona, DataCategory, PersonaDetectionResult, persona_detector
)


@dataclass
class ChartConfig:
    """Configuration for a single chart."""
    id: str
    title: str
    type: str  # kpi, bar, line, pie, scatter, heatmap, table, histogram, boxplot, gauge
    size: str = "medium"  # small, medium, large, full
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    aggregation: str = "sum"
    description: str = ""
    config: Dict = field(default_factory=dict)


@dataclass 
class DashboardTab:
    """A tab/section in the dashboard."""
    id: str
    title: str
    icon: str
    charts: List[ChartConfig] = field(default_factory=list)
    description: str = ""


@dataclass
class AdaptiveDashboard:
    """Full adaptive dashboard configuration."""
    dashboard_id: str
    title: str
    subtitle: str
    persona: UserPersona
    dashboard_type: str
    tabs: List[DashboardTab] = field(default_factory=list)
    global_filters: List[Dict] = field(default_factory=list)
    kpi_summary: List[Dict] = field(default_factory=list)
    ai_insights: List[str] = field(default_factory=list)
    color_scheme: str = "default"


class AdaptiveDashboardGenerator:
    """
    Generates completely different dashboards based on detected persona.
    
    - Business User → Revenue, Customers, Products, Branches KPIs
    - Analytics User → EDA, Distributions, Correlations, Statistics
    - ML Engineer → Model Metrics, Confusion Matrix, Feature Importance
    """
    
    def generate(
        self,
        datasets: List[Dict[str, Any]],
        persona_result: PersonaDetectionResult = None
    ) -> AdaptiveDashboard:
        """Generate adaptive dashboard based on persona."""
        
        # Detect persona if not provided
        if persona_result is None:
            persona_result = persona_detector.detect_persona(datasets)
        
        persona = persona_result.persona
        
        # Route to appropriate generator
        if persona == UserPersona.BUSINESS:
            return self._generate_business_dashboard(datasets, persona_result)
        elif persona == UserPersona.ANALYTICS:
            return self._generate_analytics_dashboard(datasets, persona_result)
        elif persona == UserPersona.ML_ENGINEER:
            return self._generate_ml_dashboard(datasets, persona_result)
        elif persona == UserPersona.DATA_SCIENTIST:
            return self._generate_datascience_dashboard(datasets, persona_result)
        else:
            return self._generate_general_dashboard(datasets, persona_result)
    
    # =========================================================================
    # BUSINESS INTELLIGENCE DASHBOARD
    # =========================================================================
    def _generate_business_dashboard(
        self, 
        datasets: List[Dict], 
        persona_result: PersonaDetectionResult
    ) -> AdaptiveDashboard:
        """Generate Power BI-style business dashboard."""
        
        columns = self._collect_all_columns(datasets)
        
        # Find key business columns
        revenue_col = self._find_column(columns['numeric'], ['revenue', 'sales', 'amount', 'total', 'value', 'price'])
        quantity_col = self._find_column(columns['numeric'], ['quantity', 'qty', 'units', 'count', 'volume'])
        profit_col = self._find_column(columns['numeric'], ['profit', 'margin', 'net', 'gross'])
        cost_col = self._find_column(columns['numeric'], ['cost', 'expense', 'spend'])
        date_col = self._find_column(columns['datetime'], ['date', 'time', 'created', 'order_date', 'transaction'])
        category_col = self._find_column(columns['categorical'], ['category', 'product', 'type', 'segment', 'department'])
        region_col = self._find_column(columns['categorical'], ['region', 'branch', 'store', 'location', 'territory', 'city'])
        customer_col = self._find_column(columns['categorical'], ['customer', 'client', 'buyer', 'account'])
        
        tabs = []
        
        # TAB 1: Executive Overview
        overview_tab = DashboardTab(
            id="overview",
            title="Executive Overview",
            icon="chart-bar",
            description="Key business metrics at a glance",
            charts=[]
        )
        
        # KPI Cards
        if revenue_col:
            overview_tab.charts.append(ChartConfig(
                id="total_revenue", title="Total Revenue", type="kpi",
                y_column=revenue_col, aggregation="sum", size="small",
                config={"format": "currency", "icon": "dollar-sign"}
            ))
        
        if profit_col:
            overview_tab.charts.append(ChartConfig(
                id="total_profit", title="Total Profit", type="kpi",
                y_column=profit_col, aggregation="sum", size="small",
                config={"format": "currency", "icon": "trending-up"}
            ))
        elif revenue_col and cost_col:
            overview_tab.charts.append(ChartConfig(
                id="total_profit", title="Gross Profit", type="kpi",
                y_column=f"{revenue_col}-{cost_col}", aggregation="calculated", size="small",
                config={"format": "currency", "icon": "trending-up"}
            ))
        
        if quantity_col:
            overview_tab.charts.append(ChartConfig(
                id="total_units", title="Units Sold", type="kpi",
                y_column=quantity_col, aggregation="sum", size="small",
                config={"format": "number", "icon": "package"}
            ))
        
        # Transaction count
        overview_tab.charts.append(ChartConfig(
            id="total_transactions", title="Total Transactions", type="kpi",
            aggregation="count", size="small",
            config={"format": "number", "icon": "file-text"}
        ))
        
        # Revenue Trend
        if revenue_col and date_col:
            overview_tab.charts.append(ChartConfig(
                id="revenue_trend", title="Revenue Trend", type="line",
                x_column=date_col, y_column=revenue_col, aggregation="sum", size="large",
                description="Revenue over time"
            ))
        
        # Revenue by Category
        if revenue_col and category_col:
            overview_tab.charts.append(ChartConfig(
                id="revenue_by_category", title=f"Revenue by {category_col}", type="bar",
                x_column=category_col, y_column=revenue_col, aggregation="sum", size="medium"
            ))
        
        # Revenue by Region
        if revenue_col and region_col:
            overview_tab.charts.append(ChartConfig(
                id="revenue_by_region", title=f"Performance by {region_col}", type="bar",
                x_column=region_col, y_column=revenue_col, aggregation="sum", size="medium"
            ))
        
        tabs.append(overview_tab)
        
        # TAB 2: Sales Analysis
        if revenue_col or quantity_col:
            sales_tab = DashboardTab(
                id="sales",
                title="Sales Analysis",
                icon="shopping-cart",
                charts=[]
            )
            
            if category_col:
                sales_tab.charts.append(ChartConfig(
                    id="category_distribution", title="Sales Distribution", type="pie",
                    x_column=category_col, y_column=revenue_col or quantity_col,
                    aggregation="sum", size="medium"
                ))
            
            if date_col and revenue_col:
                sales_tab.charts.append(ChartConfig(
                    id="monthly_sales", title="Monthly Sales Comparison", type="bar",
                    x_column=date_col, y_column=revenue_col, aggregation="sum", size="large",
                    config={"groupBy": "month"}
                ))
            
            # Top products/categories
            if category_col and revenue_col:
                sales_tab.charts.append(ChartConfig(
                    id="top_performers", title="Top Performers", type="horizontal_bar",
                    x_column=category_col, y_column=revenue_col, aggregation="sum", size="medium",
                    config={"limit": 10, "sort": "desc"}
                ))
            
            tabs.append(sales_tab)
        
        # TAB 3: Customer Insights (if customer data)
        if customer_col or DataCategory.CUSTOMER_DATA in persona_result.data_categories:
            customer_tab = DashboardTab(
                id="customers",
                title="Customer Insights",
                icon="users",
                charts=[]
            )
            
            customer_tab.charts.append(ChartConfig(
                id="unique_customers", title="Unique Customers", type="kpi",
                x_column=customer_col, aggregation="count_distinct", size="small"
            ))
            
            if revenue_col and customer_col:
                customer_tab.charts.append(ChartConfig(
                    id="revenue_per_customer", title="Avg Revenue per Customer", type="kpi",
                    y_column=revenue_col, x_column=customer_col, aggregation="avg_per_group", size="small"
                ))
                
                customer_tab.charts.append(ChartConfig(
                    id="top_customers", title="Top Customers by Revenue", type="horizontal_bar",
                    x_column=customer_col, y_column=revenue_col, aggregation="sum", size="large",
                    config={"limit": 15, "sort": "desc"}
                ))
            
            tabs.append(customer_tab)
        
        # TAB 4: Regional Performance (if region data)
        if region_col:
            region_tab = DashboardTab(
                id="regions",
                title="Regional Performance",
                icon="map-pin",
                charts=[]
            )
            
            region_tab.charts.append(ChartConfig(
                id="regions_count", title="Total Regions/Branches", type="kpi",
                x_column=region_col, aggregation="count_distinct", size="small"
            ))
            
            if revenue_col:
                region_tab.charts.append(ChartConfig(
                    id="region_comparison", title="Regional Revenue Comparison", type="bar",
                    x_column=region_col, y_column=revenue_col, aggregation="sum", size="large"
                ))
                
                region_tab.charts.append(ChartConfig(
                    id="region_share", title="Regional Market Share", type="pie",
                    x_column=region_col, y_column=revenue_col, aggregation="sum", size="medium"
                ))
            
            tabs.append(region_tab)
        
        # TAB 5: Data Table
        tabs.append(DashboardTab(
            id="data",
            title="Data Explorer",
            icon="table",
            charts=[ChartConfig(
                id="data_table", title="Raw Data", type="table", size="full"
            )]
        ))
        
        # Build global filters
        filters = self._build_filters(columns, [category_col, region_col, date_col])
        
        return AdaptiveDashboard(
            dashboard_id="business_dashboard",
            title="Business Intelligence Dashboard",
            subtitle=persona_result.summary,
            persona=UserPersona.BUSINESS,
            dashboard_type="business_intelligence",
            tabs=tabs,
            global_filters=filters,
            color_scheme="corporate"
        )
    
    # =========================================================================
    # ANALYTICS / EDA DASHBOARD
    # =========================================================================
    def _generate_analytics_dashboard(
        self,
        datasets: List[Dict],
        persona_result: PersonaDetectionResult
    ) -> AdaptiveDashboard:
        """Generate exploratory data analysis dashboard."""
        
        columns = self._collect_all_columns(datasets)
        numeric_cols = columns['numeric'][:6]  # Limit for visualization
        categorical_cols = columns['categorical'][:4]
        
        tabs = []
        
        # TAB 1: Statistical Overview
        stats_tab = DashboardTab(
            id="statistics",
            title="Statistical Overview",
            icon="bar-chart-2",
            description="Summary statistics and distributions",
            charts=[]
        )
        
        # Row count
        stats_tab.charts.append(ChartConfig(
            id="total_records", title="Total Records", type="kpi",
            aggregation="count", size="small",
            config={"icon": "database"}
        ))
        
        # Column count
        stats_tab.charts.append(ChartConfig(
            id="total_columns", title="Total Features", type="kpi",
            aggregation="column_count", size="small",
            config={"icon": "columns"}
        ))
        
        # Missing values
        stats_tab.charts.append(ChartConfig(
            id="missing_pct", title="Data Completeness", type="kpi",
            aggregation="completeness", size="small",
            config={"icon": "check-circle", "format": "percent"}
        ))
        
        # Numeric columns histogram
        for col in numeric_cols[:3]:
            stats_tab.charts.append(ChartConfig(
                id=f"dist_{col}", title=f"Distribution: {col}", type="histogram",
                x_column=col, size="medium",
                description=f"Frequency distribution of {col}"
            ))
        
        tabs.append(stats_tab)
        
        # TAB 2: Distributions
        dist_tab = DashboardTab(
            id="distributions",
            title="Distributions",
            icon="activity",
            charts=[]
        )
        
        # Box plots for numeric
        for col in numeric_cols[:4]:
            dist_tab.charts.append(ChartConfig(
                id=f"box_{col}", title=f"Box Plot: {col}", type="boxplot",
                x_column=col, size="medium"
            ))
        
        # Categorical distributions
        for col in categorical_cols[:3]:
            dist_tab.charts.append(ChartConfig(
                id=f"cat_dist_{col}", title=f"Distribution: {col}", type="bar",
                x_column=col, aggregation="count", size="medium"
            ))
        
        tabs.append(dist_tab)
        
        # TAB 3: Correlations
        if len(numeric_cols) >= 2:
            corr_tab = DashboardTab(
                id="correlations",
                title="Correlations",
                icon="git-merge",
                charts=[]
            )
            
            corr_tab.charts.append(ChartConfig(
                id="correlation_matrix", title="Correlation Matrix", type="heatmap",
                size="large",
                config={"columns": numeric_cols}
            ))
            
            # Scatter plots for top correlations
            if len(numeric_cols) >= 2:
                corr_tab.charts.append(ChartConfig(
                    id="scatter_1", title=f"{numeric_cols[0]} vs {numeric_cols[1]}", type="scatter",
                    x_column=numeric_cols[0], y_column=numeric_cols[1], size="medium"
                ))
            
            if len(numeric_cols) >= 4:
                corr_tab.charts.append(ChartConfig(
                    id="scatter_2", title=f"{numeric_cols[2]} vs {numeric_cols[3]}", type="scatter",
                    x_column=numeric_cols[2], y_column=numeric_cols[3], size="medium"
                ))
            
            tabs.append(corr_tab)
        
        # TAB 4: Time Series (if date column)
        date_col = self._find_column(columns['datetime'], ['date', 'time', 'timestamp'])
        if date_col:
            time_tab = DashboardTab(
                id="time_series",
                title="Time Series",
                icon="clock",
                charts=[]
            )
            
            for col in numeric_cols[:3]:
                time_tab.charts.append(ChartConfig(
                    id=f"trend_{col}", title=f"{col} Over Time", type="line",
                    x_column=date_col, y_column=col, aggregation="avg", size="large"
                ))
            
            tabs.append(time_tab)
        
        # TAB 5: Data Quality
        quality_tab = DashboardTab(
            id="quality",
            title="Data Quality",
            icon="shield",
            charts=[]
        )
        
        quality_tab.charts.append(ChartConfig(
            id="missing_by_column", title="Missing Values by Column", type="bar",
            aggregation="missing_count", size="large"
        ))
        
        quality_tab.charts.append(ChartConfig(
            id="data_types", title="Data Type Distribution", type="pie",
            aggregation="dtype_count", size="medium"
        ))
        
        tabs.append(quality_tab)
        
        # TAB 6: Data Table
        tabs.append(DashboardTab(
            id="data",
            title="Data Explorer",
            icon="table",
            charts=[ChartConfig(
                id="data_table", title="Raw Data", type="table", size="full"
            )]
        ))
        
        filters = self._build_filters(columns, categorical_cols[:3])
        
        return AdaptiveDashboard(
            dashboard_id="analytics_dashboard",
            title="Exploratory Data Analysis",
            subtitle=persona_result.summary,
            persona=UserPersona.ANALYTICS,
            dashboard_type="exploratory_analytics",
            tabs=tabs,
            global_filters=filters,
            color_scheme="scientific"
        )
    
    # =========================================================================
    # ML MODEL DASHBOARD
    # =========================================================================
    def _generate_ml_dashboard(
        self,
        datasets: List[Dict],
        persona_result: PersonaDetectionResult
    ) -> AdaptiveDashboard:
        """Generate ML model performance dashboard."""
        
        columns = self._collect_all_columns(datasets)
        has_model_file = DataCategory.MODEL_FILE in persona_result.data_categories
        
        # Try to find prediction/label columns
        pred_col = self._find_column(columns['all'], ['prediction', 'predicted', 'pred', 'y_pred', 'output'])
        actual_col = self._find_column(columns['all'], ['actual', 'true', 'label', 'y_true', 'target', 'ground_truth'])
        prob_col = self._find_column(columns['numeric'], ['probability', 'prob', 'confidence', 'score'])
        feature_cols = [c for c in columns['numeric'] if c not in [pred_col, actual_col, prob_col]]
        
        tabs = []
        
        # TAB 1: Model Overview
        overview_tab = DashboardTab(
            id="overview",
            title="Model Overview",
            icon="cpu",
            description="Model performance summary",
            charts=[]
        )
        
        if has_model_file:
            overview_tab.charts.append(ChartConfig(
                id="model_type", title="Model Type", type="kpi",
                aggregation="model_info", size="small",
                config={"icon": "box"}
            ))
        
        overview_tab.charts.append(ChartConfig(
            id="sample_count", title="Total Samples", type="kpi",
            aggregation="count", size="small",
            config={"icon": "database"}
        ))
        
        if pred_col and actual_col:
            overview_tab.charts.append(ChartConfig(
                id="accuracy", title="Accuracy", type="gauge",
                x_column=pred_col, y_column=actual_col, aggregation="accuracy", size="medium",
                config={"max": 100, "thresholds": [60, 80, 90]}
            ))
            
            overview_tab.charts.append(ChartConfig(
                id="confusion_matrix", title="Confusion Matrix", type="heatmap",
                x_column=pred_col, y_column=actual_col, aggregation="confusion", size="large"
            ))
        
        if pred_col:
            overview_tab.charts.append(ChartConfig(
                id="prediction_dist", title="Prediction Distribution", type="bar",
                x_column=pred_col, aggregation="count", size="medium"
            ))
        
        tabs.append(overview_tab)
        
        # TAB 2: Performance Metrics
        if pred_col and actual_col:
            metrics_tab = DashboardTab(
                id="metrics",
                title="Performance Metrics",
                icon="bar-chart",
                charts=[]
            )
            
            metrics_tab.charts.append(ChartConfig(
                id="precision", title="Precision", type="kpi",
                x_column=pred_col, y_column=actual_col, aggregation="precision", size="small"
            ))
            
            metrics_tab.charts.append(ChartConfig(
                id="recall", title="Recall", type="kpi",
                x_column=pred_col, y_column=actual_col, aggregation="recall", size="small"
            ))
            
            metrics_tab.charts.append(ChartConfig(
                id="f1_score", title="F1 Score", type="kpi",
                x_column=pred_col, y_column=actual_col, aggregation="f1", size="small"
            ))
            
            if prob_col:
                metrics_tab.charts.append(ChartConfig(
                    id="roc_curve", title="ROC Curve", type="roc",
                    x_column=prob_col, y_column=actual_col, size="large"
                ))
                
                metrics_tab.charts.append(ChartConfig(
                    id="confidence_dist", title="Confidence Distribution", type="histogram",
                    x_column=prob_col, size="medium"
                ))
            
            tabs.append(metrics_tab)
        
        # TAB 3: Feature Analysis
        if feature_cols:
            feature_tab = DashboardTab(
                id="features",
                title="Feature Analysis",
                icon="layers",
                charts=[]
            )
            
            feature_tab.charts.append(ChartConfig(
                id="feature_importance", title="Feature Importance", type="horizontal_bar",
                aggregation="feature_importance", size="large",
                config={"columns": feature_cols[:10]}
            ))
            
            # Feature distributions
            for col in feature_cols[:4]:
                feature_tab.charts.append(ChartConfig(
                    id=f"feature_{col}", title=f"Feature: {col}", type="histogram",
                    x_column=col, size="medium"
                ))
            
            tabs.append(feature_tab)
        
        # TAB 4: Error Analysis
        if pred_col and actual_col:
            error_tab = DashboardTab(
                id="errors",
                title="Error Analysis",
                icon="alert-circle",
                charts=[]
            )
            
            error_tab.charts.append(ChartConfig(
                id="error_rate", title="Error Rate", type="kpi",
                x_column=pred_col, y_column=actual_col, aggregation="error_rate", size="small"
            ))
            
            error_tab.charts.append(ChartConfig(
                id="misclassified", title="Misclassified Samples", type="table",
                x_column=pred_col, y_column=actual_col, aggregation="errors_only", size="full",
                config={"filter": "errors"}
            ))
            
            tabs.append(error_tab)
        
        # TAB 5: Predictions Table
        tabs.append(DashboardTab(
            id="predictions",
            title="Predictions",
            icon="table",
            charts=[ChartConfig(
                id="pred_table", title="All Predictions", type="table", size="full"
            )]
        ))
        
        return AdaptiveDashboard(
            dashboard_id="ml_dashboard",
            title="Model Performance Dashboard",
            subtitle=persona_result.summary,
            persona=UserPersona.ML_ENGINEER,
            dashboard_type="ml_performance",
            tabs=tabs,
            global_filters=[],
            color_scheme="technical"
        )
    
    # =========================================================================
    # DATA SCIENCE DASHBOARD (Mix of Analytics + ML)
    # =========================================================================
    def _generate_datascience_dashboard(
        self,
        datasets: List[Dict],
        persona_result: PersonaDetectionResult
    ) -> AdaptiveDashboard:
        """Generate data science workflow dashboard."""
        
        # Combine analytics and ML dashboards
        analytics_dash = self._generate_analytics_dashboard(datasets, persona_result)
        
        # Add ML-specific tabs if predictions exist
        columns = self._collect_all_columns(datasets)
        pred_col = self._find_column(columns['all'], ['prediction', 'predicted', 'pred', 'target', 'label'])
        
        if pred_col:
            ml_tab = DashboardTab(
                id="modeling",
                title="Modeling",
                icon="cpu",
                charts=[
                    ChartConfig(
                        id="target_dist", title="Target Distribution", type="bar",
                        x_column=pred_col, aggregation="count", size="medium"
                    ),
                    ChartConfig(
                        id="target_balance", title="Class Balance", type="pie",
                        x_column=pred_col, aggregation="count", size="medium"
                    )
                ]
            )
            analytics_dash.tabs.insert(2, ml_tab)
        
        analytics_dash.dashboard_id = "datascience_dashboard"
        analytics_dash.title = "Data Science Workbench"
        analytics_dash.dashboard_type = "data_science"
        analytics_dash.persona = UserPersona.DATA_SCIENTIST
        
        return analytics_dash
    
    # =========================================================================
    # GENERAL DASHBOARD
    # =========================================================================
    def _generate_general_dashboard(
        self,
        datasets: List[Dict],
        persona_result: PersonaDetectionResult
    ) -> AdaptiveDashboard:
        """Generate general-purpose dashboard."""
        
        columns = self._collect_all_columns(datasets)
        
        tabs = [
            DashboardTab(
                id="overview",
                title="Overview",
                icon="eye",
                charts=[
                    ChartConfig(id="total_rows", title="Total Records", type="kpi", 
                               aggregation="count", size="small"),
                    ChartConfig(id="total_cols", title="Total Columns", type="kpi",
                               aggregation="column_count", size="small"),
                ]
            ),
            DashboardTab(
                id="data",
                title="Data Explorer",
                icon="table",
                charts=[ChartConfig(id="data_table", title="Data", type="table", size="full")]
            )
        ]
        
        # Add charts for available columns
        if columns['numeric']:
            tabs[0].charts.append(ChartConfig(
                id="num_dist", title=f"Distribution: {columns['numeric'][0]}",
                type="histogram", x_column=columns['numeric'][0], size="medium"
            ))
        
        if columns['categorical']:
            tabs[0].charts.append(ChartConfig(
                id="cat_dist", title=f"Distribution: {columns['categorical'][0]}",
                type="bar", x_column=columns['categorical'][0], aggregation="count", size="medium"
            ))
        
        return AdaptiveDashboard(
            dashboard_id="general_dashboard",
            title="Data Dashboard",
            subtitle=persona_result.summary,
            persona=UserPersona.UNKNOWN,
            dashboard_type="general",
            tabs=tabs,
            global_filters=[],
            color_scheme="default"
        )
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    def _collect_all_columns(self, datasets: List[Dict]) -> Dict[str, List[str]]:
        """Collect and categorize columns from all datasets."""
        columns = {
            'numeric': [],
            'categorical': [],
            'datetime': [],
            'text': [],
            'all': []
        }
        
        for ds in datasets:
            df = ds.get('df')
            if df is None:
                continue
            
            for col in df.columns:
                col_lower = col.lower()
                dtype = str(df[col].dtype)
                
                columns['all'].append(col)
                
                if 'datetime' in dtype or any(x in col_lower for x in ['date', 'time', 'timestamp']):
                    columns['datetime'].append(col)
                elif dtype in ['int64', 'float64', 'int32', 'float32']:
                    columns['numeric'].append(col)
                elif df[col].dtype == 'object':
                    if df[col].nunique() < 50:
                        columns['categorical'].append(col)
                    else:
                        columns['text'].append(col)
        
        # Remove duplicates while preserving order
        for key in columns:
            columns[key] = list(dict.fromkeys(columns[key]))
        
        return columns
    
    def _find_column(self, columns: List[str], keywords: List[str]) -> Optional[str]:
        """Find a column matching any keyword."""
        for col in columns:
            col_lower = col.lower()
            for kw in keywords:
                if kw in col_lower:
                    return col
        return columns[0] if columns else None
    
    def _build_filters(self, columns: Dict, filter_cols: List[str]) -> List[Dict]:
        """Build global filter configuration."""
        filters = []
        
        for col in filter_cols:
            if col and col in columns.get('categorical', []):
                filters.append({
                    'column': col,
                    'type': 'multi_select',
                    'label': col.replace('_', ' ').title()
                })
            elif col and col in columns.get('datetime', []):
                filters.append({
                    'column': col,
                    'type': 'date_range',
                    'label': col.replace('_', ' ').title()
                })
        
        return filters[:4]  # Limit to 4 filters
    
    def to_dict(self, dashboard: AdaptiveDashboard) -> Dict[str, Any]:
        """Convert dashboard to JSON-serializable dict."""
        return {
            'dashboard_id': dashboard.dashboard_id,
            'title': dashboard.title,
            'subtitle': dashboard.subtitle,
            'persona': dashboard.persona.value,
            'dashboard_type': dashboard.dashboard_type,
            'color_scheme': dashboard.color_scheme,
            'global_filters': dashboard.global_filters,
            'kpi_summary': dashboard.kpi_summary,
            'ai_insights': dashboard.ai_insights,
            'tabs': [
                {
                    'id': tab.id,
                    'title': tab.title,
                    'icon': tab.icon,
                    'description': tab.description,
                    'charts': [
                        {
                            'id': chart.id,
                            'title': chart.title,
                            'type': chart.type,
                            'size': chart.size,
                            'x_column': chart.x_column,
                            'y_column': chart.y_column,
                            'color_column': chart.color_column,
                            'aggregation': chart.aggregation,
                            'description': chart.description,
                            'config': chart.config
                        }
                        for chart in tab.charts
                    ]
                }
                for tab in dashboard.tabs
            ]
        }


# Singleton
adaptive_dashboard_generator = AdaptiveDashboardGenerator()

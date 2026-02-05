# Dashboard Factory - Generates Persona-Specific Dashboards
# Like Power BI / Julius AI - Automatic Smart Charts Based on Content Type

from typing import Dict, List, Any, Optional
from enum import Enum
import pandas as pd
import numpy as np
from dataclasses import dataclass, field


class DashboardType(Enum):
    """Dashboard templates based on detected content."""
    SALES_PERFORMANCE = "sales_performance"
    PRODUCT_ANALYTICS = "product_analytics"
    CUSTOMER_360 = "customer_360"
    FINANCIAL_OVERVIEW = "financial_overview"
    HR_WORKFORCE = "hr_workforce"
    ML_MODEL_METRICS = "ml_model_metrics"
    LEGAL_CASE_TRACKER = "legal_case_tracker"
    HEALTHCARE_CLINICAL = "healthcare_clinical"
    SUPPLY_CHAIN = "supply_chain"
    MARKETING_CAMPAIGN = "marketing_campaign"
    GENERIC_EXPLORER = "generic_explorer"


@dataclass
class ChartSpec:
    """Specification for a single chart."""
    chart_id: str
    title: str
    chart_type: str  # bar, line, pie, scatter, table, kpi, heatmap, treemap
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    group_by: Optional[str] = None
    aggregation: str = "sum"  # sum, count, avg, min, max
    filters: Dict[str, Any] = field(default_factory=dict)
    size: str = "medium"  # small, medium, large, full
    priority: int = 1  # Lower = higher priority
    description: str = ""


@dataclass
class DashboardSheet:
    """A single sheet/page in the dashboard."""
    sheet_id: str
    title: str
    icon: str
    charts: List[ChartSpec] = field(default_factory=list)
    layout: str = "grid"  # grid, flow, tabs


@dataclass
class DashboardConfig:
    """Full dashboard configuration."""
    dashboard_id: str
    title: str
    dashboard_type: DashboardType
    persona: str
    sheets: List[DashboardSheet] = field(default_factory=list)
    global_filters: List[Dict[str, Any]] = field(default_factory=list)
    color_scheme: str = "modern"
    created_for_files: List[str] = field(default_factory=list)


class DashboardFactory:
    """Factory that generates appropriate dashboards based on content classification."""
    
    def __init__(self):
        pass  # No pre-initialization needed
    
    def generate_dashboard(
        self, 
        classifications: List[Dict[str, Any]], 
        datasets: List[Dict[str, Any]]
    ) -> DashboardConfig:
        """
        Generate a dashboard based on detected content types.
        
        Args:
            classifications: List of classification results from SmartContentClassifier
            datasets: List of dataset metadata with column info
        
        Returns:
            DashboardConfig with sheets and charts
        """
        # Determine dominant content type
        primary_type = self._determine_primary_type(classifications)
        dashboard_type = self._map_to_dashboard_type(primary_type)
        
        # Build dashboard
        config = DashboardConfig(
            dashboard_id=f"dash_{primary_type}",
            title=self._get_dashboard_title(dashboard_type),
            dashboard_type=dashboard_type,
            persona=self._get_persona(dashboard_type),
            created_for_files=[c.get('filename', '') for c in classifications]
        )
        
        # Generate sheets based on type
        config.sheets = self._generate_sheets(dashboard_type, datasets, classifications)
        
        # Add global filters
        config.global_filters = self._generate_global_filters(datasets)
        
        return config
    
    def _determine_primary_type(self, classifications: List[Dict]) -> str:
        """Find the dominant content type across all files."""
        type_scores = {}
        
        for clf in classifications:
            primary = clf.get('primary_role', 'Unknown')
            confidence = clf.get('confidence', 0.5)
            
            if primary not in type_scores:
                type_scores[primary] = 0
            type_scores[primary] += confidence
        
        if not type_scores:
            return 'generic'
        
        return max(type_scores, key=type_scores.get)
    
    def _map_to_dashboard_type(self, content_type: str) -> DashboardType:
        """Map content type to dashboard template."""
        mapping = {
            'Sales Data': DashboardType.SALES_PERFORMANCE,
            'Product Catalog': DashboardType.PRODUCT_ANALYTICS,
            'Customer Data': DashboardType.CUSTOMER_360,
            'Subscription Data': DashboardType.CUSTOMER_360,
            'Financial/Accounting': DashboardType.FINANCIAL_OVERVIEW,
            'Transaction Data': DashboardType.FINANCIAL_OVERVIEW,
            'HR/Employee Data': DashboardType.HR_WORKFORCE,
            'ML Training Data': DashboardType.ML_MODEL_METRICS,
            'Model Predictions': DashboardType.ML_MODEL_METRICS,
            'Legal/Contract Data': DashboardType.LEGAL_CASE_TRACKER,
            'Healthcare/Clinical': DashboardType.HEALTHCARE_CLINICAL,
            'Inventory Data': DashboardType.SUPPLY_CHAIN,
            'Marketing/Campaign': DashboardType.MARKETING_CAMPAIGN,
        }
        return mapping.get(content_type, DashboardType.GENERIC_EXPLORER)
    
    def _get_dashboard_title(self, dtype: DashboardType) -> str:
        """Get human-readable dashboard title."""
        titles = {
            DashboardType.SALES_PERFORMANCE: "Sales Performance Dashboard",
            DashboardType.PRODUCT_ANALYTICS: "Product Analytics Dashboard",
            DashboardType.CUSTOMER_360: "Customer 360 Dashboard",
            DashboardType.FINANCIAL_OVERVIEW: "Financial Overview Dashboard",
            DashboardType.HR_WORKFORCE: "Workforce Analytics Dashboard",
            DashboardType.ML_MODEL_METRICS: "Model Performance Dashboard",
            DashboardType.LEGAL_CASE_TRACKER: "Legal & Compliance Dashboard",
            DashboardType.HEALTHCARE_CLINICAL: "Clinical Analytics Dashboard",
            DashboardType.SUPPLY_CHAIN: "Supply Chain Dashboard",
            DashboardType.MARKETING_CAMPAIGN: "Marketing Performance Dashboard",
            DashboardType.GENERIC_EXPLORER: "Data Explorer Dashboard",
        }
        return titles.get(dtype, "Analytics Dashboard")
    
    def _get_persona(self, dtype: DashboardType) -> str:
        """Get persona for the dashboard type."""
        personas = {
            DashboardType.SALES_PERFORMANCE: "Business Analyst",
            DashboardType.PRODUCT_ANALYTICS: "Product Manager",
            DashboardType.CUSTOMER_360: "Customer Success",
            DashboardType.FINANCIAL_OVERVIEW: "Financial Analyst",
            DashboardType.HR_WORKFORCE: "HR Analyst",
            DashboardType.ML_MODEL_METRICS: "Data Scientist",
            DashboardType.LEGAL_CASE_TRACKER: "Legal Analyst",
            DashboardType.HEALTHCARE_CLINICAL: "Clinical Analyst",
            DashboardType.SUPPLY_CHAIN: "Operations Analyst",
            DashboardType.MARKETING_CAMPAIGN: "Marketing Analyst",
            DashboardType.GENERIC_EXPLORER: "Data Analyst",
        }
        return personas.get(dtype, "Analyst")
    
    def _generate_sheets(
        self, 
        dtype: DashboardType, 
        datasets: List[Dict],
        classifications: List[Dict]
    ) -> List[DashboardSheet]:
        """Generate sheets for the dashboard type."""
        
        # Collect all columns from datasets
        all_columns = self._collect_columns(datasets)
        
        if dtype == DashboardType.SALES_PERFORMANCE:
            return self._sales_sheets(all_columns, datasets)
        elif dtype == DashboardType.PRODUCT_ANALYTICS:
            return self._product_sheets(all_columns, datasets)
        elif dtype == DashboardType.CUSTOMER_360:
            return self._customer_sheets(all_columns, datasets)
        elif dtype == DashboardType.FINANCIAL_OVERVIEW:
            return self._financial_sheets(all_columns, datasets)
        elif dtype == DashboardType.HR_WORKFORCE:
            return self._hr_sheets(all_columns, datasets)
        elif dtype == DashboardType.ML_MODEL_METRICS:
            return self._ml_sheets(all_columns, datasets)
        elif dtype == DashboardType.LEGAL_CASE_TRACKER:
            return self._legal_sheets(all_columns, datasets)
        else:
            return self._generic_sheets(all_columns, datasets)
    
    def _collect_columns(self, datasets: List[Dict]) -> Dict[str, List[str]]:
        """Collect columns by type from all datasets."""
        columns = {
            'numeric': [],
            'categorical': [],
            'datetime': [],
            'text': [],
            'id': []
        }
        
        for ds in datasets:
            if 'df' not in ds:
                continue
            df = ds['df']
            if df is None:
                continue
                
            for col in df.columns:
                col_lower = col.lower()
                dtype = str(df[col].dtype)
                
                # Detect ID columns
                if any(x in col_lower for x in ['id', '_id', 'uuid', 'key']):
                    columns['id'].append(col)
                elif 'datetime' in dtype or 'date' in col_lower:
                    columns['datetime'].append(col)
                elif dtype in ['int64', 'float64', 'int32', 'float32']:
                    columns['numeric'].append(col)
                elif df[col].dtype == 'object':
                    if df[col].nunique() < 50:
                        columns['categorical'].append(col)
                    else:
                        columns['text'].append(col)
        
        return columns
    
    # ============== SALES DASHBOARD ==============
    def _sales_sheets(self, columns: Dict, datasets: List[Dict]) -> List[DashboardSheet]:
        """Generate sales performance dashboard sheets."""
        sheets = []
        
        # Overview Sheet
        overview = DashboardSheet(
            sheet_id="overview",
            title="Overview",
            icon="chart-bar",
            charts=[]
        )
        
        # Find revenue/sales columns
        numeric_cols = columns.get('numeric', [])
        revenue_col = self._find_column(numeric_cols, ['revenue', 'sales', 'amount', 'total', 'price'])
        date_col = self._find_column(columns.get('datetime', []), ['date', 'time', 'created', 'order'])
        category_col = self._find_column(columns.get('categorical', []), ['category', 'product', 'type', 'region', 'segment'])
        
        # KPI Cards
        if revenue_col:
            overview.charts.append(ChartSpec(
                chart_id="total_revenue",
                title="Total Revenue",
                chart_type="kpi",
                y_column=revenue_col,
                aggregation="sum",
                size="small",
                priority=1
            ))
            overview.charts.append(ChartSpec(
                chart_id="avg_order",
                title="Avg Order Value",
                chart_type="kpi",
                y_column=revenue_col,
                aggregation="avg",
                size="small",
                priority=2
            ))
        
        # Add count KPI
        overview.charts.append(ChartSpec(
            chart_id="total_records",
            title="Total Records",
            chart_type="kpi",
            aggregation="count",
            size="small",
            priority=3
        ))
        
        # Revenue over time
        if revenue_col and date_col:
            overview.charts.append(ChartSpec(
                chart_id="revenue_trend",
                title="Revenue Trend",
                chart_type="line",
                x_column=date_col,
                y_column=revenue_col,
                aggregation="sum",
                size="large",
                priority=4
            ))
        
        # Revenue by category
        if revenue_col and category_col:
            overview.charts.append(ChartSpec(
                chart_id="revenue_by_category",
                title=f"Revenue by {category_col}",
                chart_type="bar",
                x_column=category_col,
                y_column=revenue_col,
                aggregation="sum",
                size="medium",
                priority=5
            ))
            overview.charts.append(ChartSpec(
                chart_id="category_distribution",
                title=f"{category_col} Distribution",
                chart_type="pie",
                x_column=category_col,
                aggregation="count",
                size="medium",
                priority=6
            ))
        
        sheets.append(overview)
        
        # Detailed Analysis Sheet
        detail = DashboardSheet(
            sheet_id="details",
            title="Detailed Analysis",
            icon="table",
            charts=[]
        )
        
        # Data table
        detail.charts.append(ChartSpec(
            chart_id="data_table",
            title="Raw Data",
            chart_type="table",
            size="full",
            priority=1
        ))
        
        sheets.append(detail)
        
        return sheets
    
    # ============== PRODUCT DASHBOARD ==============
    def _product_sheets(self, columns: Dict, datasets: List[Dict]) -> List[DashboardSheet]:
        """Generate product analytics dashboard sheets."""
        sheets = []
        
        overview = DashboardSheet(
            sheet_id="catalog",
            title="Product Catalog",
            icon="box",
            charts=[]
        )
        
        price_col = self._find_column(columns.get('numeric', []), ['price', 'cost', 'msrp', 'value'])
        quantity_col = self._find_column(columns.get('numeric', []), ['quantity', 'stock', 'inventory', 'count'])
        category_col = self._find_column(columns.get('categorical', []), ['category', 'type', 'brand', 'department'])
        
        if price_col:
            overview.charts.append(ChartSpec(
                chart_id="avg_price",
                title="Avg Price",
                chart_type="kpi",
                y_column=price_col,
                aggregation="avg",
                size="small"
            ))
        
        overview.charts.append(ChartSpec(
            chart_id="product_count",
            title="Total Products",
            chart_type="kpi",
            aggregation="count",
            size="small"
        ))
        
        if price_col and category_col:
            overview.charts.append(ChartSpec(
                chart_id="price_by_category",
                title="Price Distribution by Category",
                chart_type="bar",
                x_column=category_col,
                y_column=price_col,
                aggregation="avg",
                size="large"
            ))
        
        sheets.append(overview)
        return sheets
    
    # ============== CUSTOMER DASHBOARD ==============
    def _customer_sheets(self, columns: Dict, datasets: List[Dict]) -> List[DashboardSheet]:
        """Generate customer 360 dashboard sheets."""
        sheets = []
        
        overview = DashboardSheet(
            sheet_id="customers",
            title="Customer Overview",
            icon="users",
            charts=[]
        )
        
        # KPIs
        overview.charts.append(ChartSpec(
            chart_id="total_customers",
            title="Total Customers",
            chart_type="kpi",
            aggregation="count",
            size="small"
        ))
        
        segment_col = self._find_column(columns.get('categorical', []), ['segment', 'tier', 'type', 'plan', 'status'])
        value_col = self._find_column(columns.get('numeric', []), ['value', 'revenue', 'spend', 'amount', 'ltv'])
        
        if segment_col:
            overview.charts.append(ChartSpec(
                chart_id="customer_segments",
                title="Customer Segments",
                chart_type="pie",
                x_column=segment_col,
                aggregation="count",
                size="medium"
            ))
        
        if value_col and segment_col:
            overview.charts.append(ChartSpec(
                chart_id="value_by_segment",
                title="Value by Segment",
                chart_type="bar",
                x_column=segment_col,
                y_column=value_col,
                aggregation="sum",
                size="medium"
            ))
        
        sheets.append(overview)
        return sheets
    
    # ============== FINANCIAL DASHBOARD ==============
    def _financial_sheets(self, columns: Dict, datasets: List[Dict]) -> List[DashboardSheet]:
        """Generate financial overview dashboard sheets."""
        sheets = []
        
        overview = DashboardSheet(
            sheet_id="financials",
            title="Financial Overview",
            icon="dollar-sign",
            charts=[]
        )
        
        amount_col = self._find_column(columns.get('numeric', []), ['amount', 'total', 'value', 'balance', 'revenue', 'cost'])
        date_col = self._find_column(columns.get('datetime', []), ['date', 'time', 'period'])
        type_col = self._find_column(columns.get('categorical', []), ['type', 'category', 'account', 'department'])
        
        if amount_col:
            overview.charts.append(ChartSpec(
                chart_id="total_amount",
                title="Total Amount",
                chart_type="kpi",
                y_column=amount_col,
                aggregation="sum",
                size="small"
            ))
        
        overview.charts.append(ChartSpec(
            chart_id="transaction_count",
            title="Transactions",
            chart_type="kpi",
            aggregation="count",
            size="small"
        ))
        
        if amount_col and date_col:
            overview.charts.append(ChartSpec(
                chart_id="amount_trend",
                title="Financial Trend",
                chart_type="line",
                x_column=date_col,
                y_column=amount_col,
                aggregation="sum",
                size="large"
            ))
        
        if amount_col and type_col:
            overview.charts.append(ChartSpec(
                chart_id="amount_by_type",
                title=f"Amount by {type_col}",
                chart_type="bar",
                x_column=type_col,
                y_column=amount_col,
                aggregation="sum",
                size="medium"
            ))
        
        sheets.append(overview)
        return sheets
    
    # ============== HR DASHBOARD ==============
    def _hr_sheets(self, columns: Dict, datasets: List[Dict]) -> List[DashboardSheet]:
        """Generate HR/workforce dashboard sheets."""
        sheets = []
        
        overview = DashboardSheet(
            sheet_id="workforce",
            title="Workforce Overview",
            icon="briefcase",
            charts=[]
        )
        
        salary_col = self._find_column(columns.get('numeric', []), ['salary', 'wage', 'compensation', 'pay'])
        dept_col = self._find_column(columns.get('categorical', []), ['department', 'team', 'division', 'unit'])
        role_col = self._find_column(columns.get('categorical', []), ['role', 'title', 'position', 'job'])
        
        overview.charts.append(ChartSpec(
            chart_id="headcount",
            title="Total Headcount",
            chart_type="kpi",
            aggregation="count",
            size="small"
        ))
        
        if salary_col:
            overview.charts.append(ChartSpec(
                chart_id="avg_salary",
                title="Avg Salary",
                chart_type="kpi",
                y_column=salary_col,
                aggregation="avg",
                size="small"
            ))
        
        if dept_col:
            overview.charts.append(ChartSpec(
                chart_id="by_department",
                title="By Department",
                chart_type="bar",
                x_column=dept_col,
                aggregation="count",
                size="medium"
            ))
        
        sheets.append(overview)
        return sheets
    
    # ============== ML DASHBOARD ==============
    def _ml_sheets(self, columns: Dict, datasets: List[Dict]) -> List[DashboardSheet]:
        """Generate ML model metrics dashboard sheets."""
        sheets = []
        
        overview = DashboardSheet(
            sheet_id="model_metrics",
            title="Model Metrics",
            icon="cpu",
            charts=[]
        )
        
        # Look for prediction/label columns
        pred_col = self._find_column(columns.get('numeric', []) + columns.get('categorical', []), 
                                     ['prediction', 'predicted', 'pred', 'output', 'score'])
        label_col = self._find_column(columns.get('numeric', []) + columns.get('categorical', []), 
                                      ['label', 'target', 'actual', 'true', 'class'])
        prob_col = self._find_column(columns.get('numeric', []), ['probability', 'prob', 'confidence', 'score'])
        
        overview.charts.append(ChartSpec(
            chart_id="sample_count",
            title="Total Samples",
            chart_type="kpi",
            aggregation="count",
            size="small"
        ))
        
        if pred_col:
            overview.charts.append(ChartSpec(
                chart_id="prediction_dist",
                title="Prediction Distribution",
                chart_type="bar",
                x_column=pred_col,
                aggregation="count",
                size="medium"
            ))
        
        if label_col:
            overview.charts.append(ChartSpec(
                chart_id="label_dist",
                title="Label Distribution",
                chart_type="pie",
                x_column=label_col,
                aggregation="count",
                size="medium"
            ))
        
        if prob_col:
            overview.charts.append(ChartSpec(
                chart_id="confidence_hist",
                title="Confidence Distribution",
                chart_type="histogram",
                x_column=prob_col,
                size="medium"
            ))
        
        sheets.append(overview)
        return sheets
    
    # ============== LEGAL DASHBOARD ==============
    def _legal_sheets(self, columns: Dict, datasets: List[Dict]) -> List[DashboardSheet]:
        """Generate legal/case tracker dashboard sheets."""
        sheets = []
        
        overview = DashboardSheet(
            sheet_id="cases",
            title="Case Overview",
            icon="file-text",
            charts=[]
        )
        
        status_col = self._find_column(columns.get('categorical', []), ['status', 'state', 'stage', 'outcome'])
        type_col = self._find_column(columns.get('categorical', []), ['type', 'category', 'matter', 'case_type'])
        date_col = self._find_column(columns.get('datetime', []), ['date', 'filed', 'opened', 'created'])
        
        overview.charts.append(ChartSpec(
            chart_id="total_cases",
            title="Total Cases",
            chart_type="kpi",
            aggregation="count",
            size="small"
        ))
        
        if status_col:
            overview.charts.append(ChartSpec(
                chart_id="by_status",
                title="Cases by Status",
                chart_type="pie",
                x_column=status_col,
                aggregation="count",
                size="medium"
            ))
        
        if type_col:
            overview.charts.append(ChartSpec(
                chart_id="by_type",
                title="Cases by Type",
                chart_type="bar",
                x_column=type_col,
                aggregation="count",
                size="medium"
            ))
        
        sheets.append(overview)
        return sheets
    
    # ============== GENERIC DASHBOARD ==============
    def _generic_sheets(self, columns: Dict, datasets: List[Dict]) -> List[DashboardSheet]:
        """Generate generic data explorer dashboard."""
        sheets = []
        
        overview = DashboardSheet(
            sheet_id="explorer",
            title="Data Explorer",
            icon="search",
            charts=[]
        )
        
        # KPI: Record count
        overview.charts.append(ChartSpec(
            chart_id="total_records",
            title="Total Records",
            chart_type="kpi",
            aggregation="count",
            size="small"
        ))
        
        # Bar chart for first categorical
        if columns.get('categorical'):
            cat_col = columns['categorical'][0]
            overview.charts.append(ChartSpec(
                chart_id="cat_dist",
                title=f"Distribution of {cat_col}",
                chart_type="bar",
                x_column=cat_col,
                aggregation="count",
                size="medium"
            ))
        
        # Line chart if datetime exists
        if columns.get('datetime') and columns.get('numeric'):
            overview.charts.append(ChartSpec(
                chart_id="time_trend",
                title="Trend Over Time",
                chart_type="line",
                x_column=columns['datetime'][0],
                y_column=columns['numeric'][0],
                aggregation="sum",
                size="large"
            ))
        
        # Data table
        overview.charts.append(ChartSpec(
            chart_id="data_table",
            title="Data Preview",
            chart_type="table",
            size="full"
        ))
        
        sheets.append(overview)
        return sheets
    
    def _find_column(self, columns: List[str], keywords: List[str]) -> Optional[str]:
        """Find a column matching any of the keywords."""
        for col in columns:
            col_lower = col.lower()
            for kw in keywords:
                if kw in col_lower:
                    return col
        return columns[0] if columns else None
    
    def _generate_global_filters(self, datasets: List[Dict]) -> List[Dict]:
        """Generate global filter options."""
        filters = []
        
        # Collect all date and categorical columns
        for ds in datasets:
            if 'df' not in ds or ds['df'] is None:
                continue
            df = ds['df']
            
            for col in df.columns:
                col_lower = col.lower()
                
                # Date filters
                if 'date' in col_lower or 'time' in col_lower:
                    filters.append({
                        'column': col,
                        'type': 'date_range',
                        'label': f"Filter by {col}"
                    })
                
                # Categorical filters (if few unique values)
                elif df[col].dtype == 'object' and df[col].nunique() < 20:
                    filters.append({
                        'column': col,
                        'type': 'multi_select',
                        'label': f"Filter by {col}",
                        'options': df[col].dropna().unique().tolist()[:20]
                    })
        
        return filters[:5]  # Limit to 5 global filters
    
    def to_dict(self, config: DashboardConfig) -> Dict[str, Any]:
        """Convert DashboardConfig to JSON-serializable dict."""
        return {
            'dashboard_id': config.dashboard_id,
            'title': config.title,
            'dashboard_type': config.dashboard_type.value,
            'persona': config.persona,
            'color_scheme': config.color_scheme,
            'created_for_files': config.created_for_files,
            'global_filters': config.global_filters,
            'sheets': [
                {
                    'sheet_id': sheet.sheet_id,
                    'title': sheet.title,
                    'icon': sheet.icon,
                    'layout': sheet.layout,
                    'charts': [
                        {
                            'chart_id': chart.chart_id,
                            'title': chart.title,
                            'chart_type': chart.chart_type,
                            'x_column': chart.x_column,
                            'y_column': chart.y_column,
                            'group_by': chart.group_by,
                            'aggregation': chart.aggregation,
                            'filters': chart.filters,
                            'size': chart.size,
                            'priority': chart.priority,
                            'description': chart.description
                        }
                        for chart in sorted(sheet.charts, key=lambda c: c.priority)
                    ]
                }
                for sheet in config.sheets
            ]
        }


# Singleton instance
dashboard_factory = DashboardFactory()

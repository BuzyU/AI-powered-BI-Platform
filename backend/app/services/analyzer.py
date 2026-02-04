# Dynamic Data Analyzer - Processes uploaded files and generates insights
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import logging
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """Analyzes uploaded datasets and generates dynamic insights."""
    
    def __init__(self):
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Dict] = {}
        self.relationships: List[Dict] = []
        self.analysis_result: Dict[str, Any] = {}
    
    async def load_file(self, file_path: str, file_type: str) -> Tuple[pd.DataFrame, Dict]:
        """Load and parse a data file."""
        path = Path(file_path)
        
        try:
            if file_type in ['csv']:
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
            elif file_type in ['xls', 'xlsx']:
                df = pd.read_excel(path)
            elif file_type == 'json':
                df = pd.read_json(path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Basic cleaning
            df = self._clean_dataframe(df)
            
            # Extract metadata
            metadata = self._extract_metadata(df, path.name)
            
            return df, metadata
            
        except Exception as e:
            logger.error(f"Failed to load file {path}: {e}")
            raise
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataframe - handle missing values, normalize columns."""
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Convert column names to lowercase for consistency
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Handle missing values for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                # Fill with median for numeric
                df[col] = df[col].fillna(df[col].median())
        
        # Handle missing values for string columns
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            df[col] = df[col].fillna('Unknown')
        
        return df
    
    def _extract_metadata(self, df: pd.DataFrame, filename: str) -> Dict:
        """Extract metadata from dataframe."""
        columns = []
        
        for col in df.columns:
            col_info = {
                'name': col,
                'original_name': col,
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'unique_count': int(df[col].nunique()),
                'sample_values': df[col].dropna().head(5).tolist()
            }
            
            # Detect column type
            col_info['semantic_type'] = self._detect_semantic_type(df[col], col)
            
            # Add stats for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info['stats'] = {
                    'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    'median': float(df[col].median()) if not pd.isna(df[col].median()) else None,
                    'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
                    'sum': float(df[col].sum()) if not pd.isna(df[col].sum()) else None,
                }
            
            columns.append(col_info)
        
        return {
            'filename': filename,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': columns,
            'detected_role': self._detect_dataset_role(df, columns)
        }
    
    def _detect_semantic_type(self, series: pd.Series, col_name: str) -> str:
        """Detect semantic type of a column."""
        name_lower = col_name.lower()
        
        # Check by name patterns
        if any(x in name_lower for x in ['id', 'key', 'code']):
            return 'identifier'
        if any(x in name_lower for x in ['date', 'time', 'created', 'updated']):
            return 'datetime'
        if any(x in name_lower for x in ['price', 'cost', 'amount', 'revenue', 'total', 'value']):
            return 'currency'
        if any(x in name_lower for x in ['percent', 'rate', 'ratio', 'margin']):
            return 'percentage'
        if any(x in name_lower for x in ['count', 'quantity', 'qty', 'number', 'num']):
            return 'count'
        if any(x in name_lower for x in ['name', 'title', 'label', 'description']):
            return 'text'
        if any(x in name_lower for x in ['email']):
            return 'email'
        if any(x in name_lower for x in ['phone', 'mobile']):
            return 'phone'
        if any(x in name_lower for x in ['address', 'city', 'state', 'country', 'zip']):
            return 'location'
        if any(x in name_lower for x in ['category', 'type', 'status', 'segment']):
            return 'category'
        
        # Check by data type
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
        
        # Check for date strings
        if series.dtype == 'object':
            sample = series.dropna().head(10)
            date_count = sum(self._looks_like_date(str(v)) for v in sample)
            if date_count >= len(sample) * 0.7:
                return 'datetime'
        
        return 'text'
    
    def _looks_like_date(self, value: str) -> bool:
        """Check if string looks like a date."""
        import re
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}-\d{2}-\d{4}',
        ]
        return any(re.match(p, value) for p in date_patterns)
    
    def _detect_dataset_role(self, df: pd.DataFrame, columns: List[Dict]) -> str:
        """Detect the role/type of this dataset."""
        col_names = [c['name'] for c in columns]
        semantic_types = [c['semantic_type'] for c in columns]
        
        # Transaction data
        if any('transaction' in c or 'order' in c for c in col_names):
            return 'transactions'
        if 'currency' in semantic_types and 'datetime' in semantic_types:
            return 'transactions'
        
        # Customer data
        if any('customer' in c or 'client' in c for c in col_names):
            return 'customers'
        if 'email' in semantic_types or 'phone' in semantic_types:
            return 'customers'
        
        # Product/Offering data
        if any('product' in c or 'item' in c or 'sku' in c for c in col_names):
            return 'products'
        
        # Time series / metrics
        if 'datetime' in semantic_types and semantic_types.count('numeric') >= 2:
            return 'timeseries'
        
        return 'general'
    
    def find_relationships(self, datasets: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Find potential relationships between datasets."""
        relationships = []
        dataset_names = list(datasets.keys())
        
        for i, name1 in enumerate(dataset_names):
            for name2 in dataset_names[i+1:]:
                df1, df2 = datasets[name1], datasets[name2]
                
                for col1 in df1.columns:
                    for col2 in df2.columns:
                        # Check name similarity
                        if self._columns_might_match(col1, col2):
                            # Check value overlap
                            overlap = self._calculate_overlap(df1[col1], df2[col2])
                            if overlap > 0.3:
                                relationships.append({
                                    'dataset1': name1,
                                    'column1': col1,
                                    'dataset2': name2,
                                    'column2': col2,
                                    'overlap': overlap,
                                    'type': 'potential_join'
                                })
        
        return relationships
    
    def _columns_might_match(self, col1: str, col2: str) -> bool:
        """Check if two columns might represent the same thing."""
        # Exact match
        if col1 == col2:
            return True
        
        # Common suffixes/prefixes
        c1, c2 = col1.lower(), col2.lower()
        
        # Remove common suffixes
        for suffix in ['_id', 'id', '_key', 'key', '_code', 'code']:
            c1 = c1.replace(suffix, '')
            c2 = c2.replace(suffix, '')
        
        return c1 == c2 or c1 in c2 or c2 in c1
    
    def _calculate_overlap(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate value overlap percentage between two columns."""
        try:
            set1 = set(series1.dropna().astype(str).unique())
            set2 = set(series2.dropna().astype(str).unique())
            
            if not set1 or not set2:
                return 0
            
            intersection = len(set1 & set2)
            smaller_set = min(len(set1), len(set2))
            
            return intersection / smaller_set if smaller_set > 0 else 0
        except:
            return 0
    
    def generate_analysis(self, datasets: Dict[str, Dict], filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive analysis from loaded datasets."""
        result = {
            'summary': {},
            'kpis': [],
            'charts': [],
            'insights': [],
            'tables': [],
            'slicers': []
        }
        
        all_data = []
        for name, info in datasets.items():
            df = info['df']
            meta = info['metadata']
            
            # Apply filters if any
            # Apply filters if any
            if filters:
                for col, val in filters.items():
                    if col in df.columns:
                        # Robust filtering: convert both to string to handle type mismatches (e.g. int vs "int")
                        try:
                            if isinstance(val, list):
                                # If list, assume exact match needed or string match
                                val_strs = [str(v) for v in val]
                                df = df[df[col].astype(str).isin(val_strs)]
                            else:
                                df = df[df[col].astype(str) == str(val)]
                        except Exception:
                            # Fallback (unlikely to be needed but safe)
                            if isinstance(val, list):
                                df = df[df[col].isin(val)]
                            else:
                                df = df[df[col] == val]
            
            all_data.append({'name': name, 'df': df, 'metadata': meta})
        
        if not all_data:
            return result
        
        # Merge datasets if possible for unified analysis
        primary_df = all_data[0]['df']
        primary_name = all_data[0]['name']
        primary_meta = all_data[0]['metadata']
        
        # Generate available slicers (Categorical columns with low cardinality)
        # Use unfiltered data to ensure slicers persist even after filtering reduces cardinality
        original_df = datasets[primary_name]['df']
        
        for col_info in primary_meta['columns']:
            if col_info['semantic_type'] == 'category' or (col_info['unique_count'] < 100):
                if col_info['name'] in original_df.columns:
                    # Get unique values from the ORIGINAL dataframe
                    unique_vals = sorted(original_df[col_info['name']].dropna().unique().tolist())
                    
                    if len(unique_vals) > 1 and len(unique_vals) < 50:
                        result['slicers'].append({
                            'column': col_info['name'],
                            'options': unique_vals
                        })

        # Generate KPIs based on numeric columns
        result['kpis'] = self._generate_kpis(primary_df, all_data[0]['metadata'])
        
        # Generate charts based on data types
        result['charts'] = self._generate_charts(primary_df, all_data[0]['metadata'])
        
        # Generate insights
        result['insights'] = self._generate_insights(primary_df, all_data[0]['metadata'])
        
        # Summary statistics
        result['summary'] = {
            'total_rows': sum(d['df'].shape[0] for d in all_data),
            'total_columns': sum(d['df'].shape[1] for d in all_data),
            'datasets': len(all_data),
            'data_quality': self._calculate_data_quality(all_data)
        }
        
        return result
    
    def _generate_kpis(self, df: pd.DataFrame, metadata: Dict) -> List[Dict]:
        """Generate KPIs from numeric columns."""
        kpis = []
        
        for col_info in metadata['columns']:
            if col_info['semantic_type'] in ['currency', 'numeric', 'count']:
                col = col_info['name']
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    stats = col_info.get('stats', {})
                    
                    # Total/Sum KPI
                    if col_info['semantic_type'] == 'currency':
                        kpis.append({
                            'label': f"Total {col.replace('_', ' ').title()}",
                            'value': stats.get('sum', 0),
                            'format': 'currency',
                            'column': col
                        })
                    
                    # Average KPI
                    kpis.append({
                        'label': f"Avg {col.replace('_', ' ').title()}",
                        'value': stats.get('mean', 0),
                        'format': 'currency' if col_info['semantic_type'] == 'currency' else 'number',
                        'column': col
                    })
        
        # Row count KPI
        kpis.append({
            'label': 'Total Records',
            'value': len(df),
            'format': 'number',
            'column': None
        })
        
        # Limit to top 6 most relevant KPIs
        return kpis[:6]
    
    def generate_custom_chart(self, df: pd.DataFrame, type: str, x_col: str, y_col: str = None, aggregation: str = 'sum') -> Dict:
        """Generate specific chart data based on user selection."""
        try:
            chart_data = {
                'type': type,
                'title': f'{aggregation.title()} of {y_col} by {x_col}' if y_col else f'{x_col} Distribution',
                'xLabel': x_col,
                'yLabel': y_col,
                'data': []
            }
            
            # Helper to handle grouping
            grouper = df[x_col]
            is_numeric_x = pd.api.types.is_numeric_dtype(df[x_col])
            
            if is_numeric_x and df[x_col].nunique() > 20:
                # auto-binning for high cardinality numeric X
                grouper = pd.cut(df[x_col], bins=10)
            elif pd.api.types.is_datetime64_any_dtype(df[x_col]):
                grouper = df[x_col].dt.to_period('M')

            # Handle grouping and aggregation
            if type in ['bar', 'line', 'area', 'donut']:
                if y_col:
                    grouped = df.groupby(grouper)[y_col].agg(aggregation)
                    # Sort by value for bar/donut, by index (time) for line/area
                    if type in ['bar', 'donut']:
                        grouped = grouped.sort_values(ascending=False).head(20)
                    else:
                        grouped = grouped.sort_index().head(50) # limit points
                else:
                     # Count frequency
                    grouped = df[x_col].value_counts(bins=10 if is_numeric_x and df[x_col].nunique() > 20 else None).head(20)

                # Format data for frontend
                chart_data['data'] = []
                for cat, val in grouped.items():
                    label = str(cat)
                    if is_numeric_x and isinstance(cat, pd.Interval):
                        label = f"{cat.left:.1f}-{cat.right:.1f}"
                    
                    if type in ['line', 'area']:
                         chart_data['data'].append({'x': label, 'y': float(val)})
                    else:
                         chart_data['data'].append({'label': label, 'value': float(val)})
            
            return chart_data

        except Exception as e:
            logger.error(f"Custom chart generation failed: {e}")
            raise ValueError(f"Could not generate chart: {str(e)}")

    def execute_custom_plot(self, df: pd.DataFrame, code: str) -> str:
        """Execute custom python code to generate plot."""
        try:
            # Create a new figure
            plt.figure(figsize=(10, 6))
            
            # Define safe locals
            local_vars = {'df': df, 'pd': pd, 'np': np, 'plt': plt, 'sns': sns}
            
            # Execute code
            exec(code, {}, local_vars)
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            
            # Encode
            return base64.b64encode(buf.getvalue()).decode('utf-8')
            
        except Exception as e:
            plt.close()
            logger.error(f"Plot execution failed: {e}")
            raise ValueError(f"Error in python code: {str(e)}")

    def _generate_charts(self, df: pd.DataFrame, metadata: Dict) -> List[Dict]:
        """Generate chart configurations based on data types."""
        charts = []
        
        numeric_cols = [c for c in metadata['columns'] if c['semantic_type'] in ['currency', 'numeric', 'count', 'percentage']]
        category_cols = [c for c in metadata['columns'] if c['semantic_type'] == 'category']
        datetime_cols = [c for c in metadata['columns'] if c['semantic_type'] == 'datetime']
        
        # 1. Time Series / Area Chart
        if datetime_cols and numeric_cols:
            date_col = datetime_cols[0]['name']
            value_col = numeric_cols[0]['name']
            
            try:
                df_copy = df.copy()
                df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                grouped = df_copy.groupby(df_copy[date_col].dt.to_period('M'))[value_col].sum()
                
                # Line Chart (Trend)
                charts.append({
                    'type': 'line',
                    'title': f'{value_col.replace("_", " ").title()} Trend',
                    'data': [{'x': str(period), 'y': float(val)} for period, val in grouped.items()],
                    'xLabel': 'Period',
                    'yLabel': value_col
                })

                # Area Chart (Volume)
                charts.append({
                    'type': 'area',
                    'title': f'{value_col.replace("_", " ").title()} Volume Over Time',
                    'data': [{'x': str(period), 'y': float(val)} for period, val in grouped.items()],
                    'xLabel': 'Period',
                    'yLabel': value_col
                })
            except Exception as e:
                logger.warning(f"Failed to create time series chart: {e}")
        
        # 2. Bar Chart (Top Categories)
        if category_cols and numeric_cols:
            cat_col = category_cols[0]['name']
            value_col = numeric_cols[0]['name']
            grouped = df.groupby(cat_col)[value_col].sum().sort_values(ascending=False).head(10)
            
            charts.append({
                'type': 'bar',
                'title': f'Top {cat_col} by {value_col}',
                'data': [{'label': str(cat), 'value': float(val)} for cat, val in grouped.items()],
                'xLabel': cat_col,
                'yLabel': value_col
            })
        
        # 3. Donut Chart (Distribution)
        if category_cols:
            cat_col = category_cols[0]['name']
            counts = df[cat_col].value_counts().head(6)
            charts.append({
                'type': 'donut',
                'title': f'{cat_col} Distribution',
                'data': [{'label': str(cat), 'value': int(count)} for cat, count in counts.items()],
                'xLabel': cat_col
            })
        
        # 4. Scatter Plot (Correlation between 2 numerics)
        if len(numeric_cols) >= 2:
            try:
                col1 = numeric_cols[0]['name']
                col2 = numeric_cols[1]['name']
                # Sample 100 points for performance
                sample = df[[col1, col2]].dropna().sample(min(100, len(df)))
                
                charts.append({
                    'type': 'scatter',
                    'title': f'{col1} vs {col2}',
                    'data': [{'x': float(row[col1]), 'y': float(row[col2])} for i, row in sample.iterrows()],
                    'xLabel': col1,
                    'yLabel': col2
                })
            except Exception as e:
                pass

        # 5. Box Plot (Outlier Detection)
        if numeric_cols:
            col = numeric_cols[0]['name']
            try:
                q1 = df[col].quantile(0.25)
                median = df[col].median()
                q3 = df[col].quantile(0.75)
                min_val = df[col].min()
                max_val = df[col].max()
                
                charts.append({
                    'type': 'box',
                    'title': f'{col} Distribution Analysis',
                    'stats': {
                        'min': float(min_val),
                        'q1': float(q1),
                        'median': float(median),
                        'q3': float(q3),
                        'max': float(max_val)
                    }
                })
            except:
                pass

        # 6. Heatmap (Correlation Matrix)
        if len(numeric_cols) >= 3:
            try:
                # Calculate correlation matrix
                cols_to_use = [c['name'] for c in numeric_cols[:5]] # Top 5 metrics
                corr_matrix = df[cols_to_use].corr()
                
                heatmap_data = []
                for i, row in enumerate(corr_matrix.index):
                    for j, col in enumerate(corr_matrix.columns):
                        heatmap_data.append({
                            'x': col,
                            'y': row,
                            'value': float(corr_matrix.iloc[i, j])
                        })
                
                charts.append({
                    'type': 'heatmap',
                    'title': 'Metric Correlations',
                    'data': heatmap_data,
                    'xLabel': 'Metrics',
                    'yLabel': 'Metrics'
                })
            except:
                pass
        
        return charts
    
    def _generate_insights(self, df: pd.DataFrame, metadata: Dict) -> List[Dict]:
        """Generate insights from data analysis."""
        insights = []
        
        for col_info in metadata['columns']:
            col = col_info['name']
            
            # High null percentage warning
            null_pct = col_info['null_count'] / len(df) * 100 if len(df) > 0 else 0
            if null_pct > 20:
                insights.append({
                    'type': 'warning',
                    'title': f'High Missing Values in {col}',
                    'description': f'{null_pct:.1f}% of values are missing in column {col}',
                    'column': col
                })
            
            # Check for outliers in numeric columns
            if col_info['semantic_type'] in ['currency', 'numeric'] and 'stats' in col_info:
                stats = col_info['stats']
                if stats.get('std') and stats.get('mean'):
                    cv = stats['std'] / abs(stats['mean']) if stats['mean'] != 0 else 0
                    if cv > 1.5:
                        insights.append({
                            'type': 'info',
                            'title': f'High Variability in {col}',
                            'description': f'{col} shows high variability (CV: {cv:.2f}). Consider investigating outliers.',
                            'column': col
                        })
            
            # Low cardinality for potential categorization
            if col_info['unique_count'] < 10 and col_info['unique_count'] > 1:
                if col_info['semantic_type'] not in ['category', 'identifier']:
                    insights.append({
                        'type': 'tip',
                        'title': f'{col} Could Be Categorical',
                        'description': f'Column {col} has only {col_info["unique_count"]} unique values. Consider treating as category.',
                        'column': col
                    })
        
        return insights[:10]  # Limit insights
    
    def _calculate_data_quality(self, all_data: List[Dict]) -> float:
        """Calculate overall data quality score."""
        scores = []
        
        for data in all_data:
            df = data['df']
            meta = data['metadata']
            
            # Completeness score
            total_cells = df.shape[0] * df.shape[1]
            null_cells = df.isnull().sum().sum()
            completeness = 1 - (null_cells / total_cells) if total_cells > 0 else 1
            
            # Uniqueness score (for identifier columns)
            id_cols = [c for c in meta['columns'] if c['semantic_type'] == 'identifier']
            uniqueness = 1.0
            if id_cols:
                for col_info in id_cols:
                    col = col_info['name']
                    if col in df.columns:
                        unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 1
                        uniqueness = min(uniqueness, unique_ratio)
            
            score = (completeness * 0.6 + uniqueness * 0.4) * 100
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0


# Singleton instance
analyzer = DataAnalyzer()


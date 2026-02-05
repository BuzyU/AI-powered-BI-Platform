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
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """Analyzes uploaded datasets and generates dynamic insights."""
    
    def __init__(self):
        # Detection Patterns
        self.PATTERNS = {
            # Business / Sales
            'sales': ['revenue', 'sales', 'order', 'transaction', 'sku', 'invoice', 'amount', 'profit', 'margin'],
            'products': ['product', 'sku', 'inventory', 'stock', 'unit_cost', 'item', 'category'],
            'services': ['service', 'subscription', 'plan', 'renewal', 'mrr', 'arr', 'license'],
            'customers': ['customer', 'client', 'email', 'phone', 'address', 'segment', 'crm'],
            
            # Specialized Business
            'marketing': ['campaign', 'click', 'impression', 'lead', 'conversion', 'ad_spend', 'cpc'],
            'support': ['ticket', 'issue', 'priority', 'resolution', 'agent', 'sla', 'feedback'],
            'financial': ['budget', 'expense', 'asset', 'liability', 'equity', 'tax', 'ledger'],
            
            # Legal
            'legal': ['case', 'docket', 'plaintiff', 'defendant', 'court', 'verdict', 'judge', 'statute', 'litigation'],
            'contracts': ['contract', 'agreement', 'clause', 'effective_date', 'party', 'signature', 'term'],
            
            # Model / AI
            'model_metrics': ['accuracy', 'loss', 'precision', 'recall', 'f1', 'epoch', 'batch', 'learning_rate'],
            'transformer': ['attention', 'head', 'layer', 'embedding', 'token', 'vocab', 'mask'],
        }

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
                # Excel files - use openpyxl for xlsx, xlrd for xls
                try:
                    if file_type == 'xlsx':
                        df = pd.read_excel(path, engine='openpyxl')
                    else:
                        df = pd.read_excel(path, engine='xlrd')
                except Exception as e:
                    # Fallback to default engine
                    logger.warning(f"Excel engine fallback: {e}")
                    df = pd.read_excel(path)
            elif file_type == 'json':
                # JSON files - handle different formats
                try:
                    df = pd.read_json(path)
                except ValueError:
                    # Try reading as lines (JSONL format)
                    try:
                        df = pd.read_json(path, lines=True)
                    except:
                        # Try reading as regular JSON and normalize
                        import json as json_lib
                        with open(path, 'r', encoding='utf-8') as f:
                            data = json_lib.load(f)
                        if isinstance(data, list):
                            df = pd.DataFrame(data)
                        elif isinstance(data, dict):
                            # Check if it's a nested dict with data arrays
                            if any(isinstance(v, list) for v in data.values()):
                                df = pd.DataFrame(data)
                            else:
                                df = pd.DataFrame([data])
                        else:
                            raise ValueError("Unsupported JSON structure")
            elif file_type in ['pt', 'onnx', 'h5', 'pkl', 'joblib']:
                # Model file - return placeholder DF
                return pd.DataFrame({'status': ['Model File Loaded'], 'path': [str(path)]}), {
                    'filename': path.name,
                    'file_type': file_type,
                    'is_model': True,
                    'model_type': self._detect_model_type(path)
                }
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
    
    def _detect_model_type(self, path: Path) -> str:
        """Simple heuristic for model type based on extension."""
        ext = path.suffix.lower()
        if ext == '.pt': return "PyTorch Model"
        if ext == '.onnx': return "ONNX Model"
        if ext == '.h5': return "Keras/H5 Model"
        if ext == '.pkl': return "Pickled Model"
        return "Unknown Model"

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataframe - handle missing values, normalize columns."""
        # Strip whitespace from column names
        df.columns = df.columns.astype(str).str.strip()
        
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
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'unique_count': int(df[col].nunique()),
                'sample_values': df[col].dropna().astype(str).head(5).tolist()
            }
            
            # Detect column type
            col_info['semantic_type'] = self._detect_semantic_type(df[col], col)
            
            # Add stats for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info['stats'] = {
                    'min': self._safe_float(df[col].min()),
                    'max': self._safe_float(df[col].max()),
                    'mean': self._safe_float(df[col].mean()),
                    'median': self._safe_float(df[col].median()),
                    'std': self._safe_float(df[col].std()),
                    'sum': self._safe_float(df[col].sum()),
                }
            
            columns.append(col_info)
        
        # Detect Content Type (The "Julius" logic)
        content_type = self.detect_content_type(df, columns)
        
        return {
            'filename': filename,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': columns,
            'detected_role': content_type['role'],
            'detected_type': content_type['type'],
            'confidence': content_type['confidence']
        }

    def detect_content_type(self, df: pd.DataFrame, columns: List[Dict]) -> Dict:
        """
        Sophisticated detection of dataset content/role.
        Returns {role: str, type: str, confidence: float}
        """
        col_names = set(c['name'] for c in columns)
        col_text = " ".join(col_names).lower()
        
        best_match = {'role': 'General', 'type': 'Unknown', 'score': 0}
        
        for category, keywords in self.PATTERNS.items():
            matches = sum(1 for k in keywords if any(k in c for c in col_names))
            score = matches / len(keywords) if keywords else 0
            
            # Bonus for exact matches
            exact_matches = sum(1 for k in keywords if k in col_names)
            score += (exact_matches * 0.2)
            
            if score > best_match['score']:
                best_match = {'role': category, 'type': category.title(), 'score': score}

        # Normalize confidence to 0-100
        confidence = min(100, int(best_match['score'] * 100))
        
        # Fallbacks
        if confidence < 20: 
            return {'role': 'general', 'type': 'General Dataset', 'confidence': confidence}
        
        return {'role': best_match['role'], 'type': best_match['type'].replace('_', ' '), 'confidence': confidence}

    def _safe_float(self, val):
        """Safely convert to float, handling NaN/Inf."""
        if pd.isna(val) or np.isinf(val):
            return 0.0
        try:
            return float(val)
        except:
            return 0.0
    
    def _detect_semantic_type(self, series: pd.Series, col_name: str) -> str:
        """Detect semantic type of a column."""
        name_lower = col_name.lower()
        
        if any(x in name_lower for x in ['id', 'key', 'code']): return 'identifier'
        if any(x in name_lower for x in ['date', 'time', 'created']): return 'datetime'
        if any(x in name_lower for x in ['price', 'cost', 'amount', 'revenue', 'total']): return 'currency'
        if any(x in name_lower for x in ['percent', 'rate', 'margin']): return 'percentage'
        if any(x in name_lower for x in ['count', 'quantity', 'qty']): return 'count'
        if any(x in name_lower for x in ['email']): return 'email'
        if any(x in name_lower for x in ['phone']): return 'phone'
        if any(x in name_lower for x in ['address', 'city', 'country']): return 'location'
        if any(x in name_lower for x in ['category', 'type', 'status']): return 'category'
        
        if pd.api.types.is_numeric_dtype(series): return 'numeric'
        if pd.api.types.is_datetime64_any_dtype(series): return 'datetime'
        
        # Date string check
        if series.dtype == 'object':
            sample = series.dropna().astype(str).head(10)
            date_count = sum(self._looks_like_date(str(v)) for v in sample)
            if len(sample) > 0 and date_count >= len(sample) * 0.7:
                return 'datetime'
        
        return 'text'
    
    def _looks_like_date(self, value: str) -> bool:
        """Check if string looks like a date."""
        date_patterns = [r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}', r'\d{2}-\d{2}-\d{4}']
        return any(re.match(p, value) for p in date_patterns)
    
    def find_relationships(self, datasets: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Find potential relationships between datasets."""
        relationships = []
        dataset_names = list(datasets.keys())
        
        for i, name1 in enumerate(dataset_names):
            for name2 in dataset_names[i+1:]:
                df1, df2 = datasets[name1], datasets[name2]
                
                for col1 in df1.columns:
                    for col2 in df2.columns:
                        if self._columns_might_match(col1, col2):
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
        if col1 == col2: return True
        c1, c2 = col1.lower(), col2.lower()
        suffixes = ['_id', 'id', '_key', 'key', '_code', 'code']
        for s in suffixes:
            if c1.endswith(s): c1 = c1.replace(s, '')
            if c2.endswith(s): c2 = c2.replace(s, '')
        return c1 == c2 or c1 in c2 or c2 in c1
    
    def _calculate_overlap(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate value overlap percentage."""
        try:
            set1 = set(series1.dropna().astype(str).unique())
            set2 = set(series2.dropna().astype(str).unique())
            if not set1 or not set2: return 0
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
            'slicers': []
        }
        
        if not datasets: return result
        
        # Determine primary dataset (usually first one or most rows)
        primary_name = list(datasets.keys())[0]
        primary_data = datasets[primary_name]
        primary_df = primary_data['df']
        primary_meta = primary_data['metadata']
        
        # Apply filters
        df = primary_df
        if filters:
            for col, val in filters.items():
                if col in df.columns:
                    try:
                        if isinstance(val, list):
                            val_strs = [str(v) for v in val]
                            df = df[df[col].astype(str).isin(val_strs)]
                        else:
                            df = df[df[col].astype(str) == str(val)]
                    except: pass
        
        # Slicers
        for col_info in primary_meta['columns']:
            if col_info['semantic_type'] == 'category' or (col_info['unique_count'] < 50):
                if col_info['name'] in df.columns:
                    vals = sorted(df[col_info['name']].dropna().astype(str).unique().tolist())
                    if 1 < len(vals) < 50:
                        result['slicers'].append({'column': col_info['name'], 'options': vals})
        
        # KPIs
        result['kpis'] = self._generate_kpis(df, primary_meta)
        
        # Charts
        result['charts'] = self._generate_charts(df, primary_meta)
        
        # Insights
        result['insights'] = self._generate_insights(df, primary_meta)
        
        # Summary
        result['summary'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'datasets': len(datasets),
            'data_quality': 85.0 # Placeholder
        }
        
        return result
    
    def _generate_kpis(self, df: pd.DataFrame, metadata: Dict) -> List[Dict]:
        kpis = []
        for col_info in metadata['columns']:
            col = col_info['name']
            if col_info['semantic_type'] in ['currency', 'numeric', 'count'] and col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    kpis.append({
                        'label': f"Total {col.replace('_', ' ').title()}",
                        'value': self._safe_float(df[col].sum()),
                        'format': 'currency' if col_info['semantic_type'] == 'currency' else 'number',
                        'column': col
                    })
                    kpis.append({
                        'label': f"Avg {col.replace('_', ' ').title()}",
                        'value': self._safe_float(df[col].mean()),
                        'format': 'currency' if col_info['semantic_type'] == 'currency' else 'number',
                        'column': col
                    })
        kpis.append({'label': 'Total Records', 'value': len(df), 'format': 'number', 'column': None})
        return kpis[:6]

    def _generate_charts(self, df: pd.DataFrame, metadata: Dict) -> List[Dict]:
        """Generate chart configurations."""
        charts = []
        numeric_cols = [c['name'] for c in metadata['columns'] if c['semantic_type'] in ['currency', 'numeric', 'count', 'percentage']]
        category_cols = [c['name'] for c in metadata['columns'] if c['semantic_type'] == 'category']
        datetime_cols = [c['name'] for c in metadata['columns'] if c['semantic_type'] == 'datetime']

        # Time Series
        if datetime_cols and numeric_cols:
            try:
                date_col, val_col = datetime_cols[0], numeric_cols[0]
                df_c = df.copy()
                df_c[date_col] = pd.to_datetime(df_c[date_col], errors='coerce')
                grouped = df_c.groupby(df_c[date_col].dt.to_period('M'))[val_col].sum()
                charts.append({
                    'type': 'line',
                    'title': f'{val_col} Trend',
                    'data': [{'x': str(p), 'y': self._safe_float(v)} for p, v in grouped.items()],
                    'xLabel': 'Date', 'yLabel': val_col
                })
            except: pass
            
        # Bar Chart
        if category_cols and numeric_cols:
            try:
                cat, val = category_cols[0], numeric_cols[0]
                grouped = df.groupby(cat)[val].sum().sort_values(ascending=False).head(10)
                charts.append({
                    'type': 'bar',
                    'title': f'Top {cat} by {val}',
                    'data': [{'label': str(k), 'value': self._safe_float(v)} for k, v in grouped.items()],
                    'xLabel': cat, 'yLabel': val
                })
            except: pass
            
        return charts

    def _generate_insights(self, df: pd.DataFrame, metadata: Dict) -> List[Dict]:
        insights = []
        for col in metadata['columns']:
            name = col['name']
            null_pct = col['null_count'] / len(df) * 100 if len(df) > 0 else 0
            if null_pct > 20:
                insights.append({'type': 'warning', 'title': f'High Missing Values in {name}', 'description': f'{null_pct:.1f}% missing', 'column': name})
        return insights[:5]
    
    def generate_custom_chart(self, df: pd.DataFrame, type: str, x_col: str, y_col: str = None, aggregation: str = 'sum') -> Dict:
        """Generate specific chart data."""
        chart_data = {'type': type, 'title': f'{aggregation} {y_col} by {x_col}', 'data': []}
        
        try:
            if y_col:
                if aggregation == 'sum': val = df.groupby(x_col)[y_col].sum()
                elif aggregation == 'mean': val = df.groupby(x_col)[y_col].mean()
                elif aggregation == 'count': val = df.groupby(x_col)[y_col].count()
                else: val = df.groupby(x_col)[y_col].sum()
            else:
                val = df[x_col].value_counts()
            
            # Limit results
            val = val.head(20)
                
            for k, v in val.items():
                if type in ['line', 'area', 'scatter']:
                    chart_data['data'].append({'x': str(k), 'y': self._safe_float(v)})
                else:
                    chart_data['data'].append({'label': str(k), 'value': self._safe_float(v)})
            
            return chart_data
        except Exception as e:
            logger.error(f"Custom chart error: {e}")
            raise

    def execute_custom_plot(self, df: pd.DataFrame, code: str) -> str:
        """Execute custom python code."""
        try:
            plt.figure(figsize=(10, 6))
            local_vars = {'df': df, 'pd': pd, 'np': np, 'plt': plt, 'sns': sns}
            exec(code, {}, local_vars)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            plt.close()
            logger.error(f"Plot error: {e}")
            raise

# Singleton instance
analyzer = DataAnalyzer()

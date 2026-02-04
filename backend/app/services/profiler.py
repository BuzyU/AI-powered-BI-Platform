# Advanced Data Profiler - Complete Statistical Analysis
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataProfiler:
    """
    Comprehensive data profiler that analyzes:
    - Missing values patterns
    - Outliers detection
    - Distribution analysis (skewness, kurtosis)
    - Duplicates
    - Correlations
    - Data quality scoring
    """
    
    def profile_dataset(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Dict[str, Any]:
        """Generate complete profile for a dataset."""
        
        profile = {
            "dataset_name": dataset_name,
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": [],
            "issues": [],
            "correlations": [],
            "duplicates": self._analyze_duplicates(df),
            "relationships_ready": True,
            "overall_quality": 0,
            "cleaning_suggestions": []
        }
        
        # Profile each column
        for col in df.columns:
            col_profile = self._profile_column(df, col)
            profile["columns"].append(col_profile)
            
            # Collect issues
            profile["issues"].extend(col_profile.get("issues", []))
        
        # Calculate correlations for numeric columns
        profile["correlations"] = self._calculate_correlations(df)
        
        # Calculate overall quality score
        profile["overall_quality"] = self._calculate_quality_score(df, profile["columns"])
        
        # Generate cleaning suggestions
        profile["cleaning_suggestions"] = self._generate_cleaning_suggestions(profile)
        
        return profile
    
    def _profile_column(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Profile a single column."""
        series = df[col]
        total_rows = len(df)
        
        profile = {
            "name": col,
            "dtype": str(series.dtype),
            "issues": []
        }
        
        # Missing values analysis
        missing_count = int(series.isnull().sum())
        missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0
        profile["missing"] = {
            "count": missing_count,
            "percentage": round(missing_pct, 2),
            "pattern": self._detect_missing_pattern(series)
        }
        
        if missing_pct > 0:
            severity = "critical" if missing_pct > 30 else "warning" if missing_pct > 10 else "info"
            profile["issues"].append({
                "type": "Missing Values",
                "issue_type": "Missing Values",
                "severity": severity,
                "column": col,
                "description": f"{missing_pct:.1f}% missing values ({missing_count} rows)",
                "fix_options": self._get_missing_fix_options(series, col)
            })
        
        # Unique values
        profile["unique"] = {
            "count": int(series.nunique()),
            "percentage": round(series.nunique() / total_rows * 100, 2) if total_rows > 0 else 0
        }
        
        # Detect semantic type
        profile["semantic_type"] = self._detect_type(series, col)
        
        # For numeric columns
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = series.dropna()
            
            if len(numeric_series) > 0:
                profile["statistics"] = {
                    "mean": round(float(numeric_series.mean()), 4),
                    "median": round(float(numeric_series.median()), 4),
                    "mode": round(float(numeric_series.mode().iloc[0]), 4) if len(numeric_series.mode()) > 0 else None,
                    "std": round(float(numeric_series.std()), 4),
                    "min": round(float(numeric_series.min()), 4),
                    "max": round(float(numeric_series.max()), 4),
                    "q1": round(float(numeric_series.quantile(0.25)), 4),
                    "q3": round(float(numeric_series.quantile(0.75)), 4),
                    "sum": round(float(numeric_series.sum()), 4),
                    "variance": round(float(numeric_series.var()), 4)
                }
                
                # Skewness and Kurtosis
                profile["distribution"] = {
                    "skewness": round(float(stats.skew(numeric_series)), 4),
                    "kurtosis": round(float(stats.kurtosis(numeric_series)), 4),
                    "is_normal": self._test_normality(numeric_series),
                    "interpretation": self._interpret_distribution(numeric_series)
                }
                
                # Outliers detection
                profile["outliers"] = self._detect_outliers(numeric_series, col)
                
                if profile["outliers"]["count"] > 0:
                    pct = profile["outliers"]["count"] / len(numeric_series) * 100
                    severity = "warning" if pct > 5 else "info"
                    profile["issues"].append({
                        "type": "Outliers",
                        "issue_type": "Outliers", # consistency
                        "severity": severity,
                        "column": col,
                        "description": f"{profile['outliers']['count']} outliers detected ({pct:.1f}%)",
                        "fix_options": [
                            {"method": "remove", "description": "Remove outlier rows"},
                            {"method": "cap_iqr", "description": "Cap at IQR boundaries", "formula": "Cap values to Q1-1.5*IQR and Q3+1.5*IQR"},
                            {"method": "cap_percentile", "description": "Cap at 1st/99th percentile"},
                            {"method": "keep", "description": "Keep outliers (no action)"}
                        ],
                        "values": profile["outliers"]["values"][:10]  # First 10 outliers
                    })
                
                # Check for significant skewness
                skew_val = profile["distribution"]["skewness"]
                if abs(skew_val) > 1:
                    direction = "Right (Positive)" if skew_val > 0 else "Left (Negative)"
                    severity = "warning" if abs(skew_val) > 3 else "info"
                    profile["issues"].append({
                        "type": "Skewed Distribution",
                        "issue_type": "Skewed Distribution",
                        "severity": severity,
                        "column": col,
                        "description": f"Highly {direction} skewed (skew={skew_val}).",
                        "fix_options": [
                           {"method": "log_transform", "description": "Apply Log Transformation", "formula": f"np.log1p(df['{col}'])"},
                           {"method": "sqrt_transform", "description": "Apply Square Root Transformation", "formula": f"np.sqrt(df['{col}'])"},
                           {"method": "keep", "description": "Keep as is"}
                        ]
                    })
        
        # For categorical/text columns
        else:
            value_counts = series.value_counts()
            profile["top_values"] = [
                {"value": str(v), "count": int(c), "percentage": round(c/total_rows*100, 2)}
                for v, c in value_counts.head(10).items()
            ]
            profile["cardinality"] = {
                "unique_values": int(series.nunique()),
                "is_high_cardinality": bool(series.nunique() > total_rows * 0.5)
            }
        
        # Sample values
        profile["samples"] = [str(v) for v in series.dropna().head(5).tolist()]
        
        return profile
    
    def _detect_missing_pattern(self, series: pd.Series) -> str:
        """Detect pattern of missing values."""
        if series.isnull().sum() == 0:
            return "none"
        
        # Check if missing at start/end (time series pattern)
        first_valid = series.first_valid_index()
        last_valid = series.last_valid_index()
        
        if first_valid is None:
            return "all_missing"
        
        # Check random vs systematic
        missing_indices = series.isnull()
        runs = (missing_indices != missing_indices.shift()).cumsum()
        run_lengths = missing_indices.groupby(runs).sum()
        
        if run_lengths.max() > len(series) * 0.1:
            return "systematic_blocks"
        
        return "random"
    
    def _get_missing_fix_options(self, series: pd.Series, col: str) -> List[Dict]:
        """Get options for fixing missing values."""
        options = [
            {"method": "drop", "description": "Remove rows with missing values"}
        ]
        
        if pd.api.types.is_numeric_dtype(series):
            mean_val = series.mean()
            median_val = series.median()
            options.extend([
                {"method": "fill_mean", "description": f"Fill with mean ({mean_val:.2f})", "formula": f"df['{col}'].fillna(df['{col}'].mean())"},
                {"method": "fill_median", "description": f"Fill with median ({median_val:.2f})", "formula": f"df['{col}'].fillna(df['{col}'].median())"},
                {"method": "fill_zero", "description": "Fill with 0", "formula": f"df['{col}'].fillna(0)"},
                {"method": "interpolate", "description": "Linear interpolation", "formula": f"df['{col}'].interpolate(method='linear')"},
                {"method": "ffill", "description": "Forward fill (previous value)", "formula": f"df['{col}'].ffill()"},
            ])
        else:
            mode_val = series.mode().iloc[0] if len(series.mode()) > 0 else "Unknown"
            options.extend([
                {"method": "fill_mode", "description": f"Fill with mode ({mode_val})", "formula": f"df['{col}'].fillna('{mode_val}')"},
                {"method": "fill_unknown", "description": "Fill with 'Unknown'", "formula": f"df['{col}'].fillna('Unknown')"},
            ])
        
        options.append({"method": "keep", "description": "Keep missing (no action)"})
        
        return options
    
    def _detect_type(self, series: pd.Series, col: str) -> str:
        """Detect semantic type of column."""
        name = col.lower()
        
        # By name patterns
        if any(x in name for x in ['id', 'key', 'code']):
            return 'identifier'
        if any(x in name for x in ['date', 'time', 'created', 'updated', 'timestamp']):
            return 'datetime'
        if any(x in name for x in ['price', 'cost', 'amount', 'revenue', 'total', 'salary', 'income']):
            return 'currency'
        if any(x in name for x in ['percent', 'rate', 'ratio', 'pct']):
            return 'percentage'
        if any(x in name for x in ['count', 'quantity', 'qty', 'number', 'num', 'age']):
            return 'integer'
        if any(x in name for x in ['name', 'title', 'description']):
            return 'text'
        if any(x in name for x in ['email']):
            return 'email'
        if any(x in name for x in ['category', 'type', 'status', 'gender', 'class']):
            return 'category'
        
        # By dtype
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
        
        # Check if low cardinality (potential category)
        if series.nunique() < 20 and len(series) > 100:
            return 'category'
        
        return 'text'
    
    def _detect_outliers(self, series: pd.Series, col: str) -> Dict:
        """Detect outliers using IQR method."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        return {
            "method": "IQR",
            "count": len(outliers),
            "lower_bound": round(float(lower_bound), 4),
            "upper_bound": round(float(upper_bound), 4),
            "values": sorted([round(float(v), 4) for v in outliers.tolist()]),
            "indices": outliers.index.tolist()[:50]  # First 50 indices
        }
    
    def _test_normality(self, series: pd.Series) -> Dict:
        """Test if distribution is normal."""
        if len(series) < 8:
            return {"is_normal": None, "test": "insufficient_data"}
        
        try:
            # Shapiro-Wilk test (works for n < 5000)
            if len(series) < 5000:
                stat, p_value = stats.shapiro(series.sample(min(len(series), 500)))
            else:
                # For large datasets, use D'Agostino's K-squared test
                stat, p_value = stats.normaltest(series.sample(1000))
            
            return {
                "is_normal": bool(p_value > 0.05),
                "p_value": round(float(p_value), 4),
                "test": "shapiro" if len(series) < 5000 else "dagostino"
            }
        except:
            return {"is_normal": None, "test": "error"}
    
    def _interpret_distribution(self, series: pd.Series) -> str:
        """Interpret the distribution shape."""
        skewness = stats.skew(series)
        kurtosis = stats.kurtosis(series)
        
        interpretations = []
        
        if abs(skewness) < 0.5:
            interpretations.append("approximately symmetric")
        elif skewness > 0:
            interpretations.append("right-skewed (positively skewed) - tail extends right")
        else:
            interpretations.append("left-skewed (negatively skewed) - tail extends left")
        
        if kurtosis > 1:
            interpretations.append("heavy-tailed (leptokurtic) - more outliers than normal")
        elif kurtosis < -1:
            interpretations.append("light-tailed (platykurtic) - fewer outliers than normal")
        else:
            interpretations.append("normal-like tails (mesokurtic)")
        
        return "; ".join(interpretations)
    
    def _calculate_correlations(self, df: pd.DataFrame) -> List[Dict]:
        """Calculate correlations between numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return []
        
        correlations = []
        
        corr_matrix = df[numeric_cols].corr()
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr_val = corr_matrix.loc[col1, col2]
                if not pd.isna(corr_val):
                    correlation = {
                        "column1": col1,
                        "column2": col2,
                        "correlation": round(float(corr_val), 4),
                        "strength": self._interpret_correlation(corr_val)
                    }
                    
                    # Only include significant correlations
                    if abs(corr_val) > 0.3:
                        correlations.append(correlation)
        
        return sorted(correlations, key=lambda x: abs(x["correlation"]), reverse=True)
    
    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation strength."""
        abs_corr = abs(corr)
        direction = "positive" if corr > 0 else "negative"
        
        if abs_corr > 0.8:
            return f"very strong {direction}"
        elif abs_corr > 0.6:
            return f"strong {direction}"
        elif abs_corr > 0.4:
            return f"moderate {direction}"
        elif abs_corr > 0.2:
            return f"weak {direction}"
        else:
            return "negligible"
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict:
        """Analyze duplicate rows."""
        total_duplicates = df.duplicated().sum()
        
        return {
            "count": int(total_duplicates),
            "percentage": round(total_duplicates / len(df) * 100, 2) if len(df) > 0 else 0,
            "first_duplicate_indices": df[df.duplicated()].index.tolist()[:20]
        }
    
    def _calculate_quality_score(self, df: pd.DataFrame, columns: List[Dict]) -> float:
        """Calculate overall data quality score (0-100)."""
        scores = []
        
        # Completeness score (40% weight)
        total_cells = df.shape[0] * df.shape[1]
        null_cells = df.isnull().sum().sum()
        completeness = (1 - null_cells / total_cells) * 100 if total_cells > 0 else 100
        scores.append(completeness * 0.4)
        
        # Uniqueness score for identifiers (20% weight)
        id_cols = [c for c in columns if c.get("semantic_type") == "identifier"]
        if id_cols:
            uniqueness_scores = []
            for col_info in id_cols:
                col = col_info["name"]
                if col in df.columns:
                    uniqueness_scores.append(df[col].nunique() / len(df) * 100 if len(df) > 0 else 100)
            uniqueness = sum(uniqueness_scores) / len(uniqueness_scores) if uniqueness_scores else 100
        else:
            uniqueness = 100
        scores.append(uniqueness * 0.2)
        
        # Outlier score (20% weight)
        outlier_pct = 0
        outlier_count = 0
        for col_info in columns:
            if "outliers" in col_info:
                outlier_count += col_info["outliers"]["count"]
        outlier_pct = (outlier_count / (len(df) * len(columns))) * 100 if len(df) * len(columns) > 0 else 0
        outlier_score = max(0, 100 - outlier_pct * 10)  # Penalize outliers
        scores.append(outlier_score * 0.2)
        
        # Duplicate score (20% weight)
        dup_pct = df.duplicated().sum() / len(df) * 100 if len(df) > 0 else 0
        dup_score = max(0, 100 - dup_pct * 2)
        scores.append(dup_score * 0.2)
        
        return round(sum(scores), 1)
    
    def _generate_cleaning_suggestions(self, profile: Dict) -> List[Dict]:
        """Generate prioritized cleaning suggestions."""
        suggestions = []
        
        for issue in profile["issues"]:
            priority = 1 if issue["severity"] == "critical" else 2 if issue["severity"] == "warning" else 3
            
            suggestions.append({
                "priority": priority,
                "column": issue["column"],
                "issue_type": issue["type"],
                "description": issue["description"],
                "recommended_action": issue["fix_options"][0] if issue.get("fix_options") else None,
                "all_options": issue.get("fix_options", [])
            })
        
        # Add duplicate suggestion if needed
        if profile["duplicates"]["count"] > 0:
            suggestions.append({
                "priority": 2,
                "column": "all",
                "issue_type": "duplicates",
                "description": f"{profile['duplicates']['count']} duplicate rows found",
                "recommended_action": {"method": "drop_duplicates", "description": "Remove duplicate rows"},
                "all_options": [
                    {"method": "drop_duplicates", "description": "Remove all duplicate rows", "formula": "df.drop_duplicates()"},
                    {"method": "keep_first", "description": "Keep first occurrence", "formula": "df.drop_duplicates(keep='first')"},
                    {"method": "keep_last", "description": "Keep last occurrence", "formula": "df.drop_duplicates(keep='last')"},
                    {"method": "keep", "description": "Keep duplicates (no action)"}
                ]
            })
        
        return sorted(suggestions, key=lambda x: x["priority"])


# Singleton
profiler = DataProfiler()

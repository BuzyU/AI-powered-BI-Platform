# Data Cleaning Service - Apply transformations with before/after preview
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import copy

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Handles data cleaning operations with before/after preview.
    All transformations are reversible and show the formula used.
    """
    
    def preview_cleaning(
        self, 
        df: pd.DataFrame, 
        operations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Preview what changes would be made without applying them.
        Returns before/after comparison.
        """
        original_df = df.copy()
        cleaned_df = df.copy()
        
        changes = []
        
        for op in operations:
            op_type = op.get("type")
            column = op.get("column")
            method = op.get("method")
            
            before_stats = self._get_column_stats(cleaned_df, column) if column and column in cleaned_df.columns else None
            
            # Apply operation
            cleaned_df, change_info = self._apply_operation(cleaned_df, op)
            
            after_stats = self._get_column_stats(cleaned_df, column) if column and column in cleaned_df.columns else None
            
            changes.append({
                "operation": op_type,
                "column": column,
                "method": method,
                "formula": change_info.get("formula"),
                "rows_affected": change_info.get("rows_affected", 0),
                "before": before_stats,
                "after": after_stats,
                "sample_before": change_info.get("sample_before"),
                "sample_after": change_info.get("sample_after")
            })
        
        return {
            "original_shape": {"rows": len(original_df), "columns": len(original_df.columns)},
            "new_shape": {"rows": len(cleaned_df), "columns": len(cleaned_df.columns)},
            "rows_removed": len(original_df) - len(cleaned_df),
            "changes": changes,
            "preview_data": cleaned_df.head(20).to_dict(orient='records')
        }
    
    def apply_cleaning(
        self, 
        df: pd.DataFrame, 
        operations: List[Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply cleaning operations and return cleaned dataframe.
        """
        cleaned_df = df.copy()
        applied_operations = []
        
        for op in operations:
            cleaned_df, change_info = self._apply_operation(cleaned_df, op)
            applied_operations.append({
                "operation": op.get("type"),
                "column": op.get("column"),
                "method": op.get("method"),
                "formula": change_info.get("formula"),
                "rows_affected": change_info.get("rows_affected", 0)
            })
        
        return cleaned_df, {
            "operations_applied": len(applied_operations),
            "details": applied_operations,
            "final_shape": {"rows": len(cleaned_df), "columns": len(cleaned_df.columns)}
        }
    
    def _apply_operation(
        self, 
        df: pd.DataFrame, 
        operation: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply a single cleaning operation."""
        
        op_type = str(operation.get("type", "")).lower().replace(" ", "_")
        column = operation.get("column")
        method = operation.get("method")
        custom_value = operation.get("custom_value")
        
        change_info = {"formula": "", "rows_affected": 0}
        
        if op_type == "missing_values":
            df, change_info = self._handle_missing(df, column, method, custom_value)
        
        elif op_type == "outliers":
            df, change_info = self._handle_outliers(df, column, method)
            
        elif op_type == "skewed_distribution" or op_type == "custom_formula":
             if operation.get("formula"):
                 df, change_info = self._apply_formula(df, column, operation.get("formula"))
        
        elif op_type == "duplicates":
            df, change_info = self._handle_duplicates(df, method)
        
        elif op_type == "rename":
            new_name = operation.get("new_name")
            df, change_info = self._rename_column(df, column, new_name)
        
        elif op_type == "drop_column":
            df, change_info = self._drop_column(df, column)
        
        elif op_type == "convert_type":
            target_type = operation.get("target_type")
            df, change_info = self._convert_type(df, column, target_type)
        

        
        return df, change_info
    
    def _handle_missing(
        self, 
        df: pd.DataFrame, 
        column: str, 
        method: str,
        custom_value: Any = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """Handle missing values."""
        
        if column not in df.columns:
            return df, {"formula": "Column not found", "rows_affected": 0}
        
        sample_before = df[df[column].isnull()].head(3).to_dict(orient='records')
        missing_count = df[column].isnull().sum()
        
        if method == "drop":
            df = df.dropna(subset=[column])
            formula = f"df.dropna(subset=['{column}'])"
            
        elif method == "fill_mean":
            mean_val = df[column].mean()
            df[column] = df[column].fillna(mean_val)
            formula = f"df['{column}'].fillna({mean_val:.4f})"
            
        elif method == "fill_median":
            median_val = df[column].median()
            df[column] = df[column].fillna(median_val)
            formula = f"df['{column}'].fillna({median_val:.4f})"
            
        elif method == "fill_mode":
            mode_val = df[column].mode().iloc[0] if len(df[column].mode()) > 0 else ""
            df[column] = df[column].fillna(mode_val)
            formula = f"df['{column}'].fillna('{mode_val}')"
            
        elif method == "fill_zero":
            df[column] = df[column].fillna(0)
            formula = f"df['{column}'].fillna(0)"
            
        elif method == "fill_unknown":
            df[column] = df[column].fillna("Unknown")
            formula = f"df['{column}'].fillna('Unknown')"
            
        elif method == "fill_custom":
            df[column] = df[column].fillna(custom_value)
            formula = f"df['{column}'].fillna({custom_value})"
            
        elif method == "interpolate":
            df[column] = df[column].interpolate(method='linear')
            formula = f"df['{column}'].interpolate(method='linear')"
            
        elif method == "ffill":
            df[column] = df[column].ffill()
            formula = f"df['{column}'].ffill()"
            
        elif method == "bfill":
            df[column] = df[column].bfill()
            formula = f"df['{column}'].bfill()"
            
        else:  # keep
            formula = "No changes (kept as-is)"
        
        sample_after = df.head(3).to_dict(orient='records')
        
        return df, {
            "formula": formula,
            "rows_affected": int(missing_count),
            "sample_before": sample_before,
            "sample_after": sample_after
        }
    
    def _handle_outliers(
        self, 
        df: pd.DataFrame, 
        column: str, 
        method: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """Handle outliers."""
        
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            return df, {"formula": "Column not numeric", "rows_affected": 0}
        
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        outlier_mask = (df[column] < lower) | (df[column] > upper)
        outlier_count = outlier_mask.sum()
        
        sample_before = df[outlier_mask].head(3).to_dict(orient='records')
        
        if method == "remove":
            df = df[~outlier_mask]
            formula = f"df = df[(df['{column}'] >= {lower:.4f}) & (df['{column}'] <= {upper:.4f})]"
            
        elif method == "cap_iqr":
            df[column] = df[column].clip(lower=lower, upper=upper)
            formula = f"df['{column}'].clip(lower={lower:.4f}, upper={upper:.4f})"
            
        elif method == "cap_percentile":
            p1 = df[column].quantile(0.01)
            p99 = df[column].quantile(0.99)
            df[column] = df[column].clip(lower=p1, upper=p99)
            formula = f"df['{column}'].clip(lower={p1:.4f}, upper={p99:.4f})"
            
        elif method == "replace_mean":
            mean_val = df[~outlier_mask][column].mean()
            df.loc[outlier_mask, column] = mean_val
            formula = f"df.loc[outliers, '{column}'] = {mean_val:.4f}  # mean of non-outliers"
            
        elif method == "replace_median":
            median_val = df[~outlier_mask][column].median()
            df.loc[outlier_mask, column] = median_val
            formula = f"df.loc[outliers, '{column}'] = {median_val:.4f}  # median of non-outliers"
            
        else:  # keep
            formula = "No changes (kept as-is)"
        
        sample_after = df.head(3).to_dict(orient='records')
        
        return df, {
            "formula": formula,
            "rows_affected": int(outlier_count),
            "sample_before": sample_before,
            "sample_after": sample_after
        }
    
    def _handle_duplicates(
        self, 
        df: pd.DataFrame, 
        method: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """Handle duplicate rows."""
        
        dup_count = df.duplicated().sum()
        sample_before = df[df.duplicated()].head(3).to_dict(orient='records')
        
        if method == "drop_duplicates":
            df = df.drop_duplicates()
            formula = "df.drop_duplicates()"
            
        elif method == "keep_first":
            df = df.drop_duplicates(keep='first')
            formula = "df.drop_duplicates(keep='first')"
            
        elif method == "keep_last":
            df = df.drop_duplicates(keep='last')
            formula = "df.drop_duplicates(keep='last')"
            
        else:  # keep
            formula = "No changes (kept as-is)"
        
        return df, {
            "formula": formula,
            "rows_affected": int(dup_count),
            "sample_before": sample_before,
            "sample_after": df.head(3).to_dict(orient='records')
        }
    
    def _rename_column(
        self, 
        df: pd.DataFrame, 
        old_name: str, 
        new_name: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """Rename a column."""
        
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
            formula = f"df.rename(columns={{'{old_name}': '{new_name}'}})"
            return df, {"formula": formula, "rows_affected": len(df)}
        
        return df, {"formula": "Column not found", "rows_affected": 0}
    
    def _drop_column(
        self, 
        df: pd.DataFrame, 
        column: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """Drop a column."""
        
        if column in df.columns:
            df = df.drop(columns=[column])
            formula = f"df.drop(columns=['{column}'])"
            return df, {"formula": formula, "rows_affected": len(df)}
        
        return df, {"formula": "Column not found", "rows_affected": 0}
    
    def _convert_type(
        self, 
        df: pd.DataFrame, 
        column: str, 
        target_type: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """Convert column to different type."""
        
        if column not in df.columns:
            return df, {"formula": "Column not found", "rows_affected": 0}
        
        try:
            if target_type == "int":
                df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                formula = f"df['{column}'] = pd.to_numeric(df['{column}']).astype('Int64')"
                
            elif target_type == "float":
                df[column] = pd.to_numeric(df[column], errors='coerce')
                formula = f"df['{column}'] = pd.to_numeric(df['{column}'])"
                
            elif target_type == "string":
                df[column] = df[column].astype(str)
                formula = f"df['{column}'] = df['{column}'].astype(str)"
                
            elif target_type == "datetime":
                df[column] = pd.to_datetime(df[column], errors='coerce')
                formula = f"df['{column}'] = pd.to_datetime(df['{column}'])"
                
            elif target_type == "category":
                df[column] = df[column].astype('category')
                formula = f"df['{column}'] = df['{column}'].astype('category')"
                
            else:
                formula = f"Unknown type: {target_type}"
                
            return df, {"formula": formula, "rows_affected": len(df)}
            
        except Exception as e:
            return df, {"formula": f"Error: {str(e)}", "rows_affected": 0}
    
    def _apply_formula(
        self, 
        df: pd.DataFrame, 
        column: str, 
        formula: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """Apply a custom formula to a column."""
        
        try:
            sample_before = df.head(3).to_dict(orient='records')

            # Create a safe execution environment
            local_vars = {"df": df, "pd": pd, "np": np}
            
            # Execute the formula
            exec(f"result = {formula}", {"__builtins__": {}}, local_vars)
            
            if column:
                df[column] = local_vars.get("result", df[column])
            
            sample_after = df.head(3).to_dict(orient='records')

            return df, {
                "formula": formula, 
                "rows_affected": len(df),
                "sample_before": sample_before,
                "sample_after": sample_after
            }
            
        except Exception as e:
            return df, {"formula": f"Error: {str(e)}", "rows_affected": 0}
    
    def _get_column_stats(self, df: pd.DataFrame, column: str) -> Optional[Dict]:
        """Get basic stats for a column."""
        
        if column not in df.columns:
            return None
        
        series = df[column]
        
        stats = {
            "null_count": int(series.isnull().sum()),
            "unique_count": int(series.nunique()),
            "dtype": str(series.dtype)
        }
        
        if pd.api.types.is_numeric_dtype(series):
            stats.update({
                "mean": round(float(series.mean()), 4) if not pd.isna(series.mean()) else None,
                "median": round(float(series.median()), 4) if not pd.isna(series.median()) else None,
                "min": round(float(series.min()), 4) if not pd.isna(series.min()) else None,
                "max": round(float(series.max()), 4) if not pd.isna(series.max()) else None,
            })
        
        return stats


# Singleton
cleaner = DataCleaner()

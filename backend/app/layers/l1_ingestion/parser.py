# L1: Ingestion Layer - File Parser
import pandas as pd
import json
import chardet
from typing import Dict, Any, List, Optional
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor


# Thread pool for CPU-bound parsing
executor = ThreadPoolExecutor(max_workers=4)


async def get_file_info(file_path: str, file_type: str) -> Dict[str, Any]:
    """Get basic file information without loading entire file."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        _get_file_info_sync,
        file_path,
        file_type
    )


def _get_file_info_sync(file_path: str, file_type: str) -> Dict[str, Any]:
    """Synchronous file info extraction."""
    info = {
        "row_count": 0,
        "column_count": 0,
        "columns": [],
        "sheets": None
    }
    
    if file_type == "csv":
        # Detect encoding
        with open(file_path, 'rb') as f:
            raw = f.read(10000)
            result = chardet.detect(raw)
            encoding = result['encoding'] or 'utf-8'
        
        # Read just first few rows for column info
        df = pd.read_csv(file_path, encoding=encoding, nrows=5)
        info["columns"] = df.columns.tolist()
        info["column_count"] = len(df.columns)
        
        # Count rows efficiently
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            info["row_count"] = sum(1 for _ in f) - 1  # Exclude header
    
    elif file_type in ["xls", "xlsx"]:
        xl = pd.ExcelFile(file_path)
        info["sheets"] = xl.sheet_names
        
        # Use first sheet for info
        df = pd.read_excel(file_path, sheet_name=0, nrows=5)
        info["columns"] = df.columns.tolist()
        info["column_count"] = len(df.columns)
        
        # Get row count from first sheet
        df_full = pd.read_excel(file_path, sheet_name=0)
        info["row_count"] = len(df_full)
    
    elif file_type == "json":
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            info["row_count"] = len(data)
            if data:
                info["columns"] = list(data[0].keys()) if isinstance(data[0], dict) else []
                info["column_count"] = len(info["columns"])
        elif isinstance(data, dict):
            # Try to find the array of records
            for key, value in data.items():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    info["row_count"] = len(value)
                    info["columns"] = list(value[0].keys())
                    info["column_count"] = len(info["columns"])
                    break
    
    return info


async def parse_file(
    file_path: str, 
    file_type: str, 
    sheet_name: Optional[str] = None
) -> pd.DataFrame:
    """Parse file and return DataFrame."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        _parse_file_sync,
        file_path,
        file_type,
        sheet_name
    )


def _parse_file_sync(
    file_path: str, 
    file_type: str, 
    sheet_name: Optional[str] = None
) -> pd.DataFrame:
    """Synchronous file parsing."""
    if file_type == "csv":
        # Detect encoding
        with open(file_path, 'rb') as f:
            raw = f.read(10000)
            result = chardet.detect(raw)
            encoding = result['encoding'] or 'utf-8'
        
        return pd.read_csv(file_path, encoding=encoding)
    
    elif file_type in ["xls", "xlsx"]:
        if sheet_name:
            return pd.read_excel(file_path, sheet_name=sheet_name)
        return pd.read_excel(file_path, sheet_name=0)
    
    elif file_type == "json":
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # Try to find the array of records
            for key, value in data.items():
                if isinstance(value, list):
                    return pd.DataFrame(value)
            # Fall back to treating dict as single record
            return pd.DataFrame([data])
    
    raise ValueError(f"Unsupported file type: {file_type}")


def detect_date_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that contain date values."""
    date_columns = []
    
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            date_columns.append(col)
            continue
        
        # Try to parse string columns as dates
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(100)
            if len(sample) == 0:
                continue
            
            try:
                parsed = pd.to_datetime(sample, infer_datetime_format=True, errors='coerce')
                valid_ratio = parsed.notna().sum() / len(sample)
                if valid_ratio > 0.8:
                    date_columns.append(col)
            except:
                pass
    
    return date_columns


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that contain numeric values."""
    numeric_columns = []
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)
            continue
        
        # Try to parse string columns as numbers
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(100)
            if len(sample) == 0:
                continue
            
            try:
                # Remove currency symbols and commas
                cleaned = sample.astype(str).str.replace(r'[$,€£]', '', regex=True)
                parsed = pd.to_numeric(cleaned, errors='coerce')
                valid_ratio = parsed.notna().sum() / len(sample)
                if valid_ratio > 0.8:
                    numeric_columns.append(col)
            except:
                pass
    
    return numeric_columns

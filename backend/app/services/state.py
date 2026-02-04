# State Manager - Stores datasets, profiles, and analysis results
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# In-memory storage (for production, use Redis or database)
_state: Dict[str, Any] = {
    "datasets": {},      # tenant_id -> {dataset_id -> dataset_info}
    "profiles": {},      # tenant_id -> {dataset_id -> profile}
    "analysis": {},      # tenant_id -> analysis_result
    "cleaned": {},       # tenant_id -> {dataset_id -> cleaned_df}
    "chat_history": {}   # tenant_id -> List[messages]
}


def store_dataset(tenant_id: str, dataset_id: str, dataset_info: Dict[str, Any]):
    """Store a dataset."""
    if tenant_id not in _state["datasets"]:
        _state["datasets"][tenant_id] = {}
    
    dataset_info["id"] = dataset_id
    dataset_info["uploaded_at"] = datetime.now().isoformat()
    _state["datasets"][tenant_id][dataset_id] = dataset_info
    
    logger.info(f"Stored dataset {dataset_id} for tenant {tenant_id}")


def get_dataset(tenant_id: str, dataset_id: str) -> Optional[Dict]:
    """Get a dataset by ID."""
    return _state["datasets"].get(tenant_id, {}).get(dataset_id)


def get_all_datasets(tenant_id: str) -> List[Dict]:
    """Get all datasets for a tenant."""
    datasets = _state["datasets"].get(tenant_id, {})
    return list(datasets.values())


def get_dataset_df(tenant_id: str, dataset_id: str) -> Optional[pd.DataFrame]:
    """Get the dataframe for a dataset."""
    dataset = get_dataset(tenant_id, dataset_id)
    if dataset:
        return dataset.get("df")
    return None


def store_profile(tenant_id: str, dataset_id: str, profile: Dict[str, Any]):
    """Store a dataset profile."""
    if tenant_id not in _state["profiles"]:
        _state["profiles"][tenant_id] = {}
    
    _state["profiles"][tenant_id][dataset_id] = profile
    logger.info(f"Stored profile for dataset {dataset_id}")


def get_profile(tenant_id: str, dataset_id: str) -> Optional[Dict]:
    """Get profile for a dataset."""
    return _state["profiles"].get(tenant_id, {}).get(dataset_id)


def get_all_profiles(tenant_id: str) -> List[Dict]:
    """Get all profiles for a tenant."""
    profiles = _state["profiles"].get(tenant_id, {})
    return list(profiles.values())


def store_analysis(tenant_id: str, analysis: Dict[str, Any]):
    """Store analysis results."""
    _state["analysis"][tenant_id] = analysis
    logger.info(f"Stored analysis for tenant {tenant_id}")


def get_analysis(tenant_id: str) -> Optional[Dict]:
    """Get analysis results."""
    return _state["analysis"].get(tenant_id)


def store_cleaned_df(tenant_id: str, dataset_id: str, df: pd.DataFrame):
    """Store a cleaned dataframe."""
    if tenant_id not in _state["cleaned"]:
        _state["cleaned"][tenant_id] = {}
    
    _state["cleaned"][tenant_id][dataset_id] = df
    logger.info(f"Stored cleaned dataset {dataset_id}")


def get_cleaned_df(tenant_id: str, dataset_id: str) -> Optional[pd.DataFrame]:
    """Get a cleaned dataframe."""
    return _state["cleaned"].get(tenant_id, {}).get(dataset_id)


def update_dataset_df(tenant_id: str, dataset_id: str, df: pd.DataFrame):
    """Update the dataframe in a dataset."""
    if tenant_id in _state["datasets"] and dataset_id in _state["datasets"][tenant_id]:
        _state["datasets"][tenant_id][dataset_id]["df"] = df
        _state["datasets"][tenant_id][dataset_id]["row_count"] = len(df)
        logger.info(f"Updated dataset {dataset_id}")


def add_chat_message(tenant_id: str, message: Dict[str, Any]):
    """Add a chat message to history."""
    if tenant_id not in _state["chat_history"]:
        _state["chat_history"][tenant_id] = []
    
    _state["chat_history"][tenant_id].append({
        **message,
        "timestamp": datetime.now().isoformat()
    })


def get_chat_history(tenant_id: str, limit: int = 50) -> List[Dict]:
    """Get chat history."""
    history = _state["chat_history"].get(tenant_id, [])
    return history[-limit:]


def clear_tenant_data(tenant_id: str):
    """Clear all data for a tenant."""
    for key in _state:
        if tenant_id in _state[key]:
            del _state[key][tenant_id]
    logger.info(f"Cleared all data for tenant {tenant_id}")


def get_full_context(tenant_id: str) -> Dict[str, Any]:
    """Get full context for AI (analysis + profiles)."""
    return {
        "analysis": get_analysis(tenant_id),
        "profiles": get_all_profiles(tenant_id),
        "datasets": [
            {
                "id": d["id"],
                "filename": d.get("filename"),
                "row_count": d.get("metadata", {}).get("row_count"),
                "column_count": d.get("metadata", {}).get("column_count")
            }
            for d in get_all_datasets(tenant_id)
        ]
    }

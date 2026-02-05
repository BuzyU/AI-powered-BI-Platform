# Enhanced State Manager - Multi-Session Architecture
# Each session is completely isolated with its own data, analysis, and chat history

from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
import logging
import json
import os
import pickle
import threading
import tempfile
import shutil
from pathlib import Path
from enum import Enum
from contextlib import contextmanager
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Constants
PERSISTENCE_DIR = Path("./brain_storage")
PERSISTENCE_DIR.mkdir(exist_ok=True)
MAX_SESSIONS_IN_MEMORY = 50  # Limit sessions to prevent memory leaks

# Thread safety
_session_lock = threading.RLock()

@contextmanager
def session_lock():
    """Context manager for thread-safe session operations."""
    with _session_lock:
        yield


class SessionStatus(Enum):
    """Session lifecycle status."""
    CREATED = "created"
    UPLOADING = "uploading"
    ANALYZING = "analyzing"
    READY = "ready"
    ERROR = "error"


class PersonaType(Enum):
    """Analysis persona types."""
    BUSINESS_ANALYST = "business_analyst"
    DATA_SCIENTIST = "data_scientist"
    ML_ENGINEER = "ml_engineer"
    LEGAL_ANALYST = "legal_analyst"
    FINANCIAL_ANALYST = "financial_analyst"
    OPERATIONS_ANALYST = "operations_analyst"
    MARKETING_ANALYST = "marketing_analyst"
    GENERAL = "general"


class SessionState:
    """
    Manages state for a single isolated session.
    Each session has its own datasets, profiles, analysis, and chat history.
    """
    
    def __init__(self, session_id: str, name: str = None):
        self.session_id = session_id
        self.name = name or f"Session {datetime.now().strftime('%b %d, %H:%M')}"
        self.status = SessionStatus.CREATED
        self.persona = PersonaType.GENERAL
        
        # Data storage
        self.datasets: Dict[str, Any] = {}
        self.profiles: Dict[str, Any] = {}
        self.classifications: Dict[str, Any] = {}
        
        # Analysis results
        self.analysis: Optional[Dict[str, Any]] = None
        self.linkage_report: Optional[Dict[str, Any]] = None
        self.dashboard_config: Optional[Dict[str, Any]] = None
        self.evaluation_results: Optional[Dict[str, Any]] = None  # Model evaluation
        
        # Chat/Q&A
        self.chat_history: List[Dict[str, Any]] = []
        
        # Timestamps
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.analyzed_at: Optional[str] = None
        
        # In-memory dataframe storage
        self._dfs: Dict[str, pd.DataFrame] = {}
    
    def update_timestamp(self):
        self.updated_at = datetime.now().isoformat()
    
    def set_status(self, status: SessionStatus):
        self.status = status
        self.update_timestamp()
    
    def set_persona(self, persona: PersonaType):
        self.persona = persona
        self.update_timestamp()
    
    def generate_name(self) -> str:
        if not self.datasets:
            return self.name
        
        first_dataset = list(self.datasets.values())[0]
        filename = first_dataset.get('filename', 'Analysis')
        base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
        
        if self.classifications:
            first_class = list(self.classifications.values())[0]
            content_type = first_class.get('content_type', '').replace('_', ' ').title()
            if content_type and content_type != 'Unknown':
                return f"{base_name} - {content_type}"
        
        return f"{base_name} Analysis"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "status": self.status.value,
            "persona": self.persona.value,
            "datasets": self.datasets,
            "profiles": self.profiles,
            "classifications": self.classifications,
            "analysis": self.analysis,
            "linkage_report": self.linkage_report,
            "dashboard_config": self.dashboard_config,
            "evaluation_results": self.evaluation_results,
            "chat_history": self.chat_history,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "analyzed_at": self.analyzed_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SessionState":
        session = cls(data["session_id"], data.get("name"))
        session.status = SessionStatus(data.get("status", "created"))
        session.persona = PersonaType(data.get("persona", "general"))
        session.datasets = data.get("datasets", {})
        session.profiles = data.get("profiles", {})
        session.classifications = data.get("classifications", {})
        session.analysis = data.get("analysis")
        session.linkage_report = data.get("linkage_report")
        session.dashboard_config = data.get("dashboard_config")
        session.chat_history = data.get("chat_history", [])
        session.created_at = data.get("created_at", datetime.now().isoformat())
        session.updated_at = data.get("updated_at", datetime.now().isoformat())
        session.analyzed_at = data.get("analyzed_at")
        return session


# Global Registry - Use OrderedDict for LRU-style eviction
_sessions: OrderedDict[str, SessionState] = OrderedDict()


def _evict_old_sessions():
    """Evict oldest sessions if over memory limit."""
    while len(_sessions) > MAX_SESSIONS_IN_MEMORY:
        oldest_id, oldest = _sessions.popitem(last=False)
        oldest._dfs.clear()  # Free DataFrame memory
        logger.info(f"Evicted old session from memory: {oldest_id}")


def get_session(session_id: str) -> SessionState:
    """Get or create a session. Thread-safe."""
    with session_lock():
        if session_id in _sessions:
            _sessions.move_to_end(session_id)  # Mark as recently used
            return _sessions[session_id]
        
        if _load_session_from_disk(session_id):
            _evict_old_sessions()
            return _sessions[session_id]
        
        _sessions[session_id] = SessionState(session_id)
        _evict_old_sessions()
        logger.info(f"Created new session: {session_id}")
        return _sessions[session_id]


def get_existing_session(session_id: str) -> Optional[SessionState]:
    """Get a session only if it exists (no auto-creation). Thread-safe."""
    with session_lock():
        if session_id in _sessions:
            _sessions.move_to_end(session_id)
            return _sessions[session_id]
        
        if _load_session_from_disk(session_id):
            return _sessions[session_id]
        
        return None


def create_session(session_id: str, name: str = None) -> SessionState:
    """Explicitly create a new session. Thread-safe."""
    with session_lock():
        session = SessionState(session_id, name)
        _sessions[session_id] = session
        _evict_old_sessions()
        _persist_session_metadata(session)
        logger.info(f"Created session: {session_id} ({name})")
        return session


def list_sessions() -> List[Dict]:
    """List all sessions. Thread-safe."""
    with session_lock():
        _scan_disk_for_sessions()
        sessions = []
        for s in list(_sessions.values()):  # Copy to avoid iteration issues
            sessions.append({
                "id": s.session_id,
                "name": s.name,
                "status": s.status.value,
                "persona": s.persona.value,
                "datasets_count": len(s.datasets),
                "has_analysis": s.analysis is not None,
                "chat_count": len(s.chat_history),
                "created_at": s.created_at,
                "updated_at": s.updated_at,
                "analyzed_at": s.analyzed_at
            })
    sessions.sort(key=lambda x: x["updated_at"], reverse=True)
    return sessions


def clear_session(session_id: str):
    """Clear a session's data. Thread-safe."""
    with session_lock():
        if session_id in _sessions:
            session = _sessions[session_id]
            dataset_ids = list(session.datasets.keys())  # Copy to avoid iteration issues
            
            for dataset_id in dataset_ids:
                df_file = _df_path(session_id, dataset_id)
                if df_file.exists():
                    try:
                        df_file.unlink()
                    except Exception as e:
                        logger.error(f"Failed to delete df file: {e}")
            
            meta_file = _idx_path(session_id)
            if meta_file.exists():
                try:
                    meta_file.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete session file: {e}")
            
            del _sessions[session_id]
            logger.info(f"Cleared session: {session_id}")


def delete_session(session_id: str) -> bool:
    clear_session(session_id)
    return True


def _idx_path(session_id: str) -> Path:
    return PERSISTENCE_DIR / f"{session_id}.json"


def _df_path(session_id: str, dataset_id: str) -> Path:
    return PERSISTENCE_DIR / f"{session_id}_{dataset_id}.pkl"


def _persist_session_metadata(session: SessionState):
    """Persist session metadata atomically using temp file + rename."""
    target_path = _idx_path(session.session_id)
    try:
        # Write to temp file in same directory, then rename (atomic on same filesystem)
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=PERSISTENCE_DIR,
            suffix='.tmp',
            delete=False
        ) as f:
            json.dump(session.to_dict(), f, indent=2, default=str)
            temp_path = f.name
        # Atomic rename (works on Windows too with replace)
        shutil.move(temp_path, target_path)
    except Exception as e:
        logger.error(f"Failed to persist session {session.session_id}: {e}")
        # Clean up temp file if it exists
        if 'temp_path' in locals() and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except:
                pass


def _persist_dataframe(session_id: str, dataset_id: str, df: pd.DataFrame):
    """Persist DataFrame atomically using temp file + rename."""
    target_path = _df_path(session_id, dataset_id)
    try:
        with tempfile.NamedTemporaryFile(
            mode='wb',
            dir=PERSISTENCE_DIR,
            suffix='.tmp',
            delete=False
        ) as f:
            df.to_pickle(f.name)
            temp_path = f.name
        shutil.move(temp_path, target_path)
    except Exception as e:
        logger.error(f"Failed to persist dataframe {dataset_id}: {e}")
        if 'temp_path' in locals() and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except:
                pass


def _load_dataframe(session_id: str, dataset_id: str) -> Optional[pd.DataFrame]:
    path = _df_path(session_id, dataset_id)
    if path.exists():
        try:
            return pd.read_pickle(path)
        except Exception as e:
            logger.error(f"Failed to load dataframe {dataset_id}, file may be corrupted: {e}")
            # Optionally remove corrupted file
            try:
                path.unlink()
                logger.warning(f"Removed corrupted dataframe file: {path}")
            except:
                pass
    return None


def _load_session_from_disk(session_id: str) -> bool:
    """Load session from disk. Called inside session_lock context."""
    path = _idx_path(session_id)
    if path.exists():
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            session = SessionState.from_dict(data)
            _sessions[session_id] = session
            _sessions.move_to_end(session_id)  # Mark as recently used
            return True
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted session JSON {session_id}: {e}")
            # Rename corrupted file for debugging
            try:
                corrupted_path = path.with_suffix('.corrupted')
                shutil.move(path, corrupted_path)
                logger.warning(f"Renamed corrupted session file to: {corrupted_path}")
            except:
                pass
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
    return False


def _scan_disk_for_sessions():
    for file in PERSISTENCE_DIR.glob("*.json"):
        session_id = file.stem
        if session_id not in _sessions and "_" not in session_id:
            _load_session_from_disk(session_id)


def store_dataset(session_id: str, dataset_id: str, dataset_info: Dict[str, Any]):
    session = get_session(session_id)
    df = dataset_info.pop("df", None)
    if df is not None:
        session._dfs[dataset_id] = df
        _persist_dataframe(session_id, dataset_id, df)
    
    dataset_info["id"] = dataset_id
    dataset_info["uploaded_at"] = datetime.now().isoformat()
    session.datasets[dataset_id] = dataset_info
    
    # Clear cached dashboard/analysis when datasets change
    session.dashboard_config = None
    session.analysis = None
    
    session.update_timestamp()
    session.name = session.generate_name()
    _persist_session_metadata(session)
    logger.info(f"Stored dataset {dataset_id} in session {session_id}")


def get_dataset(session_id: str, dataset_id: str) -> Optional[Dict]:
    session = get_session(session_id)
    return session.datasets.get(dataset_id)


def get_all_datasets(session_id: str) -> List[Dict]:
    session = get_session(session_id)
    return list(session.datasets.values())


def get_dataset_df(session_id: str, dataset_id: str) -> Optional[pd.DataFrame]:
    session = get_session(session_id)
    if dataset_id in session._dfs:
        return session._dfs[dataset_id]
    df = _load_dataframe(session_id, dataset_id)
    if df is not None:
        session._dfs[dataset_id] = df
        return df
    return None


def get_all_dataframes(session_id: str) -> Dict[str, pd.DataFrame]:
    session = get_session(session_id)
    result = {}
    for dataset_id in session.datasets.keys():
        df = get_dataset_df(session_id, dataset_id)
        if df is not None:
            result[dataset_id] = df
    return result


def update_dataset_df(session_id: str, dataset_id: str, df: pd.DataFrame):
    session = get_session(session_id)
    session._dfs[dataset_id] = df
    _persist_dataframe(session_id, dataset_id, df)
    # Clear cached dashboard/analysis when data changes
    session.dashboard_config = None
    session.analysis = None
    session.update_timestamp()
    _persist_session_metadata(session)


def delete_dataset(session_id: str, dataset_id: str) -> bool:
    session = get_session(session_id)
    if dataset_id not in session.datasets:
        return False
    
    del session.datasets[dataset_id]
    if dataset_id in session.profiles:
        del session.profiles[dataset_id]
    if dataset_id in session.classifications:
        del session.classifications[dataset_id]
    if dataset_id in session._dfs:
        del session._dfs[dataset_id]
    
    df_file = _df_path(session_id, dataset_id)
    if df_file.exists():
        try:
            df_file.unlink()
        except Exception as e:
            logger.error(f"Failed to delete dataframe file: {e}")
    
    # Clear cached dashboard/analysis when datasets change
    session.dashboard_config = None
    session.analysis = None
    
    session.update_timestamp()
    _persist_session_metadata(session)
    logger.info(f"Deleted dataset {dataset_id} from session {session_id}")
    return True


def store_classification(session_id: str, dataset_id: str, classification: Dict[str, Any]):
    session = get_session(session_id)
    session.classifications[dataset_id] = classification
    _update_session_persona(session)
    session.update_timestamp()
    _persist_session_metadata(session)


def get_classification(session_id: str, dataset_id: str) -> Optional[Dict]:
    session = get_session(session_id)
    return session.classifications.get(dataset_id)


def get_all_classifications(session_id: str) -> Dict[str, Any]:
    session = get_session(session_id)
    return session.classifications


def _update_session_persona(session: SessionState):
    if not session.classifications:
        return
    categories = {}
    for classification in session.classifications.values():
        cat = classification.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    if not categories:
        return
    dominant = max(categories.items(), key=lambda x: x[1])[0]
    persona_mapping = {
        "business": PersonaType.BUSINESS_ANALYST,
        "ml_model": PersonaType.ML_ENGINEER,
        "analytics": PersonaType.DATA_SCIENTIST,
        "legal": PersonaType.LEGAL_ANALYST,
        "healthcare": PersonaType.OPERATIONS_ANALYST,
    }
    session.persona = persona_mapping.get(dominant, PersonaType.GENERAL)


def store_profile(session_id: str, dataset_id: str, profile: Dict[str, Any]):
    session = get_session(session_id)
    session.profiles[dataset_id] = profile
    session.update_timestamp()
    _persist_session_metadata(session)


def get_profile(session_id: str, dataset_id: str) -> Optional[Dict]:
    session = get_session(session_id)
    return session.profiles.get(dataset_id)


def get_all_profiles(session_id: str) -> List[Dict]:
    session = get_session(session_id)
    return list(session.profiles.values())


def store_analysis(session_id: str, analysis: Dict[str, Any]):
    session = get_session(session_id)
    session.analysis = analysis
    session.analyzed_at = datetime.now().isoformat()
    session.status = SessionStatus.READY
    session.update_timestamp()
    _persist_session_metadata(session)


def get_analysis(session_id: str) -> Optional[Dict]:
    session = get_session(session_id)
    return session.analysis


def store_linkage_report(session_id: str, report: Dict[str, Any]):
    session = get_session(session_id)
    session.linkage_report = report
    session.update_timestamp()
    _persist_session_metadata(session)


def get_linkage_report(session_id: str) -> Optional[Dict]:
    session = get_session(session_id)
    return session.linkage_report


def store_dashboard_config(session_id: str, config: Dict[str, Any]):
    session = get_session(session_id)
    session.dashboard_config = config
    session.update_timestamp()
    _persist_session_metadata(session)


def get_dashboard_config(session_id: str) -> Optional[Dict]:
    session = get_session(session_id)
    return session.dashboard_config


def clear_dashboard_config(session_id: str):
    """Clear the cached dashboard config to force regeneration."""
    session = get_session(session_id)
    session.dashboard_config = None
    session.update_timestamp()
    _persist_session_metadata(session)


def add_chat_message(session_id: str, role: str, content: str, metadata: Dict = None):
    session = get_session(session_id)
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    session.chat_history.append(message)
    session.update_timestamp()
    _persist_session_metadata(session)
    return message


def get_chat_history(session_id: str) -> List[Dict]:
    session = get_session(session_id)
    return session.chat_history


def clear_chat_history(session_id: str):
    session = get_session(session_id)
    session.chat_history = []
    session.update_timestamp()
    _persist_session_metadata(session)


def get_session_info(session_id: str) -> Dict[str, Any]:
    session = get_session(session_id)
    return {
        "session_id": session.session_id,
        "name": session.name,
        "status": session.status.value,
        "persona": session.persona.value,
        "datasets": list(session.datasets.values()),
        "classifications": session.classifications,
        "has_analysis": session.analysis is not None,
        "has_linkage": session.linkage_report is not None,
        "chat_count": len(session.chat_history),
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "analyzed_at": session.analyzed_at
    }


def update_session_name(session_id: str, name: str):
    session = get_session(session_id)
    session.name = name
    session.update_timestamp()
    _persist_session_metadata(session)


def update_session_status(session_id: str, status: SessionStatus):
    session = get_session(session_id)
    session.set_status(status)
    _persist_session_metadata(session)

def _cleanup_memory_cache():
    """
    Clean up in-memory DataFrame cache to free memory.
    Data is still persisted on disk and will be reloaded on access.
    """
    import gc
    
    # Clear in-memory DataFrames from inactive sessions
    for session in _sessions.values():
        if session._dfs:
            session._dfs.clear()
    
    # Force garbage collection
    gc.collect()
    logger.info("Memory cache cleaned up")


def get_memory_usage() -> Dict[str, Any]:
    """Get current memory usage statistics."""
    import sys
    
    total_dfs = sum(len(s._dfs) for s in _sessions.values())
    total_sessions = len(_sessions)
    
    # Estimate memory usage
    memory_estimate = 0
    for session in _sessions.values():
        for df in session._dfs.values():
            if df is not None:
                memory_estimate += df.memory_usage(deep=True).sum()
    
    return {
        "total_sessions": total_sessions,
        "total_dataframes_in_memory": total_dfs,
        "estimated_memory_bytes": memory_estimate,
        "estimated_memory_mb": round(memory_estimate / (1024 * 1024), 2)
    }


# ============== MODEL EVALUATION STATE ==============

def store_evaluation(session_id: str, evaluation: Dict[str, Any]):
    """Store model evaluation results in session."""
    session = get_session(session_id)
    session.evaluation_results = evaluation
    session.update_timestamp()
    _persist_session_metadata(session)
    
    # Also clear dashboard config to trigger regeneration with new data
    session.dashboard_config = None


def get_evaluation(session_id: str) -> Optional[Dict[str, Any]]:
    """Get model evaluation results for a session."""
    session = get_session(session_id)
    return session.evaluation_results


def clear_evaluation(session_id: str):
    """Clear evaluation results from session."""
    session = get_session(session_id)
    session.evaluation_results = None
    session.update_timestamp()
    _persist_session_metadata(session)
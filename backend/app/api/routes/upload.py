# Complete API Routes - Upload, Profile, Clean, Analyze, Ask
# Refactored for Session Isolation & Multi-Persona Support
from fastapi import APIRouter, UploadFile, File, HTTPException, Body, Header
from typing import List, Dict, Any, Optional
import aiofiles
from pathlib import Path
from uuid import uuid4
import os
import pandas as pd
import logging
import json
from pydantic import BaseModel

from app.config import settings
from app.services.analyzer import DataAnalyzer
from app.services.profiler import DataProfiler
from app.services.cleaner import DataCleaner
from app.services.groq_ai import create_groq_ai
from app.services import state
from app.layers.l2_classification.content_classifier import SmartContentClassifier
from app.layers.l5_relationship.linkage_validator import LinkageValidator

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
analyzer = DataAnalyzer()
profiler = DataProfiler()
cleaner = DataCleaner()
classifier = SmartContentClassifier()
linkage_validator = LinkageValidator()

# Groq API key
GROQ_API_KEY = settings.GROQ_API_KEY

# Ensure upload directory exists
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx', 'json', 'pt', 'onnx', 'h5', 'pkl'}


def get_file_extension(filename: str) -> str:
    return filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''


# ============== SESSION ROUTES ==============

@router.post("/sessions")
async def create_session():
    """Create a new analysis session."""
    session_id = str(uuid4())
    state.get_session(session_id) # Initializes it
    return {"session_id": session_id}

@router.get("/sessions")
async def list_sessions():
    """List active sessions."""
    return {"sessions": state.list_sessions()}

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    state.clear_session(session_id)
    return {"success": True}


@router.get("/sessions/{session_id}/info")
async def get_session_info(session_id: str):
    """Get detailed info about a session."""
    return state.get_session_info(session_id)


@router.put("/sessions/{session_id}/name")
async def update_session_name_route(session_id: str, body: Dict[str, Any] = Body(...)):
    """Update session name."""
    name = body.get("name", "")
    state.update_session_name(session_id, name)
    return {"success": True}


# ============== UPLOAD ROUTES ==============

@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    x_session_id: Optional[str] = Header(None)
):
    """Upload files to a session and detect content type using Smart Classification."""
    # Default session if none provided (backward compatibility)
    session_id = x_session_id or "default_session"
    
    # Ensure session directory
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    uploaded_datasets = []  # For linkage check
    
    for file in files:
        ext = get_file_extension(file.filename)
        
        if ext not in ALLOWED_EXTENSIONS:
            results.append({"filename": file.filename, "status": "error", "error": f"File type .{ext} not allowed"})
            continue
        
        dataset_id = str(uuid4())
        file_path = session_dir / f"{dataset_id}.{ext}"
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        try:
            # 1. Load & Understand Content
            df, metadata = await analyzer.load_file(str(file_path), ext)
            
            # 2. Run Smart Classification (for data files)
            classification_result = None
            if not metadata.get('is_model') and df is not None:
                classification_result = classifier.classify(df, file.filename)
                
                # Store classification in session state
                state.store_classification(session_id, dataset_id, classification_result)
                
                # Merge classification into metadata
                metadata['detected_role'] = classification_result.get('primary_role', 'Unknown')
                metadata['detected_type'] = classification_result.get('category', 'Unknown')
                metadata['confidence'] = classification_result.get('confidence', 0)
                metadata['all_roles'] = classification_result.get('all_roles', [])
                metadata['content_summary'] = classification_result.get('summary', '')
            
            # 3. Generate Profile (for Data files)
            profile = {}
            if not metadata.get('is_model') and df is not None:
                profile = profiler.profile_dataset(df, file.filename)
                state.store_profile(session_id, dataset_id, profile)
            
            # 4. Store in Session State
            dataset_info = {
                'id': dataset_id,
                'filename': file.filename,
                'file_path': str(file_path),
                'file_type': ext,
                'metadata': metadata,
                'df': df,
                'status': 'analyzed',
                'classification': classification_result
            }
            state.store_dataset(session_id, dataset_id, dataset_info)
            
            # Track for linkage
            uploaded_datasets.append({
                'id': dataset_id,
                'filename': file.filename,
                'df': df,
                'is_model': metadata.get('is_model', False)
            })
            
            # Prepared result
            result_entry = {
                'id': dataset_id,
                'filename': file.filename,
                'file_type': ext,
                'detected_role': metadata.get('detected_role', 'Unknown'),
                'detected_type': metadata.get('detected_type', 'Unknown'),
                'confidence': metadata.get('confidence', 0),
                'all_roles': metadata.get('all_roles', []),
                'content_summary': metadata.get('content_summary', ''),
                'is_model': metadata.get('is_model', False),
                'status': 'analyzed'
            }
            
            if not metadata.get('is_model'):
                result_entry.update({
                    'shape': profile.get('shape'),
                    'qa_score': profile.get('overall_quality', 0),
                    'issues_count': len(profile.get('issues', []))
                })
                
            results.append(result_entry)
            
        except Exception as e:
            logger.error(f"Failed to verify {file.filename}: {e}")
            results.append({
                'filename': file.filename,
                'status': 'error',
                'error': str(e)
            })
    
    # 5. Linkage Check - Validate relationships between datasets
    linkage_report = None
    data_datasets = [d for d in uploaded_datasets if not d.get('is_model') and d.get('df') is not None]
    
    if len(data_datasets) > 1:
        # Multiple data files - check linkage
        linkage_report = linkage_validator.validate_linkage(data_datasets)
        state.store_linkage_report(session_id, linkage_report)
    elif len(data_datasets) == 1:
        # Check against existing session datasets
        existing = state.get_all_datasets(session_id)
        existing_data = []
        for ex in existing:
            if ex['id'] != data_datasets[0]['id'] and not ex.get('metadata', {}).get('is_model'):
                ex_df = state.get_dataset_df(session_id, ex['id'])
                if ex_df is not None:
                    existing_data.append({
                        'id': ex['id'],
                        'filename': ex['filename'],
                        'df': ex_df
                    })
        
        if existing_data:
            all_data = data_datasets + existing_data
            linkage_report = linkage_validator.validate_linkage(all_data)
            state.store_linkage_report(session_id, linkage_report)
    
    response = {'session_id': session_id, 'datasets': results}
    
    if linkage_report:
        response['linkage'] = {
            'health_score': linkage_report.get('overall_health', 0),
            'relationships': len(linkage_report.get('relationships', [])),
            'warnings': linkage_report.get('warnings', [])
        }
    
    return response


# ============== DATASET ROUTES ==============

@router.get("/datasets")
async def list_datasets(x_session_id: str = Header("default_session")):
    """List datasets in session."""
    datasets = state.get_all_datasets(x_session_id)
    return {'datasets': datasets, 'session_id': x_session_id}

@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str, x_session_id: str = Header("default_session")):
    """Get dataset details."""
    dataset = state.get_dataset(x_session_id, dataset_id)
    if not dataset: raise HTTPException(404, "Dataset not found")
    return dataset

@router.get("/datasets/{dataset_id}/classification")
async def get_dataset_classification(dataset_id: str, x_session_id: str = Header("default_session")):
    """Get smart classification result for a dataset."""
    classification = state.get_classification(x_session_id, dataset_id)
    if not classification:
        raise HTTPException(404, "Classification not found")
    return classification

@router.get("/classifications")
async def get_all_classifications(x_session_id: str = Header("default_session")):
    """Get all classifications in session."""
    classifications = state.get_all_classifications(x_session_id)
    return {'classifications': classifications}

@router.get("/linkage")
async def get_linkage_report(x_session_id: str = Header("default_session")):
    """Get linkage validation report for session datasets."""
    report = state.get_linkage_report(x_session_id)
    if not report:
        # Generate fresh report
        datasets = state.get_all_datasets(x_session_id)
        data_for_linkage = []
        for d in datasets:
            if not d.get('metadata', {}).get('is_model'):
                df = state.get_dataset_df(x_session_id, d['id'])
                if df is not None:
                    data_for_linkage.append({
                        'id': d['id'],
                        'filename': d['filename'],
                        'df': df
                    })
        
        if len(data_for_linkage) > 1:
            report = linkage_validator.validate_linkage(data_for_linkage)
            state.store_linkage_report(x_session_id, report)
        else:
            return {'message': 'Need at least 2 data files for linkage validation'}
    
    return report

@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str, x_session_id: str = Header("default_session")):
    """Delete a dataset from the session."""
    try:
        # Remove from state
        success = state.delete_dataset(x_session_id, dataset_id)
        if not success:
            raise HTTPException(404, "Dataset not found")
        
        # Try to delete the file as well
        session_dir = UPLOAD_DIR / x_session_id
        for ext in ALLOWED_EXTENSIONS:
            file_path = session_dir / f"{dataset_id}.{ext}"
            if file_path.exists():
                file_path.unlink()
                break
        
        return {"success": True, "message": "Dataset deleted"}
    except Exception as e:
        logger.error(f"Error deleting dataset: {e}")
        raise HTTPException(500, f"Failed to delete dataset: {str(e)}")

# ============== PROFILE & CLEANING ==============
# Updated to use x_session_id

@router.get("/datasets/{dataset_id}/profile")
async def get_profile(dataset_id: str, x_session_id: str = Header("default_session")):
    profile = state.get_profile(x_session_id, dataset_id)
    if not profile: raise HTTPException(404, "Profile not found")
    return profile

@router.post("/datasets/{dataset_id}/clean/apply")
async def apply_cleaning(
    dataset_id: str, 
    operations: List[Dict[str, Any]] = Body(...),
    x_session_id: str = Header("default_session")
):
    df = state.get_dataset_df(x_session_id, dataset_id)
    if df is None: raise HTTPException(404, "Dataset not found")
    
    cleaned_df, result = cleaner.apply_cleaning(df, operations)
    
    # Update state
    state.update_dataset_df(x_session_id, dataset_id, cleaned_df)
    
    # New Profile
    dataset = state.get_dataset(x_session_id, dataset_id)
    new_profile = profiler.profile_dataset(cleaned_df, dataset.get('filename'))
    state.store_profile(x_session_id, dataset_id, new_profile)
    
    return {'success': True, 'result': result}

# ============== ANALYSIS & CHAT ==============

class AnalyzeRequest(BaseModel):
    filters: Optional[Dict[str, Any]] = None

@router.post("/analyze")
async def run_analysis(
    request: AnalyzeRequest = None,
    x_session_id: str = Header("default_session")
):
    """Run analysis on session datasets."""
    datasets = state.get_all_datasets(x_session_id)
    if not datasets: raise HTTPException(400, "No datasets in session")
    
    # Filter valid dataframes
    valid_datasets = {}
    for d in datasets:
        df = state.get_dataset_df(x_session_id, d['id'])
        if df is not None and not d.get('metadata', {}).get('is_model'):
            valid_datasets[d['id']] = {'df': df, 'metadata': d.get('metadata', {})}
            
    if not valid_datasets:
        return {'status': 'no_data', 'message': 'No data files to analyze'}
        
    analysis = analyzer.generate_analysis(valid_datasets, request.filters if request else None)
    
    # AI Summary
    try:
        profiles = state.get_all_profiles(x_session_id)
        ai = create_groq_ai(GROQ_API_KEY)
        analysis['ai_summary'] = await ai.generate_insights_summary(profiles)
        await ai.close()
    except: pass
    
    state.store_analysis(x_session_id, analysis)
    return analysis

@router.post("/ask")
async def ask_question(
    body: Dict[str, Any] = Body(...),
    x_session_id: str = Header("default_session")
):
    """Context-aware Q&A."""
    question = body.get("question")
    analysis = state.get_analysis(x_session_id)
    profiles = state.get_all_profiles(x_session_id)
    
    state.add_chat_message(x_session_id, {"role": "user", "content": question})
    
    try:
        ai = create_groq_ai(GROQ_API_KEY)
        # TODO: Inject Persona based on detected content types (Legal vs Business)
        result = await ai.answer_question(question, analysis or {}, profiles)
        await ai.close()
        
        state.add_chat_message(x_session_id, {"role": "assistant", "content": result.get("answer")})
        return result
    except Exception as e:
        return {"success": False, "answer": str(e)}

@router.get("/chat/history")
async def get_history(x_session_id: str = Header("default_session")):
    return {"messages": state.get_chat_history(x_session_id)}


# ============== CUSTOM CHARTS / PYTHON ==============

class CustomChartRequest(BaseModel):
    type: str
    x_col: str
    y_col: Optional[str] = None
    aggregation: str = 'sum'
    dataset_id: Optional[str] = None

class PythonPlotRequest(BaseModel):
    code: str
    dataset_id: Optional[str] = None

@router.post("/analyze/chart/custom")
async def custom_chart(
    request: CustomChartRequest,
    x_session_id: str = Header("default_session")
):
    df = None
    if request.dataset_id:
        df = state.get_dataset_df(x_session_id, request.dataset_id)
    else:
        # Default to first
        datasets = state.get_all_datasets(x_session_id)
        if datasets:
            df = state.get_dataset_df(x_session_id, datasets[0]['id'])
            
    if df is None: raise HTTPException(400, "Dataset not found")
    
    return analyzer.generate_custom_chart(df, request.type, request.x_col, request.y_col, request.aggregation)

@router.post("/analyze/python")
async def python_plot(
    request: PythonPlotRequest,
    x_session_id: str = Header("default_session")
):
    df = None
    if request.dataset_id:
        df = state.get_dataset_df(x_session_id, request.dataset_id)
    else:
        datasets = state.get_all_datasets(x_session_id)
        if datasets:
            df = state.get_dataset_df(x_session_id, datasets[0]['id'])
            
    if df is None: raise HTTPException(400, "Dataset not found")
    
    # Sanitization check
    if any(k in request.code for k in ['import os', 'import sys', 'subprocess']):
         raise HTTPException(400, "Unsafe code")
         
    img = analyzer.execute_custom_plot(df, request.code)
    return {'image': f"data:image/png;base64,{img}"}

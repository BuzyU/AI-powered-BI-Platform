# Complete API Routes - Upload, Profile, Clean, Analyze, Ask
from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from typing import List, Dict, Any, Optional
import aiofiles
from pathlib import Path
from uuid import uuid4
import os
import pandas as pd
import logging

from app.config import settings
from app.services.analyzer import DataAnalyzer
from app.services.profiler import DataProfiler
from app.services.cleaner import DataCleaner
from app.services.groq_ai import create_groq_ai
from app.services import state

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
analyzer = DataAnalyzer()
profiler = DataProfiler()
cleaner = DataCleaner()

# Groq API key (from settings or environment)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Ensure upload directory exists
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx', 'json'}


def get_file_extension(filename: str) -> str:
    return filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''


# ============== UPLOAD ROUTES ==============

@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload files and generate initial profiles."""
    tenant_id = "default"
    
    results = []
    
    for file in files:
        ext = get_file_extension(file.filename)
        
        if ext not in ALLOWED_EXTENSIONS:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": f"File type .{ext} not allowed"
            })
            continue
        
        # Save file
        dataset_id = str(uuid4())
        file_path = UPLOAD_DIR / tenant_id / f"{dataset_id}.{ext}"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        try:
            # Load and analyze
            df, metadata = await analyzer.load_file(str(file_path), ext)
            
            # Generate detailed profile
            profile = profiler.profile_dataset(df, file.filename)
            
            # Store everything
            state.store_dataset(tenant_id, dataset_id, {
                'filename': file.filename,
                'file_path': str(file_path),
                'file_type': ext,
                'metadata': metadata,
                'df': df,
                'status': 'profiled'
            })
            
            state.store_profile(tenant_id, dataset_id, profile)
            
            # Issues Breakdown
            missing_count = sum(1 for i in profile['issues'] if i['issue_type'] == 'Missing Values')
            outlier_count = sum(1 for i in profile['issues'] if i['issue_type'] == 'Outliers')
            skew_count = sum(1 for i in profile['issues'] if i['issue_type'] == 'Skewed Distribution')
            
            # Extract detailed skew info
            skew_issues = [i for i in profile['issues'] if i['issue_type'] == 'Skewed Distribution']
            def get_dir(desc):
                if "Right" in desc: return "Right"
                if "Left" in desc: return "Left"
                return ""
            skew_details = []
            for i in skew_issues:
                d = get_dir(i['description'])
                if d: skew_details.append(f"{i['column']} ({d})")
                else: skew_details.append(i['column'])
            
            results.append({
                'id': dataset_id,
                'filename': file.filename,
                'file_type': ext,
                'shape': profile['shape'],
                'overall_quality': profile['overall_quality'],
                'issues_count': len(profile['issues']),
                'issues_summary': {
                    'missing': missing_count,
                    'outliers': outlier_count,
                    'skew': skew_count,
                    'skew_details': ", ".join(skew_details)
                },
                'duplicates_count': profile['duplicates']['count'],
                'status': 'profiled'
            })
            
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            results.append({
                'id': dataset_id,
                'filename': file.filename,
                'file_type': ext,
                'status': 'error',
                'error': str(e)
            })
    
    return {'datasets': results}


# ============== DATASET ROUTES ==============

@router.get("/datasets")
async def list_datasets():
    """List all uploaded datasets with basic info."""
    tenant_id = "default"
    datasets = state.get_all_datasets(tenant_id)
    
    result = []
    for d in datasets:
        profile = state.get_profile(tenant_id, d['id'])
        result.append({
            'id': d['id'],
            'filename': d.get('filename'),
            'file_type': d.get('file_type'),
            'shape': profile.get('shape') if profile else None,
            'overall_quality': profile.get('overall_quality') if profile else None,
            'issues_count': len(profile.get('issues', [])) if profile else 0,
            'status': d.get('status')
        })
    
    return {'datasets': result}


@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get dataset details."""
    tenant_id = "default"
    dataset = state.get_dataset(tenant_id, dataset_id)
    
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    
    profile = state.get_profile(tenant_id, dataset_id)
    
    return {
        'id': dataset_id,
        'filename': dataset.get('filename'),
        'file_type': dataset.get('file_type'),
        'metadata': dataset.get('metadata'),
        'status': dataset.get('status')
    }


@router.get("/datasets/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, rows: int = 100):
    """Get a preview of dataset rows."""
    tenant_id = "default"
    df = state.get_dataset_df(tenant_id, dataset_id)
    
    if df is None:
        raise HTTPException(404, "Dataset not found")
    
    return {
        'id': dataset_id,
        'total_rows': len(df),
        'columns': list(df.columns),
        'data': df.head(rows).to_dict(orient='records')
    }


# ============== PROFILE ROUTES ==============

@router.get("/datasets/{dataset_id}/profile")
async def get_dataset_profile(dataset_id: str):
    """Get detailed profile for a dataset."""
    tenant_id = "default"
    profile = state.get_profile(tenant_id, dataset_id)
    
    if not profile:
        raise HTTPException(404, "Profile not found")
    
    return profile


@router.get("/profiles")
async def get_all_profiles():
    """Get profiles for all datasets."""
    tenant_id = "default"
    profiles = state.get_all_profiles(tenant_id)
    
    return {'profiles': profiles}


# ============== CLEANING ROUTES ==============

@router.post("/datasets/{dataset_id}/clean/preview")
async def preview_cleaning(
    dataset_id: str, 
    operations: List[Dict[str, Any]] = Body(...)
):
    """Preview cleaning operations without applying them."""
    tenant_id = "default"
    df = state.get_dataset_df(tenant_id, dataset_id)
    
    if df is None:
        raise HTTPException(404, "Dataset not found")
    
    preview = cleaner.preview_cleaning(df, operations)
    
    return preview


@router.post("/datasets/{dataset_id}/clean/apply")
async def apply_cleaning(
    dataset_id: str, 
    operations: List[Dict[str, Any]] = Body(...)
):
    """Apply cleaning operations to a dataset."""
    tenant_id = "default"
    df = state.get_dataset_df(tenant_id, dataset_id)
    
    if df is None:
        raise HTTPException(404, "Dataset not found")
    
    # Apply cleaning
    cleaned_df, result = cleaner.apply_cleaning(df, operations)
    
    # Update the dataset
    state.update_dataset_df(tenant_id, dataset_id, cleaned_df)
    
    # Re-generate profile
    dataset = state.get_dataset(tenant_id, dataset_id)
    new_profile = profiler.profile_dataset(cleaned_df, dataset.get('filename', 'dataset'))
    state.store_profile(tenant_id, dataset_id, new_profile)
    
    # Update status
    dataset['status'] = 'cleaned'
    
    return {
        'success': True,
        'result': result,
        'new_profile': {
            'shape': new_profile['shape'],
            'overall_quality': new_profile['overall_quality'],
            'issues_count': len(new_profile['issues'])
        }
    }


# ============== ANALYSIS ROUTES ==============

from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    filters: Optional[Dict[str, Any]] = None

@router.post("/analyze")
async def run_analysis(request: AnalyzeRequest = None):
    """Run full analysis on all datasets."""
    tenant_id = "default"
    filters = request.filters if request else None
    
    datasets = state.get_all_datasets(tenant_id)
    
    if not datasets:
        raise HTTPException(400, "No datasets uploaded")
    
    # Prepare for analysis
    datasets_for_analysis = {}
    for d in datasets:
        if d.get('df') is not None:
            datasets_for_analysis[d['id']] = {
                'df': d['df'],
                'metadata': d.get('metadata', {})
            }
    
    if not datasets_for_analysis:
        raise HTTPException(400, "No valid datasets")
    
    # Find relationships
    dfs = {k: v['df'] for k, v in datasets_for_analysis.items()}
    relationships = analyzer.find_relationships(dfs)
    
    # Generate analysis
    analysis = analyzer.generate_analysis(datasets_for_analysis, filters=filters)
    analysis['relationships'] = relationships
    analysis['datasets'] = [
        {
            'id': d['id'],
            'filename': d.get('filename'),
            'detected_role': d.get('metadata', {}).get('detected_role'),
            'columns': d.get('metadata', {}).get('columns', [])
        }
        for d in datasets
    ]
    
    # Get profiles for AI context
    profiles = state.get_all_profiles(tenant_id)
    
    # Generate AI summary if available
    try:
        ai = create_groq_ai(GROQ_API_KEY)
        ai_summary = await ai.generate_insights_summary(profiles)
        analysis['ai_summary'] = ai_summary
        await ai.close()
    except Exception as e:
        logger.warning(f"Failed to generate AI summary: {e}")
        analysis['ai_summary'] = None
    
    # Store results
    state.store_analysis(tenant_id, analysis)
    
    analysis['status'] = 'complete'
    return analysis


@router.get("/analysis")
async def get_analysis():
    """Get analysis results."""
    tenant_id = "default"
    analysis = state.get_analysis(tenant_id)
    
    if not analysis:
        raise HTTPException(404, "No analysis available. Run /analyze first.")
    
    return analysis


# ============== AI Q&A ROUTES ==============

@router.post("/ask")
async def ask_question(body: Dict[str, Any] = Body(...)):
    """Ask a question about the data using AI."""
    tenant_id = "default"
    question = body.get("question", "")
    
    if not question:
        raise HTTPException(400, "Question is required")
    
    # Get context
    analysis = state.get_analysis(tenant_id)
    profiles = state.get_all_profiles(tenant_id)
    
    if not profiles:
        raise HTTPException(400, "No data has been analyzed yet")
    
    # Store user message
    state.add_chat_message(tenant_id, {
        "role": "user",
        "content": question
    })
    
    try:
        ai = create_groq_ai(GROQ_API_KEY)
        result = await ai.answer_question(
            question=question,
            analysis_context=analysis or {},
            dataset_profiles=profiles
        )
        await ai.close()
        
        # Store AI response
        state.add_chat_message(tenant_id, {
            "role": "assistant",
            "content": result.get("answer", ""),
            "success": result.get("success", False)
        })
        
        return {
            "question": question,
            "answer": result.get("answer"),
            "success": result.get("success", False),
            "model": result.get("model"),
            "tokens": result.get("tokens_used")
        }
        
    except Exception as e:
        logger.error(f"AI Q&A error: {e}")
        return {
            "question": question,
            "answer": f"I encountered an error: {str(e)}. Please try again.",
            "success": False
        }


@router.get("/chat/history")
async def get_chat_history():
    """Get chat history."""
    tenant_id = "default"
    history = state.get_chat_history(tenant_id)
    
    return {"messages": history}


# ============== UTILITY ROUTES ==============

@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset."""
    tenant_id = "default"
    dataset = state.get_dataset(tenant_id, dataset_id)
    
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    
    # Delete file
    file_path = dataset.get('file_path')
    if file_path:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
            else:
                logger.warning(f"File not found for deletion: {file_path}")
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            # Continue to clear state anyway
    
    # Clear from state
    if tenant_id in state._state["datasets"]:
        if dataset_id in state._state["datasets"][tenant_id]:
            del state._state["datasets"][tenant_id][dataset_id]
            logger.info(f"Removed dataset {dataset_id} from state")
    
    if tenant_id in state._state["profiles"]:
        if dataset_id in state._state["profiles"][tenant_id]:
            del state._state["profiles"][tenant_id][dataset_id]
    
    return {"success": True, "message": f"Dataset {dataset_id} deleted"}


@router.post("/reset")
async def reset_all():
    """Reset all data for the tenant."""
    tenant_id = "default"
    state.clear_tenant_data(tenant_id)
    
    return {"success": True, "message": "All data cleared"}


class CustomChartRequest(BaseModel):
    type: str
    x_col: str
    y_col: Optional[str] = None
    aggregation: str = 'sum'
    dataset_id: Optional[str] = None

@router.post("/analyze/chart/custom")
async def generate_custom_chart_endpoint(request: CustomChartRequest):
    """Generate a specific chart based on user selection."""
    tenant_id = "default"
    
    datasets = state.get_all_datasets(tenant_id)
    if not datasets:
         raise HTTPException(400, "No datasets available")
    
    # Select dataset
    target_dataset = None
    if request.dataset_id:
        target_dataset = next((d for d in datasets if d['id'] == request.dataset_id), None)
    
    if not target_dataset:
        # Fallback to first
        target_dataset = datasets[0]
        
    df = target_dataset.get('df')
    if df is None:
        raise HTTPException(400, "Dataset data not loaded")
        
    try:
        chart = analyzer.generate_custom_chart(
            df=df,
            type=request.type,
            x_col=request.x_col,
            y_col=request.y_col,
            aggregation=request.aggregation
        )
        return chart
    except Exception as e:
        raise HTTPException(500, str(e))


class PythonPlotRequest(BaseModel):
    code: str
    dataset_id: Optional[str] = None

@router.post("/analyze/python")
async def execute_python_plot_endpoint(request: PythonPlotRequest):
    """Execute custom python code to generate visualization."""
    tenant_id = "default"
    
    datasets = state.get_all_datasets(tenant_id)
    if not datasets:
         raise HTTPException(400, "No datasets available")
    
    # Select dataset
    target_dataset = None
    if request.dataset_id:
        target_dataset = next((d for d in datasets if d['id'] == request.dataset_id), None)
    
    if not target_dataset:
        target_dataset = datasets[0]
        
    df = target_dataset.get('df')
    if df is None:
        raise HTTPException(400, "Dataset data not loaded")
        
    try:
        # Check against basic malicious keywords (very basic)
        blocked = ['import os', 'import sys', 'subprocess', 'open(', 'eval(', 'exec(']
        if any(b in request.code for b in blocked):
             raise HTTPException(400, "Unsafe code detected")

        image_base64 = analyzer.execute_custom_plot(df, request.code)
        return {'image': f"data:image/png;base64,{image_base64}"}
    except Exception as e:
        raise HTTPException(400, str(e))

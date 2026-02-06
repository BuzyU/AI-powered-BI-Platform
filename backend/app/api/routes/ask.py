# Q&A API Routes - Session-Aware
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, Optional

from app.api.deps import get_session, get_tenant, get_validated_session_id
from app.models.db import Tenant
from app.models.schemas import AskRequest, AnswerResponse
from app.services import state
from app.services.groq_ai import create_groq_ai
from app.config import settings

import structlog

router = APIRouter()
logger = structlog.get_logger()


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(
    request: AskRequest,
    x_session_id: str = Header("default_session"),
    db: AsyncSession = Depends(get_session)
):
    """
    Answer natural language questions about the specific session's data.
    Uses Groq AI with context from the uploaded datasets.
    """
    # 1. Validate Session
    session_id = await get_validated_session_id(x_session_id)
    question = request.question.strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    logger.info("Processing session question", session_id=session_id, question=question[:50])
    
    # 2. Retrieve Session Context
    # We need to know what data the user is talking about
    analysis = state.get_analysis(session_id)
    profiles = state.get_all_profiles(session_id)
    
    # If no analysis exists yet, we can't really answer data questions
    if not analysis and not profiles:
        # Check if they at least have a model loaded
        # (This is a future enhancement: chat with models)
        return AnswerResponse(
            question=question,
            answer="I don't see any analyzed data in this session yet. Please upload a dataset and run analysis, or upload a model.",
            success=False
        )

    # 3. Initialize AI Service
    if not settings.GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set")
        return AnswerResponse(
            question=question,
            answer="AI service is not configured (missing API key). responding with limited context.",
            success=False
        )

    try:
        ai_service = create_groq_ai(settings.GROQ_API_KEY)
        
        # 4. Ask AI with Context
        result = await ai_service.answer_question(
            question=question,
            analysis_context=analysis,
            dataset_profiles=profiles
        )
        
        await ai_service.close()
        
        if result.get("success"):
            return AnswerResponse(
                question=question,
                answer=result["answer"],
                meta=result.get("tokens_used")
            )
        else:
             # Fallback error message from service
             return AnswerResponse(
                question=question,
                answer=result.get("answer", "I encountered an error analyzing your question."),
                success=False
            )
            
    except Exception as e:
        logger.error("AI Question Error", error=str(e))
        return AnswerResponse(
            question=question,
            answer="I ran into a technical issue processing your request.",
            success=False
        )


@router.get("/ask/suggestions")
async def get_question_suggestions(
    x_session_id: str = Header("default_session")
):
    """Get context-aware suggestions based on session data."""
    session_id = await get_validated_session_id(x_session_id)
    analysis = state.get_analysis(session_id)
    
    suggestions = [
        "What patterns do you see in the data?",
        "Are there any outliers?",
        "Summarize the key findings",
        "What is the data quality score?"
    ]
    
    # Add data-specific suggestions if available
    if analysis and analysis.get('kpis'):
        for kpi in analysis['kpis'][:2]:
            suggestions.append(f"Explain the {kpi['label']} metric")
            
    return {"suggestions": suggestions}

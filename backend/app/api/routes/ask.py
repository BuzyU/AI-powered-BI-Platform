# Q&A API Routes
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any
import structlog

from app.api.deps import get_session, get_tenant
from app.models.db import Tenant
from app.models.schemas import AskRequest, AnswerResponse
from app.layers.l10_llm.qa import answer_question

router = APIRouter()
logger = structlog.get_logger()


# Question patterns for routing
QUESTION_PATTERNS = {
    "revenue": ["revenue", "sales", "income", "earnings", "how much"],
    "loss": ["loss", "losing", "negative", "unprofitable"],
    "profit": ["profit", "margin", "profitable"],
    "customer": ["customer", "client", "buyer", "who buys"],
    "product": ["product", "service", "offering", "item"],
    "growth": ["grow", "increase", "improve", "focus"],
    "risk": ["risk", "risky", "dangerous", "concern"],
    "discontinue": ["discontinue", "stop", "remove", "eliminate"],
    "compare": ["compare", "vs", "versus", "difference"]
}


def classify_question(question: str) -> str:
    """Classify question to route to appropriate handler."""
    question_lower = question.lower()
    
    for category, keywords in QUESTION_PATTERNS.items():
        for keyword in keywords:
            if keyword in question_lower:
                return category
    
    return "general"


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(
    request: AskRequest,
    db: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_tenant)
):
    """
    Answer natural language questions about the business data.
    Uses deterministic analytics for data, LLM for explanation.
    """
    question = request.question.strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    logger.info("Processing question", question=question[:100])
    
    try:
        # Classify question
        question_type = classify_question(question)
        
        # Get answer from QA handler
        answer = await answer_question(
            db=db,
            tenant_id=tenant.id,
            question=question,
            question_type=question_type,
            context=request.context
        )
        
        return AnswerResponse(
            question=question,
            answer=answer
        )
        
    except Exception as e:
        logger.error("Question answering failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(e)}"
        )


@router.get("/ask/suggestions")
async def get_question_suggestions(
    db: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_tenant)
):
    """Get suggested questions based on available data."""
    suggestions = [
        "What is my total revenue?",
        "Where am I losing money?",
        "Which products are most profitable?",
        "Which offerings should I focus on?",
        "How many customers do I have?",
        "What is my profit margin?",
        "Which offerings are risky?",
        "Compare this month to last month"
    ]
    
    return {"suggestions": suggestions}

# Groq AI Integration for Data Analysis Q&A
import httpx
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqAI:
    """
    AI-powered Q&A using Groq's fast inference.
    Answers questions based on the analysis performed on the data.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "llama-3.3-70b-versatile"  # Fast and capable
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def answer_question(
        self, 
        question: str, 
        analysis_context: Dict[str, Any],
        dataset_profiles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Answer a question about the data using AI.
        
        Args:
            question: User's natural language question
            analysis_context: Summary of analysis performed
            dataset_profiles: Detailed profiles of each dataset
        """
        
        # Build context for AI
        context = self._build_context(analysis_context, dataset_profiles)
        
        system_prompt = """You are an expert data analyst assistant. The user has uploaded datasets and performed analysis on them.

Your role is to:
1. Answer questions about the data clearly and accurately
2. Reference specific statistics, columns, and findings from the analysis
3. Provide actionable insights when relevant
4. Suggest further analysis if appropriate
5. Be concise but thorough

When discussing numbers:
- Format large numbers with commas (e.g., 1,234,567)
- Round decimals to 2 places for readability
- Use percentages where appropriate

The analysis has already been performed. Base your answers ONLY on the provided analysis data."""
        
        user_prompt = f"""Here is the analysis of the uploaded datasets:

{context}

User Question: {question}

Please answer based on the analysis above. If the question cannot be answered from the available data, explain what additional data or analysis would be needed."""

        try:
            response = await self.client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3,  # Lower for more factual responses
                    "max_tokens": 1500
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                
                return {
                    "success": True,
                    "answer": answer,
                    "model": self.model,
                    "tokens_used": result.get("usage", {})
                }
            else:
                error_msg = response.text
                logger.error(f"Groq API error: {response.status_code} - {error_msg}")
                return {
                    "success": False,
                    "error": f"API Error: {response.status_code}",
                    "answer": "I encountered an error while processing your question. Please try again."
                }
                
        except Exception as e:
            logger.error(f"Groq API exception: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "I encountered an error while processing your question. Please try again."
            }
    
    def _build_context(
        self, 
        analysis_context: Dict[str, Any], 
        dataset_profiles: List[Dict[str, Any]]
    ) -> str:
        """Build a comprehensive context string for the AI."""
        
        parts = []
        
        # Overall summary
        if analysis_context:
            parts.append("=== ANALYSIS SUMMARY ===")
            
            summary = analysis_context.get("summary", {})
            parts.append(f"Total Datasets: {summary.get('datasets', 0)}")
            parts.append(f"Total Rows: {summary.get('total_rows', 0):,}")
            parts.append(f"Total Columns: {summary.get('total_columns', 0)}")
            parts.append(f"Data Quality Score: {summary.get('data_quality', 0):.1f}%")
            parts.append("")
            
            # KPIs
            if analysis_context.get("kpis"):
                parts.append("=== KEY METRICS ===")
                for kpi in analysis_context["kpis"]:
                    value = kpi.get("value", 0)
                    if kpi.get("format") == "currency":
                        parts.append(f"- {kpi['label']}: ${value:,.2f}")
                    elif kpi.get("format") == "percent":
                        parts.append(f"- {kpi['label']}: {value:.1f}%")
                    else:
                        parts.append(f"- {kpi['label']}: {value:,}")
                parts.append("")
            
            # Insights
            if analysis_context.get("insights"):
                parts.append("=== INSIGHTS & ISSUES ===")
                for insight in analysis_context["insights"]:
                    parts.append(f"- [{insight.get('type', 'info').upper()}] {insight['title']}: {insight['description']}")
                parts.append("")
        
        # Dataset profiles
        for profile in dataset_profiles:
            parts.append(f"=== DATASET: {profile.get('dataset_name', 'Unknown')} ===")
            
            shape = profile.get("shape", {})
            parts.append(f"Rows: {shape.get('rows', 0):,}, Columns: {shape.get('columns', 0)}")
            parts.append(f"Data Quality: {profile.get('overall_quality', 0):.1f}%")
            
            # Duplicates
            dups = profile.get("duplicates", {})
            if dups.get("count", 0) > 0:
                parts.append(f"Duplicates: {dups['count']} rows ({dups.get('percentage', 0):.1f}%)")
            
            # Column details
            parts.append("\nColumn Details:")
            for col in profile.get("columns", []):
                col_str = f"  - {col['name']} ({col.get('semantic_type', 'unknown')})"
                
                # Missing values
                missing = col.get("missing", {})
                if missing.get("count", 0) > 0:
                    col_str += f" | Missing: {missing['count']} ({missing.get('percentage', 0):.1f}%)"
                
                # Statistics for numeric
                if "statistics" in col:
                    stats = col["statistics"]
                    col_str += f" | Mean: {stats.get('mean', 0):,.2f}, Median: {stats.get('median', 0):,.2f}"
                    col_str += f", Min: {stats.get('min', 0):,.2f}, Max: {stats.get('max', 0):,.2f}"
                
                # Outliers
                if "outliers" in col and col["outliers"].get("count", 0) > 0:
                    col_str += f" | Outliers: {col['outliers']['count']}"
                
                # Distribution info
                if "distribution" in col:
                    dist = col["distribution"]
                    col_str += f" | Skewness: {dist.get('skewness', 0):.2f}"
                
                parts.append(col_str)
            
            # Correlations
            if profile.get("correlations"):
                parts.append("\nSignificant Correlations:")
                for corr in profile["correlations"][:5]:  # Top 5
                    parts.append(f"  - {corr['column1']} ↔ {corr['column2']}: {corr['correlation']:.3f} ({corr['strength']})")
            
            # Cleaning suggestions
            if profile.get("cleaning_suggestions"):
                parts.append("\nCleaning Suggestions:")
                for sugg in profile["cleaning_suggestions"][:5]:  # Top 5
                    parts.append(f"  - [{sugg['column']}] {sugg['description']}")
            
            parts.append("")
        
        return "\n".join(parts)
    
    async def generate_insights_summary(
        self,
        dataset_profiles: List[Dict[str, Any]]
    ) -> str:
        """Generate an AI summary of the key insights."""
        
        context = self._build_context({}, dataset_profiles)
        
        prompt = f"""Based on this data analysis, provide a brief executive summary (3-5 bullet points) of the most important findings:

{context}

Focus on:
1. Data quality issues that need attention
2. Key statistical patterns
3. Notable correlations
4. Recommended next steps"""

        try:
            response = await self.client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return "Unable to generate summary at this time."
                
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return "Unable to generate summary at this time."
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Factory function to create AI instance
def create_groq_ai(api_key: str) -> GroqAI:
    return GroqAI(api_key)

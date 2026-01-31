"""
LLM Tasks - Celery tasks for AI-powered features

Handles async LLM operations that may take longer:
- Bulk cleaning suggestions
- Detailed explanations
- Feature engineering for large datasets
"""
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from celery import shared_task
from loguru import logger

from config import settings


@shared_task(
    bind=True,
    name="worker.tasks.llm.suggest_cleaning_async",
    max_retries=2,
    default_retry_delay=30,
    track_started=True,
    soft_time_limit=300,  # 5 min soft limit
    time_limit=360,  # 6 min hard limit
)
def suggest_cleaning_async(
    self,
    dataset_id: str,
    dataset_path: str,
) -> Dict[str, Any]:
    """
    Async task for generating cleaning suggestions.
    
    For larger datasets or when immediate response isn't needed.
    """
    import polars as pl
    import httpx
    
    logger.info(f"[LLM] Starting cleaning suggestions for dataset {dataset_id}")
    
    self.update_state(
        state="PROGRESS",
        meta={"progress": 10, "stage": "loading_data"}
    )
    
    # Load dataset
    try:
        df = pl.read_csv(dataset_path)
    except Exception as e:
        return {
            "dataset_id": dataset_id,
            "status": "failed",
            "error": f"Failed to load dataset: {e}",
        }
    
    self.update_state(
        state="PROGRESS",
        meta={"progress": 20, "stage": "analyzing"}
    )
    
    # Calculate statistics
    columns = df.columns
    dtypes = {col: str(df[col].dtype) for col in columns}
    null_counts = {col: df[col].null_count() for col in columns}
    unique_counts = {col: df[col].n_unique() for col in columns}
    sample_data = df.head(10).to_dicts()
    
    self.update_state(
        state="PROGRESS",
        meta={"progress": 40, "stage": "generating_suggestions"}
    )
    
    # Build prompt
    context = f"""Dataset Analysis:
- Total rows: {len(df)}
- Columns: {len(columns)}

Column Details:
"""
    for col in columns:
        null_pct = (null_counts.get(col, 0) / len(df) * 100) if len(df) > 0 else 0
        unique = unique_counts.get(col, 0)
        context += f"- {col}: {dtypes.get(col, 'unknown')}, {null_pct:.1f}% nulls, {unique} unique values\n"
    
    context += f"\nSample data (first 3 rows):\n{json.dumps(sample_data[:3], indent=2, default=str)}"
    
    prompt = f"""{context}

Analyze this dataset and suggest up to 5 data cleaning operations. For each issue found, provide:
1. What the issue is
2. Why it matters  
3. Polars code to fix it

Return a JSON array of suggestions with fields: issue, description, polars_code, confidence"""

    # Call Ollama
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{settings.OLLAMA_URL}/api/generate",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "prompt": prompt,
                    "system": "You are a data cleaning expert. Only suggest Polars operations. Return valid JSON.",
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 2048,
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("response", "")
            
    except Exception as e:
        logger.error(f"[LLM] Ollama request failed: {e}")
        return {
            "dataset_id": dataset_id,
            "status": "failed",
            "error": f"LLM request failed: {e}",
        }
    
    self.update_state(
        state="PROGRESS",
        meta={"progress": 80, "stage": "parsing_response"}
    )
    
    # Parse response
    try:
        # Handle markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        suggestions = json.loads(content.strip())
        if isinstance(suggestions, dict):
            suggestions = [suggestions]
            
    except json.JSONDecodeError:
        suggestions = [{
            "issue": "Analysis Complete",
            "description": content,
            "polars_code": "# See description",
            "confidence": 0.3,
        }]
    
    logger.info(f"[LLM] Generated {len(suggestions)} suggestions for dataset {dataset_id}")
    
    return {
        "dataset_id": dataset_id,
        "status": "completed",
        "suggestions": suggestions[:5],
        "completed_at": datetime.utcnow().isoformat(),
    }


@shared_task(
    bind=True,
    name="worker.tasks.llm.explain_results_async",
    max_retries=2,
    track_started=True,
    soft_time_limit=180,
    time_limit=240,
)
def explain_results_async(
    self,
    job_id: str,
    model_name: str,
    problem_type: str,
    metrics: Dict[str, float],
    feature_importance: Optional[Dict[str, float]] = None,
    confusion_matrix: Optional[List[List[int]]] = None,
) -> Dict[str, Any]:
    """
    Async task for generating detailed model explanations.
    """
    import httpx
    
    logger.info(f"[LLM] Starting explanation for job {job_id}, model {model_name}")
    
    self.update_state(
        state="PROGRESS",
        meta={"progress": 20, "stage": "building_context"}
    )
    
    # Build context
    context = f"""Model Type: {problem_type}
Model Algorithm: {model_name}

Performance Metrics:
"""
    for metric, value in metrics.items():
        if value is not None:
            context += f"- {metric}: {value:.4f}\n"
    
    if feature_importance:
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        context += f"\nTop 10 Important Features:\n"
        for feat, imp in top_features:
            context += f"- {feat}: {imp:.4f}\n"
    
    if confusion_matrix:
        context += f"\nConfusion Matrix (rows=actual, cols=predicted):\n{confusion_matrix}\n"
    
    prompt = f"""{context}

Provide a detailed analysis:
1. Overall model performance assessment
2. Key strengths and what's working well
3. Areas that need improvement
4. Specific actionable recommendations
5. If classification: optimal probability threshold suggestion

Return as JSON with fields: summary, strengths, weaknesses, recommendations, threshold_suggestion"""

    self.update_state(
        state="PROGRESS",
        meta={"progress": 40, "stage": "generating_explanation"}
    )
    
    # Call Ollama
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{settings.OLLAMA_URL}/api/generate",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "prompt": prompt,
                    "system": "You are a machine learning expert. Provide practical, actionable insights. Return valid JSON.",
                    "stream": False,
                    "options": {
                        "temperature": 0.5,
                        "num_predict": 1500,
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("response", "")
            
    except Exception as e:
        logger.error(f"[LLM] Ollama request failed: {e}")
        return {
            "job_id": job_id,
            "status": "failed",
            "error": f"LLM request failed: {e}",
        }
    
    self.update_state(
        state="PROGRESS",
        meta={"progress": 80, "stage": "parsing"}
    )
    
    # Parse response
    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        explanation = json.loads(content.strip())
        
    except json.JSONDecodeError:
        explanation = {
            "summary": content,
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
        }
    
    logger.info(f"[LLM] Generated explanation for job {job_id}")
    
    return {
        "job_id": job_id,
        "model_name": model_name,
        "status": "completed",
        "explanation": explanation,
        "completed_at": datetime.utcnow().isoformat(),
    }


@shared_task(
    bind=True,
    name="worker.tasks.llm.generate_report",
    max_retries=1,
    track_started=True,
    soft_time_limit=300,
    time_limit=360,
)
def generate_report(
    self,
    job_id: str,
    include_cleaning: bool = True,
    include_explanation: bool = True,
    include_feature_suggestions: bool = True,
) -> Dict[str, Any]:
    """
    Generate a comprehensive report for a completed training job.
    
    Combines multiple LLM analyses into a single report.
    """
    import httpx
    
    logger.info(f"[LLM] Generating comprehensive report for job {job_id}")
    
    # This would typically load job data from the database
    # and call multiple analysis functions
    
    self.update_state(
        state="PROGRESS",
        meta={"progress": 10, "stage": "loading_job_data"}
    )
    
    # Placeholder - in production, this would:
    # 1. Load job and model data from DB
    # 2. Generate cleaning suggestions if applicable
    # 3. Generate result explanations
    # 4. Generate feature engineering suggestions
    # 5. Compile into a comprehensive report
    
    report = {
        "job_id": job_id,
        "generated_at": datetime.utcnow().isoformat(),
        "sections": [],
    }
    
    if include_explanation:
        report["sections"].append({
            "title": "Model Performance Analysis",
            "type": "explanation",
            "content": "Detailed explanation would go here",
        })
    
    if include_cleaning:
        report["sections"].append({
            "title": "Data Quality Recommendations",
            "type": "cleaning",
            "content": "Cleaning suggestions would go here",
        })
    
    if include_feature_suggestions:
        report["sections"].append({
            "title": "Feature Engineering Opportunities",
            "type": "features",
            "content": "Feature suggestions would go here",
        })
    
    return {
        "job_id": job_id,
        "status": "completed",
        "report": report,
    }

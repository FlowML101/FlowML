"""
LLM Router - AI-powered data analysis and explanation endpoints

Features:
- /llm/status - Check Ollama availability
- /llm/suggest-cleaning - Get data cleaning suggestions
- /llm/explain-results - Get model result explanations
- /llm/feature-engineering - Get feature engineering suggestions
- /llm/execute-cleaning - Execute approved cleaning code (with preview)
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import polars as pl
from loguru import logger

from database import get_session
from models.dataset import Dataset
from models.trained_model import TrainedModel
from models.job import Job
from services.llm_service import (
    get_llm_service, 
    check_ollama_status,
    CleaningSuggestion,
    ResultExplanation,
)
from exceptions import not_found, bad_request, service_unavailable

router = APIRouter(prefix="/llm", tags=["llm"])


# ============ Request/Response Models ============

class LLMStatusResponse(BaseModel):
    """Ollama status response"""
    available: bool
    url: str
    default_model: str
    models: List[str]


class CleaningRequest(BaseModel):
    """Request for cleaning suggestions"""
    dataset_id: str


class CleaningSuggestionResponse(BaseModel):
    """Single cleaning suggestion"""
    issue: str
    description: str
    polars_code: str
    confidence: float


class CleaningSuggestionsResponse(BaseModel):
    """Response with cleaning suggestions"""
    dataset_id: str
    dataset_name: str
    suggestions: List[CleaningSuggestionResponse]
    llm_available: bool


class ExecuteCleaningRequest(BaseModel):
    """Request to execute cleaning code"""
    dataset_id: str
    polars_code: str
    preview_only: bool = True  # Default to preview mode for safety
    new_dataset_name: Optional[str] = None


class ExecuteCleaningResponse(BaseModel):
    """Response from cleaning execution"""
    success: bool
    preview_rows: Optional[List[Dict[str, Any]]] = None
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    new_dataset_id: Optional[str] = None
    error: Optional[str] = None


class ExplainRequest(BaseModel):
    """Request for result explanation"""
    job_id: Optional[str] = None
    model_id: Optional[str] = None


class ExplainResponse(BaseModel):
    """Model explanation response"""
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    threshold_suggestion: Optional[float] = None
    model_name: Optional[str] = None
    job_id: Optional[str] = None


class FeatureEngineeringRequest(BaseModel):
    """Request for feature engineering suggestions"""
    dataset_id: str
    target_column: str
    problem_type: str = "classification"


class FeatureSuggestion(BaseModel):
    """Single feature engineering suggestion"""
    name: str
    description: str
    polars_code: str


class FeatureEngineeringResponse(BaseModel):
    """Feature engineering suggestions response"""
    dataset_id: str
    suggestions: List[FeatureSuggestion]


# ============ Endpoints ============

@router.get("/status", response_model=LLMStatusResponse)
async def get_llm_status():
    """
    Check Ollama LLM service status.
    
    Returns availability, URL, and available models.
    """
    status = await check_ollama_status()
    return LLMStatusResponse(**status)


@router.post("/suggest-cleaning", response_model=CleaningSuggestionsResponse)
async def suggest_cleaning(
    request: CleaningRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Analyze a dataset and suggest data cleaning operations.
    
    Uses LLM to identify data quality issues and generate
    Polars code to fix them. Requires human approval before execution.
    """
    # Get dataset
    result = await session.execute(
        select(Dataset).where(Dataset.id == request.dataset_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise not_found("Dataset", request.dataset_id)
    
    # Load dataset for analysis
    try:
        df = pl.read_csv(dataset.file_path)
    except Exception as e:
        raise bad_request(f"Failed to load dataset: {e}")
    
    # Calculate statistics
    columns = df.columns
    dtypes = {col: str(df[col].dtype) for col in columns}
    null_counts = {col: df[col].null_count() for col in columns}
    unique_counts = {col: df[col].n_unique() for col in columns}
    sample_data = df.head(10).to_dicts()
    
    # Get LLM service
    llm_service = get_llm_service()
    
    # Check if Ollama is available
    status = await check_ollama_status()
    
    if not status["available"]:
        # Return empty suggestions with availability flag
        return CleaningSuggestionsResponse(
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            suggestions=[],
            llm_available=False,
        )
    
    # Get suggestions from LLM
    suggestions = await llm_service.suggest_cleaning(
        columns=columns,
        dtypes=dtypes,
        sample_data=sample_data,
        null_counts=null_counts,
        unique_counts=unique_counts,
        num_rows=len(df),
    )
    
    return CleaningSuggestionsResponse(
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        suggestions=[
            CleaningSuggestionResponse(
                issue=s.issue,
                description=s.description,
                polars_code=s.polars_code,
                confidence=s.confidence,
            )
            for s in suggestions
        ],
        llm_available=True,
    )


@router.post("/execute-cleaning", response_model=ExecuteCleaningResponse)
async def execute_cleaning(
    request: ExecuteCleaningRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Execute approved cleaning code on a dataset.
    
    By default runs in preview mode (preview_only=True) to show
    the effect before committing. Set preview_only=False to
    create a new cleaned dataset.
    
    SAFETY: Only Polars operations on 'df' variable are allowed.
    """
    # Get dataset
    result = await session.execute(
        select(Dataset).where(Dataset.id == request.dataset_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise not_found("Dataset", request.dataset_id)
    
    # Load dataset
    try:
        df = pl.read_csv(dataset.file_path)
    except Exception as e:
        raise bad_request(f"Failed to load dataset: {e}")
    
    rows_before = len(df)
    columns_before = len(df.columns)
    
    # Validate code (basic security check)
    code = request.polars_code.strip()
    
    # Blocklist dangerous operations
    dangerous_patterns = [
        "import os", "import sys", "import subprocess",
        "open(", "exec(", "eval(", "__", "globals", "locals",
        "import shutil", "pathlib", "requests", "urllib",
    ]
    
    for pattern in dangerous_patterns:
        if pattern in code:
            raise bad_request(f"Code contains disallowed pattern: {pattern}")
    
    # Execute code in restricted namespace
    try:
        # Only allow pl (Polars) in namespace
        namespace = {"df": df, "pl": pl}
        
        # Execute the code
        exec(code, namespace)
        
        # Get the modified dataframe
        df_result = namespace.get("df", df)
        
        if not isinstance(df_result, (pl.DataFrame, pl.LazyFrame)):
            raise bad_request("Code must produce a Polars DataFrame")
        
        if isinstance(df_result, pl.LazyFrame):
            df_result = df_result.collect()
        
    except Exception as e:
        return ExecuteCleaningResponse(
            success=False,
            rows_before=rows_before,
            rows_after=rows_before,
            columns_before=columns_before,
            columns_after=columns_before,
            error=f"Code execution failed: {str(e)}",
        )
    
    rows_after = len(df_result)
    columns_after = len(df_result.columns)
    
    if request.preview_only:
        # Return preview
        preview_rows = df_result.head(20).to_dicts()
        return ExecuteCleaningResponse(
            success=True,
            preview_rows=preview_rows,
            rows_before=rows_before,
            rows_after=rows_after,
            columns_before=columns_before,
            columns_after=columns_after,
        )
    
    # Create new dataset
    from datetime import datetime
    from pathlib import Path
    from config import settings
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_name = request.new_dataset_name or f"{dataset.name}_cleaned"
    new_filename = f"{timestamp}_{new_name}.csv"
    new_path = settings.UPLOAD_DIR / new_filename
    
    # Save cleaned dataset
    df_result.write_csv(new_path)
    
    # Create new dataset record
    new_dataset = Dataset(
        name=new_name,
        filename=new_filename,
        file_path=str(new_path),
        file_size=new_path.stat().st_size,
        num_rows=rows_after,
        num_columns=columns_after,
        columns=json.dumps(df_result.columns),
        dtypes=json.dumps({col: str(df_result[col].dtype) for col in df_result.columns}),
        description=f"Cleaned version of {dataset.name}",
    )
    
    session.add(new_dataset)
    await session.commit()
    await session.refresh(new_dataset)
    
    logger.info(f"Created cleaned dataset {new_dataset.id} from {dataset.id}")
    
    return ExecuteCleaningResponse(
        success=True,
        preview_rows=df_result.head(10).to_dicts(),
        rows_before=rows_before,
        rows_after=rows_after,
        columns_before=columns_before,
        columns_after=columns_after,
        new_dataset_id=new_dataset.id,
    )


@router.post("/explain-results", response_model=ExplainResponse)
async def explain_results(
    request: ExplainRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Get AI-powered explanation of model results.
    
    Provide either job_id (to explain best model) or model_id.
    Returns plain-language insights and recommendations.
    """
    if not request.job_id and not request.model_id:
        raise bad_request("Must provide either job_id or model_id")
    
    # Get model
    if request.model_id:
        result = await session.execute(
            select(TrainedModel).where(TrainedModel.id == request.model_id)
        )
        model = result.scalar_one_or_none()
        if not model:
            raise not_found("Model", request.model_id)
        job_id = model.job_id
    else:
        # Get best model for job
        result = await session.execute(
            select(TrainedModel)
            .where(TrainedModel.job_id == request.job_id)
            .order_by(TrainedModel.rank)
            .limit(1)
        )
        model = result.scalar_one_or_none()
        if not model:
            raise not_found("Models for job", request.job_id)
        job_id = request.job_id
    
    # Get job for problem type
    job_result = await session.execute(select(Job).where(Job.id == job_id))
    job = job_result.scalar_one_or_none()
    problem_type = job.problem_type if job else "classification"
    
    # Build metrics dict
    metrics = {}
    if model.accuracy is not None:
        metrics["accuracy"] = model.accuracy
    if model.f1_score is not None:
        metrics["f1_score"] = model.f1_score
    if model.precision is not None:
        metrics["precision"] = model.precision
    if model.recall is not None:
        metrics["recall"] = model.recall
    if model.auc is not None:
        metrics["auc"] = model.auc
    if model.rmse is not None:
        metrics["rmse"] = model.rmse
    if model.mae is not None:
        metrics["mae"] = model.mae
    if model.r2 is not None:
        metrics["r2"] = model.r2
    
    # Get feature importance
    feature_importance = None
    if model.feature_importance:
        try:
            feature_importance = json.loads(model.feature_importance)
        except:
            pass
    
    # Get confusion matrix
    confusion_matrix = None
    if model.confusion_matrix:
        try:
            confusion_matrix = json.loads(model.confusion_matrix)
        except:
            pass
    
    # Check Ollama availability
    status = await check_ollama_status()
    
    if not status["available"]:
        return ExplainResponse(
            summary="LLM service is not available. Please start Ollama to get AI-powered explanations.",
            strengths=[],
            weaknesses=[],
            recommendations=["Start Ollama: docker-compose up ollama"],
            model_name=model.name,
            job_id=job_id,
        )
    
    # Get explanation from LLM
    llm_service = get_llm_service()
    
    explanation = await llm_service.explain_results(
        problem_type=problem_type or "classification",
        metrics=metrics,
        feature_importance=feature_importance,
        confusion_matrix=confusion_matrix,
    )
    
    return ExplainResponse(
        summary=explanation.summary,
        strengths=explanation.strengths,
        weaknesses=explanation.weaknesses,
        recommendations=explanation.recommendations,
        threshold_suggestion=explanation.threshold_suggestion,
        model_name=model.name,
        job_id=job_id,
    )


@router.post("/feature-engineering", response_model=FeatureEngineeringResponse)
async def suggest_feature_engineering(
    request: FeatureEngineeringRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Get AI suggestions for feature engineering.
    
    Analyzes dataset structure and suggests new features
    that could improve model performance.
    """
    # Get dataset
    result = await session.execute(
        select(Dataset).where(Dataset.id == request.dataset_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise not_found("Dataset", request.dataset_id)
    
    # Parse columns and dtypes
    columns = json.loads(dataset.columns)
    dtypes = json.loads(dataset.dtypes)
    
    # Check if target column exists
    if request.target_column not in columns:
        raise bad_request(f"Target column '{request.target_column}' not found in dataset")
    
    # Check Ollama
    status = await check_ollama_status()
    
    if not status["available"]:
        return FeatureEngineeringResponse(
            dataset_id=dataset.id,
            suggestions=[],
        )
    
    # Get suggestions
    llm_service = get_llm_service()
    
    suggestions = await llm_service.generate_feature_engineering(
        columns=columns,
        dtypes=dtypes,
        target_column=request.target_column,
        problem_type=request.problem_type,
    )
    
    return FeatureEngineeringResponse(
        dataset_id=dataset.id,
        suggestions=[
            FeatureSuggestion(
                name=s.get("name", ""),
                description=s.get("description", ""),
                polars_code=s.get("polars_code", ""),
            )
            for s in suggestions
        ],
    )


@router.post("/chat")
async def chat_with_llm(
    message: str,
    context: Optional[str] = None,
):
    """
    General chat endpoint for LLM interaction.
    
    Can be used for ad-hoc questions about data or ML.
    """
    status = await check_ollama_status()
    
    if not status["available"]:
        raise service_unavailable("Ollama LLM")
    
    llm_service = get_llm_service()
    
    prompt = message
    if context:
        prompt = f"Context: {context}\n\nQuestion: {message}"
    
    response = await llm_service.client.generate(
        prompt=prompt,
        system="You are FlowML Assistant, an AI helper for data science and machine learning tasks. Be helpful, concise, and practical.",
        temperature=0.7,
    )
    
    if not response.success:
        raise HTTPException(status_code=500, detail=response.error)
    
    return {
        "response": response.content,
        "model": response.model,
        "tokens_used": response.tokens_used,
    }

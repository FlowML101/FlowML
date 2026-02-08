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
from services.cleaning_operations import (
    CleaningOperation,
    CleaningExecutor,
    analyze_dataset,
    FillMissingOperation,
    DropMissingOperation,
    DropDuplicatesOperation,
    CastTypeOperation,
    RemoveOutliersOperation,
    StandardizeOperation,
    NormalizeOperation,
    StringCleanOperation,
    RenameColumnOperation,
    DropColumnOperation,
    EncodeCategoricalOperation,
    OperationType,
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


class CleaningOperationSuggestion(BaseModel):
    """Single cleaning operation suggestion (safe, structured)"""
    operation: str  # Operation type from OperationType enum
    column: Optional[str] = None
    columns: Optional[List[str]] = None
    description: str
    confidence: float
    parameters: Dict[str, Any]  # Operation-specific parameters


class CleaningSuggestionsResponse(BaseModel):
    """Response with safe cleaning operation suggestions"""
    dataset_id: str
    dataset_name: str
    suggestions: List[CleaningOperationSuggestion]
    dataset_analysis: Dict[str, Any]  # Full dataset statistics
    llm_available: bool


class ApplyOperationsRequest(BaseModel):
    """Request to apply cleaning operations"""
    dataset_id: str
    operations: List[Dict[str, Any]]  # List of operation configurations
    preview_only: bool = True  # Default to preview mode for safety
    new_dataset_name: Optional[str] = None


class ApplyOperationsResponse(BaseModel):
    """Response from applying operations"""
    success: bool
    preview_rows: Optional[List[Dict[str, Any]]] = None
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    operations_applied: List[str]
    new_dataset_id: Optional[str] = None
    new_dataset_name: Optional[str] = None
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
    Analyze a dataset and suggest SAFE cleaning operations.
    
    Uses LLM to identify data quality issues and returns structured
    operations (not arbitrary code). Operations are from a predefined
    safe whitelist and can be executed without security risks.
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
        df = pl.read_csv(
            dataset.file_path,
            infer_schema_length=None,
            ignore_errors=True
        )
    except Exception as e:
        raise bad_request(f"Failed to load dataset: {e}")
    
    # Analyze dataset comprehensively
    dataset_analysis = analyze_dataset(df)
    
    # Check if Ollama is available
    status = await check_ollama_status()
    
    if not status["available"]:
        # Return empty suggestions with analysis
        return CleaningSuggestionsResponse(
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            suggestions=[],
            dataset_analysis=dataset_analysis,
            llm_available=False,
        )
    
    # Get LLM service and get suggestions
    llm_service = get_llm_service()
    suggestions_raw = await llm_service.suggest_cleaning(dataset_analysis)
    
    # Convert to response format
    suggestions = []
    for sug in suggestions_raw:
        suggestions.append(CleaningOperationSuggestion(
            operation=sug.get('operation', 'unknown'),
            column=sug.get('column'),
            columns=sug.get('columns'),
            description=sug.get('description', 'No description'),
            confidence=sug.get('confidence', 0.5),
            parameters=sug.get('parameters', {}),
        ))
    
    return CleaningSuggestionsResponse(
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        suggestions=suggestions,
        dataset_analysis=dataset_analysis,
        llm_available=True,
    )


@router.post("/apply-operations", response_model=ApplyOperationsResponse)
async def apply_operations(
    request: ApplyOperationsRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Apply safe cleaning operations to a dataset.
    
    Operations are executed from a predefined whitelist - NO arbitrary
    code execution. By default runs in preview mode. Set preview_only=False
    to create a new cleaned dataset.
    
    SAFE: All operations are validated and executed through CleaningExecutor.
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
        df = pl.read_csv(
            dataset.file_path,
            infer_schema_length=None,
            ignore_errors=True
        )
    except Exception as e:
        raise bad_request(f"Failed to load dataset: {e}")
    
    rows_before = len(df)
    columns_before = len(df.columns)
    applied_operations = []
    
    # Apply operations sequentially
    try:
        for op_config in request.operations:
            operation_type = op_config.get('operation')
            column = op_config.get('column')
            columns = op_config.get('columns')
            description = op_config.get('description', 'No description')
            parameters = op_config.get('parameters', {})
            
            # Create appropriate operation object based on type
            operation = None
            
            if operation_type == OperationType.FILL_MISSING:
                operation = FillMissingOperation(
                    column=column,
                    strategy=parameters.get('strategy', 'mean'),
                    fill_value=parameters.get('fill_value'),
                    description=description,
                )
            elif operation_type == OperationType.DROP_MISSING:
                operation = DropMissingOperation(
                    axis=parameters.get('axis', 'rows'),
                    threshold=parameters.get('threshold'),
                    columns=parameters.get('columns'),
                    description=description,
                )
            elif operation_type == OperationType.DROP_DUPLICATES:
                operation = DropDuplicatesOperation(
                    columns=parameters.get('columns'),
                    keep=parameters.get('keep', 'first'),
                    description=description,
                )
            elif operation_type == OperationType.CAST_TYPE:
                operation = CastTypeOperation(
                    column=column,
                    target_type=parameters.get('target_type', 'string'),
                    date_format=parameters.get('date_format'),
                    description=description,
                )
            elif operation_type == OperationType.REMOVE_OUTLIERS:
                operation = RemoveOutliersOperation(
                    column=column,
                    method=parameters.get('method', 'iqr'),
                    threshold=parameters.get('threshold', 1.5),
                    percentile_range=parameters.get('percentile_range'),
                    description=description,
                )
            elif operation_type == OperationType.STANDARDIZE:
                operation = StandardizeOperation(
                    column=column,
                    description=description,
                )
            elif operation_type == OperationType.NORMALIZE:
                operation = NormalizeOperation(
                    column=column,
                    description=description,
                )
            elif operation_type == OperationType.STRING_CLEAN:
                operation = StringCleanOperation(
                    column=column,
                    operations=parameters.get('operations', ['strip']),
                    description=description,
                )
            elif operation_type == OperationType.RENAME_COLUMN:
                operation = RenameColumnOperation(
                    column=column,
                    new_name=parameters.get('new_name', f"{column}_renamed"),
                    description=description,
                )
            elif operation_type == OperationType.DROP_COLUMN:
                operation = DropColumnOperation(
                    column=column,
                    description=description,
                )
            elif operation_type == OperationType.ENCODE_CATEGORICAL:
                operation = EncodeCategoricalOperation(
                    column=column,
                    method=parameters.get('method', 'label'),
                    description=description,
                )
            else:
                logger.warning(f"Unknown operation type: {operation_type}")
                continue
            
            if operation:
                # Execute the operation safely
                df = CleaningExecutor.execute(df, operation)
                applied_operations.append(f"{operation_type}: {description}")
        
    except Exception as e:
        logger.error(f"Failed to apply operations: {e}")
        return ApplyOperationsResponse(
            success=False,
            rows_before=rows_before,
            rows_after=rows_before,
            columns_before=columns_before,
            columns_after=columns_before,
            operations_applied=applied_operations,
            error=f"Operation failed: {str(e)}",
        )
    
    rows_after = len(df)
    columns_after = len(df.columns)
    
    if request.preview_only:
        # Return preview
        preview_rows = df.head(20).to_dicts()
        return ApplyOperationsResponse(
            success=True,
            preview_rows=preview_rows,
            rows_before=rows_before,
            rows_after=rows_after,
            columns_before=columns_before,
            columns_after=columns_after,
            operations_applied=applied_operations,
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
    df.write_csv(new_path)
    
    # Get new dataset info
    new_columns = df.columns
    new_dtypes = {col: str(df[col].dtype) for col in new_columns}
    
    # Create new dataset record with versioning
    new_dataset = Dataset(
        name=new_name,
        filename=new_filename,
        file_path=str(new_path),
        file_size=new_path.stat().st_size,
        num_rows=len(df),
        num_columns=len(new_columns),
        columns=json.dumps(new_columns),
        dtypes=json.dumps(new_dtypes),
        description=f"Cleaned version of '{dataset.name}' with {len(applied_operations)} operations",
        parent_id=dataset.id,  # Track lineage
        version=dataset.version + 1,
        operation_history=json.dumps(applied_operations),
    )
    
    session.add(new_dataset)
    await session.commit()
    await session.refresh(new_dataset)
    
    return ApplyOperationsResponse(
        success=True,
        rows_before=rows_before,
        rows_after=rows_after,
        columns_before=columns_before,
        columns_after=columns_after,
        operations_applied=applied_operations,
        new_dataset_id=new_dataset.id,
        new_dataset_name=new_dataset.name,
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

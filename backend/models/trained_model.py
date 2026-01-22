"""
Trained Model - represents a model produced by AutoML
"""
from datetime import datetime
from sqlmodel import SQLModel, Field
from typing import Optional
import uuid


class TrainedModelBase(SQLModel):
    """Base trained model fields"""
    name: str = Field(index=True)  # e.g., "XGBoost", "Random Forest"
    job_id: str = Field(foreign_key="jobs.id", index=True)
    dataset_id: str = Field(foreign_key="datasets.id", index=True)
    
    # Metrics
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    auc: Optional[float] = None
    rmse: Optional[float] = None  # For regression
    mae: Optional[float] = None   # For regression
    r2: Optional[float] = None    # For regression
    
    # Training info
    training_time: float  # seconds
    rank: int = Field(default=0)  # Leaderboard rank
    
    # Model file
    model_path: Optional[str] = None
    model_size: Optional[int] = None  # bytes


class TrainedModel(TrainedModelBase, table=True):
    """Trained model database model"""
    __tablename__ = "trained_models"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Hyperparameters (JSON string)
    hyperparameters: Optional[str] = None
    
    # Feature importance (JSON string)
    feature_importance: Optional[str] = None
    
    # Confusion matrix (JSON string) - for classification
    confusion_matrix: Optional[str] = None


class TrainedModelRead(TrainedModelBase):
    """Schema for reading a trained model"""
    id: str
    created_at: datetime
    hyperparameters: Optional[dict] = None
    feature_importance: Optional[dict] = None


class ModelLeaderboard(SQLModel):
    """Leaderboard entry"""
    rank: int
    id: str
    name: str
    accuracy: Optional[float]
    f1_score: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    auc: Optional[float]
    training_time: float


class PredictionRequest(SQLModel):
    """Single prediction request"""
    model_id: str
    features: dict


class PredictionResponse(SQLModel):
    """Prediction response"""
    prediction: str | float | int
    probability: Optional[float] = None
    confidence: Optional[float] = None
    model_name: str
    latency_ms: float

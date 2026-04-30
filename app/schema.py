"""Pydantic models for request/response validation."""
from typing import List, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime


class LoanApplicationRequest(BaseModel):
    """Single loan application for prediction."""
    
    age: int = Field(..., ge=18, le=100, description="Age in years")
    income_monthly: float = Field(..., gt=0, description="Monthly income in INR")
    loan_amount: float = Field(..., gt=0, description="Loan amount in INR")
    loan_tenure_m: int = Field(..., ge=6, le=120, description="Tenure in months")
    interest_rate: float = Field(..., ge=0, le=50, description="Interest rate %")
    existing_loans: int = Field(..., ge=0, description="Number of existing loans")
    credit_score: int = Field(..., ge=300, le=900, description="Credit score")
    employment_type: str = Field(..., description="Salaried/Self-Employed/Business/Freelancer")
    education: str = Field(..., description="Education level")
    residence_type: str = Field(..., description="Own/Rented/Family")
    delinquency_30d: int = Field(..., ge=0, le=20, description="Days past 30 in last 12m")
    utilisation_pct: float = Field(..., ge=0, le=100, description="Credit utilization %")
    months_employed: int = Field(..., ge=0, description="Months in current employment")
    loan_purpose: str = Field(..., description="Purpose of loan")
    
    @validator('employment_type')
    def validate_employment_type(cls, v):
        allowed = {'Salaried', 'Self-Employed', 'Business', 'Freelancer'}
        if v not in allowed:
            raise ValueError(f'employment_type must be one of {allowed}')
        return v
    
    @validator('education')
    def validate_education(cls, v):
        allowed = {'Under-Graduate', 'Graduate', 'Post-Graduate', 'Professional'}
        if v not in allowed:
            raise ValueError(f'education must be one of {allowed}')
        return v
    
    @validator('residence_type')
    def validate_residence(cls, v):
        allowed = {'Own', 'Rented', 'Family'}
        if v not in allowed:
            raise ValueError(f'residence_type must be one of {allowed}')
        return v


class PredictionResponse(BaseModel):
    """Single prediction response."""
    
    default_probability: float = Field(..., description="Probability of default (0-1)")
    predicted_class: int = Field(..., description="0=Non-Default, 1=Default")
    risk_score: float = Field(..., ge=0, ge=100, description="Risk score 0-100")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    model_version: str = Field(..., description="MLflow model version/stage")
    timestamp: datetime


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    
    loans: List[LoanApplicationRequest] = Field(..., min_items=1, max_items=10000)
    request_id: Optional[str] = Field(None, description="Tracking ID for batch")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    
    request_id: str
    total_records: int
    successful_predictions: int
    failed_records: int
    predictions: List[PredictionResponse]
    processing_time_seconds: float


class HealthCheckResponse(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool
    mlflow_connected: bool
    database_healthy: bool
    timestamp: datetime


class ModelMetricsResponse(BaseModel):
    """Current model performance metrics."""
    
    model_version: str
    gini_coefficient: float = Field(..., ge=0, le=1, description="Gini on validation set")
    ks_statistic: float = Field(..., ge=0, le=1, description="KS statistic")
    auc_roc: float
    precision: float
    recall: float
    f1_score: float
    predictions_24h: int
    avg_latency_ms: float
    drift_detected: bool
    days_since_retrain: int
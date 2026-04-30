"""Production FastAPI app for loan default predictions."""
import os
import logging
import time
from datetime import datetime
from typing import List
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest

from .schema import (
    LoanApplicationRequest, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, HealthCheckResponse, ModelMetricsResponse
)
from .model_loader import get_model_loader


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
predictions_counter = Counter(
    'predictions_total',
    'Total predictions made',
    ['model_version', 'prediction_class']
)
prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)
batch_size_histogram = Histogram(
    'batch_size',
    'Batch prediction sizes'
)


# Global state
model_loader = None
prediction_stats = {
    'total_predictions': 0,
    'latencies': [],
    'last_retrain': datetime.now(),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle."""
    # Startup
    global model_loader
    logger.info("🚀 Starting FastAPI app...")
    model_loader = get_model_loader()
    
    if not model_loader.is_loaded():
        logger.error("❌ Failed to load model on startup")
        raise RuntimeError("Model failed to load")
    
    logger.info("✓ Model loaded successfully")
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down FastAPI app...")


# Create app
app = FastAPI(
    title="FinRisk MLOps API",
    description="Production-grade loan default prediction API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy" if model_loader.is_loaded() else "degraded",
        model_loaded=model_loader.is_loaded(),
        mlflow_connected=True,  # TODO: Add MLflow connectivity check
        database_healthy=True,  # TODO: Add DB check
        timestamp=datetime.now()
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return JSONResponse(content=generate_latest().decode())


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: LoanApplicationRequest):
    """Predict default probability for single loan application."""
    try:
        start_time = time.time()
        
        # Prepare input
        input_df = pd.DataFrame([request.dict()])
        
        # Preprocess if available
        if model_loader.preprocessor:
            input_processed = model_loader.preprocessor.transform(input_df)
        else:
            input_processed = input_df
        
        # Predict
        default_prob = float(model_loader.model.predict_proba(input_processed)[0][1])
        predicted_class = int(model_loader.model.predict(input_processed)[0])
        
        # Risk score (0-100)
        risk_score = max(0, min(100, default_prob * 100))
        
        # Confidence (use prediction probability as confidence)
        confidence = max(default_prob, 1 - default_prob)
        
        latency = time.time() - start_time
        prediction_stats['latencies'].append(latency)
        prediction_stats['total_predictions'] += 1
        
        # Record metrics
        predictions_counter.labels(
            model_version=model_loader.model_version,
            prediction_class=predicted_class
        ).inc()
        prediction_latency.observe(latency)
        
        logger.info(
            f"Prediction: prob={default_prob:.3f}, class={predicted_class}, "
            f"latency={latency*1000:.1f}ms"
        )
        
        return PredictionResponse(
            default_probability=default_prob,
            predicted_class=predicted_class,
            risk_score=risk_score,
            confidence=confidence,
            model_version=model_loader.model_version,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction for multiple applications."""
    try:
        start_time = time.time()
        batch_size_histogram.observe(len(request.loans))
        
        predictions = []
        failed_records = 0
        
        # Process each loan
        for i, loan in enumerate(request.loans):
            try:
                input_df = pd.DataFrame([loan.dict()])
                
                if model_loader.preprocessor:
                    input_processed = model_loader.preprocessor.transform(input_df)
                else:
                    input_processed = input_df
                
                default_prob = float(model_loader.model.predict_proba(input_processed)[0][1])
                predicted_class = int(model_loader.model.predict(input_processed)[0])
                risk_score = max(0, min(100, default_prob * 100))
                confidence = max(default_prob, 1 - default_prob)
                
                predictions.append(PredictionResponse(
                    default_probability=default_prob,
                    predicted_class=predicted_class,
                    risk_score=risk_score,
                    confidence=confidence,
                    model_version=model_loader.model_version,
                    timestamp=datetime.now()
                ))
                
            except Exception as e:
                logger.warning(f"Failed to predict record {i}: {e}")
                failed_records += 1
                continue
        
        processing_time = time.time() - start_time
        
        return BatchPredictionResponse(
            request_id=request.request_id or f"batch-{int(time.time())}",
            total_records=len(request.loans),
            successful_predictions=len(predictions),
            failed_records=failed_records,
            predictions=predictions,
            processing_time_seconds=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-metrics", response_model=ModelMetricsResponse)
async def get_model_metrics():
    """Get current model performance metrics."""
    try:
        # TODO: Fetch from MLflow experiment runs
        avg_latency = (
            np.mean(prediction_stats['latencies']) * 1000
            if prediction_stats['latencies']
            else 0
        )
        
        days_since_retrain = (
            datetime.now() - prediction_stats['last_retrain']
        ).days
        
        return ModelMetricsResponse(
            model_version=model_loader.model_version,
            gini_coefficient=0.42,  # TODO: fetch from MLflow
            ks_statistic=0.35,
            auc_roc=0.75,
            precision=0.70,
            recall=0.68,
            f1_score=0.69,
            predictions_24h=prediction_stats['total_predictions'],
            avg_latency_ms=avg_latency,
            drift_detected=False,  # TODO: call monitoring service
            days_since_retrain=days_since_retrain
        )
        
    except Exception as e:
        logger.error(f"Failed to fetch metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "FinRisk MLOps API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "batch_predict": "/batch-predict (POST)",
            "health": "/health (GET)",
            "metrics": "/metrics (GET)",
            "model_metrics": "/model-metrics (GET)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        workers=int(os.getenv("WORKERS", 4))
    )
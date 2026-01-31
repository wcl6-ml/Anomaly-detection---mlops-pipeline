"""FastAPI application for serving fraud detection model."""
import sys
from pathlib import Path
import os

# CRITICAL: Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Change working directory to project root (fixes relative paths)
os.chdir(project_root)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import mlflow.pyfunc
import numpy as np
import pandas as pd
from datetime import datetime
import logging

from config.mlflow_config import setup_mlflow

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time anomaly detection for fraud prevention",
    version="1.0.0"
)

# Global model variable (loaded on startup)
model = None
model_metadata = {}


class PredictionRequest(BaseModel):
    """Request schema for predictions."""
    features: List[List[float]] = Field(
        ..., 
        description="List of feature vectors",
        example=[[0.1, 0.2, 0.3, 0.4, 0.5]]
    )


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    predictions: List[int]
    anomaly_scores: List[float]
    model_version: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


# Startup event
@app.on_event("startup")
async def load_model():
    """Load model from MLflow registry on startup."""
    global model, model_metadata
    
    setup_mlflow()
    
    try:
        # Load production model
        model_uri = "models:/fraud-detector/Production"
        logger.info(f"Loading model from: {model_uri}")
        
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Get model metadata
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Find production version
        versions = client.search_model_versions("name='fraud-detector'")
        prod_versions = [v for v in versions if v.current_stage == "Production"]
        
        if prod_versions:
            prod_version = prod_versions[0]
            model_metadata = {
                "version": prod_version.version,
                "run_id": prod_version.run_id,
                "created_at": prod_version.creation_timestamp
            }
            logger.info(f"Model loaded: version {prod_version.version}")
        else:
            model_metadata = {"version": "unknown"}
            logger.warning("No production version found")
        
        # Store startup time
        model_metadata["startup_time"] = datetime.now()
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint with model validation.
    
    Returns model status and basic diagnostics.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Calculate uptime
    uptime = (datetime.now() - model_metadata.get("startup_time", datetime.now())).total_seconds()
    
    # Test inference on dummy data
    try:
        dummy_input = pd.DataFrame([[0.0] * 30])  # Adjust feature count
        _ = model.predict(dummy_input)
        status = "healthy"
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        status = "unhealthy"
    
    return HealthResponse(
        status=status,
        model_loaded=model is not None,
        model_version=model_metadata.get("version", "unknown"),
        uptime_seconds=uptime
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict anomalies for given features.
    
    Returns predictions and anomaly scores.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.features)
        
        # Time the inference
        start_time = datetime.now()
        
        # Get predictions (scores for isolation forest)
        predictions = model.predict(df)
        
        inference_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Convert to list
        if hasattr(predictions, 'tolist'):
            pred_list = predictions.tolist()
        else:
            pred_list = list(predictions)
        
        # For isolation forest, scores are negative (more negative = more anomalous)
        # Convert to anomaly scores [0, 1]
        anomaly_scores = [abs(float(p)) for p in pred_list]
        
        # Binary predictions (1 = anomaly)
        threshold = np.percentile(anomaly_scores, 90)  # Top 10% are anomalies
        binary_preds = [1 if score > threshold else 0 for score in anomaly_scores]
        
        # Log prediction (in production, send to monitoring service)
        logger.info(f"Predicted {sum(binary_preds)}/{len(binary_preds)} anomalies")
        
        return PredictionResponse(
            predictions=binary_preds,
            anomaly_scores=anomaly_scores,
            model_version=model_metadata.get("version", "unknown"),
            inference_time_ms=inference_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def model_info():
    """Get information about the currently loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "fraud-detector",
        "version": model_metadata.get("version", "unknown"),
        "run_id": model_metadata.get("run_id", "unknown"),
        "loaded_at": model_metadata.get("startup_time", "unknown").isoformat() if model_metadata.get("startup_time") else "unknown",
        "stage": "Production"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
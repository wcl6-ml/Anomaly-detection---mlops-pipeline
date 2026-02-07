"""FastAPI application for serving fraud detection model."""
import sys
from pathlib import Path
import os

# Add project root to path and change working directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import mlflow.pyfunc
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from prometheus_fastapi_instrumentator import Instrumentator
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import yaml

from src.drift.detector import DriftDetector

# Define the local path where Docker will have the model
MODEL_PATH = Path(__file__).parent / "model/artifacts"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time anomaly detection for fraud prevention",
    version="1.0.0"
)

# For Grafana monitoring
prediction_counter = Counter('model_predictions_total', 'Total predictions made')
prediction_latency = Histogram('model_inference_seconds', 'Model inference latency')
anomaly_score_gauge = Gauge('model_anomaly_score', 'Latest average anomaly score')
feature_null_rate = Gauge('feature_null_rate', 'Null rate in input features', ['feature_index'])
error_counter = Counter('prediction_errors_total', 'Total prediction errors', ['error_type'])

# Define Gauge for Prometheus
MODEL_DRIFT_GAUGE = Gauge(
    "model_drift_psi", 
    "Population Stability Index for data drift",
    ["feature"]
)

# Initialize Instrumentator but DON'T expose yet
instrumentator = Instrumentator().instrument(app)

# Create the Prometheus ASGI app
metrics_app = make_asgi_app()

# Mount it to the /metrics path
app.mount("/metrics", metrics_app)

# Global model variable
model = None
model_metadata = {}

# For drift detection
drift_detector = None  

class PredictionRequest(BaseModel):
    """Request schema for predictions."""
    features: List[List[float]] = Field(
        ..., 
        description="List of feature vectors (each should have 29 features)",
        example=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    )


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    model_config = {"protected_namespaces": ()}  
    predictions: List[int]
    anomaly_scores: List[float]
    model_version: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


@app.on_event("startup")
async def load_model():
    """Load model from local folder on startup."""
    global model, model_metadata, drift_detector
    
    try:
        logger.info(f"Looking for model in: {MODEL_PATH}")
        
        # 1. Check if the directory exists
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model directory not found at {MODEL_PATH}")
        
        # 2. Load the model using pyfunc from the LOCAL path
        model = mlflow.pyfunc.load_model(str(MODEL_PATH))
        
        # 3. Extract metadata from the MLmodel file
        mlmodel_file = MODEL_PATH / "MLmodel"
        if mlmodel_file.exists():
            with open(mlmodel_file, "r") as f:
                config = yaml.safe_load(f)
                model_metadata = {
                    "run_id": config.get("run_id", "unknown"),
                    "version": "baked-in",
                    "utc_time_created": config.get("utc_time_created", "unknown")
                }
        
        model_metadata["startup_time"] = datetime.now()
        logger.info(f"Successfully loaded model from {MODEL_PATH}")
        
        # 4. Load reference data for drift detection
        ref_path = project_root / "data/processed/reference.csv"
        if not ref_path.exists():
            logger.warning(f"Reference data not found at {ref_path}. Drift detection disabled.")
            return
        
        reference_df = pd.read_csv(ref_path)
        
        # Drop columns not used in prediction
        cols_to_drop = ['Time', 'Class']
        reference_df = reference_df.drop(columns=cols_to_drop, errors='ignore')
        
        # 5. Calculate "System Noise" (Self-PSI)
        mid = len(reference_df) // 2
        ref_a = reference_df.iloc[:mid]
        ref_b = reference_df.iloc[mid:]
        
        temp_detector = DriftDetector(ref_a)
        self_drift_results = temp_detector.detect_drift(ref_b)
        self_psi = self_drift_results['overall_psi']
        
        # 6. Set dynamic threshold
        dynamic_threshold = 0.28  # ← Change the threshold for alerting here
        
        drift_detector = DriftDetector(reference_df, threshold_psi=dynamic_threshold)
        
        logger.info(f"Drift detector initialized. Base Noise: {self_psi:.4f}")
        logger.info(f"Dynamic Alert Threshold set to: {dynamic_threshold:.4f}")
        
    except Exception as e:
        logger.error(f"Critical error loading model: {e}")
        model = None

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "status": "running" if model is not None else "model_not_loaded",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with model validation."""
    if model is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_version="none",
            uptime_seconds=0.0
        )
    
    # Calculate uptime
    uptime = (datetime.now() - model_metadata.get("startup_time", datetime.now())).total_seconds()
    
    # Test inference on dummy data
    try:
        # Adjust number of features based on your model
        dummy_input = pd.DataFrame([[0.0] * 29])
        _ = model.predict(dummy_input)
        status = "healthy"
    except Exception as e:
        logger.error(f"Health check inference failed: {e}")
        status = "degraded"
    
    return HealthResponse(
        status=status,
        model_loaded=True,
        model_version=str(model_metadata.get("version", "unknown")),
        uptime_seconds=uptime
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict anomalies for given features."""
    if model is None:
        error_counter.labels(error_type='model_not_loaded').inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame WITH COLUMN NAMES
        if drift_detector is not None:
            df = pd.DataFrame(request.features, columns=drift_detector.feature_names)
        else:
            df = pd.DataFrame(request.features)
        
        # Track null rates per feature
        for i in range(len(df.columns)):
            null_rate = df.iloc[:, i].isna().sum() / len(df)
            feature_null_rate.labels(feature_index=i).set(null_rate)
        
        logger.info(f"Received prediction request with {len(df)} samples, {len(df.columns)} features")
        
        # Time the inference
        start_time = datetime.now()
        
        # Get predictions
        predictions = model.predict(df)
        
        inference_time = (datetime.now() - start_time).total_seconds()
        
        # Record latency metric
        prediction_latency.observe(inference_time)
        
        # Convert to list
        if hasattr(predictions, 'tolist'):
            pred_list = predictions.tolist()
        else:
            pred_list = list(predictions)
        
        # For isolation forest, scores are negative (more negative = more anomalous)
        anomaly_scores = [abs(float(p)) for p in pred_list]
        
        # Binary predictions (1 = anomaly)
        if len(anomaly_scores) > 1:
            threshold = np.percentile(anomaly_scores, 90)
        else:
            threshold = 0.0
            
        binary_preds = [1 if score > threshold else 0 for score in anomaly_scores]
        
        # Record metrics
        prediction_counter.inc(len(binary_preds))
        anomaly_score_gauge.set(np.mean(anomaly_scores))
                # AFTER the predictions, ADD DRIFT DETECTION:
        if drift_detector is not None:
            try:
                drift_results = drift_detector.detect_drift(df)
                
                # Emit overall PSI
                MODEL_DRIFT_GAUGE.labels(feature="overall").set(drift_results['overall_psi'])
                
                # Optionally emit per-feature PSI (for detailed monitoring)
                for feature, metrics in drift_results['feature_drifts'].items():
                    MODEL_DRIFT_GAUGE.labels(feature=feature).set(metrics['psi'])
                
                if drift_results['drift_detected']:
                    logger.warning(f"DRIFT ALERT: Overall PSI={drift_results['overall_psi']:.4f}")
            except Exception as e:
                logger.error(f"Drift detection failed: {e}")
        
        logger.info(f"Predicted {sum(binary_preds)}/{len(binary_preds)} anomalies in {inference_time*1000:.2f}ms")
        
        return PredictionResponse(
            predictions=binary_preds,
            anomaly_scores=anomaly_scores,
            model_version=str(model_metadata.get("version", "unknown")),
            inference_time_ms=inference_time * 1000
        )
        
    except KeyError as e:
        error_counter.labels(error_type='missing_field').inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Missing field: {str(e)}")
    except Exception as e:
        error_counter.labels(error_type='model_error').inc()
        logger.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/model-info")
async def model_info():
    """Get information about the currently loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "fraud-detector",
        "version": str(model_metadata.get("version", "unknown")),
        "run_id": str(model_metadata.get("run_id", "unknown")),
        "loaded_at": model_metadata.get("startup_time").isoformat() if model_metadata.get("startup_time") else "unknown",
        "stage": "Production"
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server from __main__")
    uvicorn.run(app, host="0.0.0.0", port=8000)
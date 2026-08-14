# Anomaly Detection MLOps Pipeline (Foundational Prototype)

An end-to-end MLOps system built to explore experiment tracking, model serving, drift detection, and observability infrastructure.

**Project Focus:** This project represents my foundational hands-on exploration transitioning from academic PhD research into MLOps and DevOps engineering. Rather than focusing on offline model tuning, the goal was to build a complete end-to-end operational workflow—connecting FastAPI serving, PostgreSQL prediction logging, Population Stability Index (PSI) drift detection, and Prometheus/Grafana monitoring in a containerized environment.

## Key Capabilities Implemented

- **Containerized Model Serving:** FastAPI app with Dockerized deployment, custom 422 error handlers, and health checks.
- **Observability & Metrics:** Prometheus instrumentation paired with Grafana dashboards for latency (p50/p95), request counts, and error tracking.
- **Data Drift Detection:** Population Stability Index (PSI) calculation to identify input distribution shifts in real-time.
- **Persistence & Audit Logging:** PostgreSQL integration for logging prediction metadata, drift metrics, and validation failures.
- **Simulated Production Workloads:** Custom load generators to test batch processing, streaming traffic, and drift triggers.

---

## Architecture & Project Structure

**Dataset:** Credit Card Fraud Detection (Kaggle)  
**Models:** Isolation Forest vs. Autoencoder  
**Stack:** MLflow, FastAPI, Docker, PostgreSQL, Prometheus, Grafana, GitHub Actions  

The project separates data preparation, experiment tracking, serving logic, and monitoring infrastructure:

* `src/prepare_data.py`: Splits data using time-based splits (reference vs. sequential test batches).
* `src/training.py`: Trains, evaluates, and logs models to MLflow.
* `src/utils/register_model.py`: Registers the top-performing model based on evaluation metrics.
* `src/utils/export_model.py`: Exports the registered model artifact to the serving directory.



Key folder structure:
```
├── src/
│   ├── prepare_data.py
│   ├── training.py
│   └── utils/
│       ├── register_model.py
│       └── export_model.py
│   └── drift/
│       └── detector.py
├── serve/
│   ├── app.py
│   └── Dockerfile.prod      # Production container
├── scripts/
│   ├── batch_processor.py   # Batch inference with drift detection
│   └── generate_mixed_load.py # Load testing
├── monitoring/
│   ├── prometheus.yml       # Metrics collection config
│   └── alerts.yml           # Drift & latency alerting rules
└── .github/workflows/       # CI/CD automation
```


---

## Getting Started

### Environment Setup

This project uses VS Code Dev Containers for consistent local development setup (`.devcontainer`).

```bash
code .

# or via DevPod
devpod up .
````

### 1. Data Preparation & Model Training

The pipeline splits the raw dataset into a **Reference set** (for baseline calculations) and **10 sequential batches** to simulate streaming traffic.

Bash

```
# Prepare data splits
python src/prepare_data.py

# Run training (logs experiments to MLflow)
python src/training.py

# Register best model
python src/utils/register_model.py

# Export artifact for serving
python src/utils/export_model.py
```

### 2. Local Model Serving

Bash

```
# Start FastAPI service locally
python -m uvicorn serve.app:app --port 8000

# Perform health check
curl http://localhost:8000/health
```

### 3. Containerized Deployment

Bash

```
# Build & run production container
docker build -t anomaly-detection:v1 -f serve/Dockerfile.prod .
docker run -p 8000:8000 anomaly-detection:v1
```

### 4. Full Monitoring Stack

Bash

```
# Spin up PostgreSQL, Prometheus, Grafana, and API service
docker-compose up -d

# Simulate production traffic & data drift
python scripts/generate_mixed_load.py
```

### 5. Automated Pipeline Script

Bash

```
./train_to_deploy.sh
```

## API Usage

The serving API accepts feature vectors, returning predictions (`1` for anomaly, `0` for normal), anomaly scores, and inference latency metadata.

### Single Sample Inference

Bash

```
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{ 
        "features": [
          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
           0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
           0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ],
        "batch_id": "batch_001"
      }'
```

**Example Response:**

JSON

```
{
  "predictions": [0],
  "anomaly_scores": [0.12],
  "model_version": "1.0.0",
  "inference_time_ms": 8.45,
  "psi_score": 0.04,
  "batch_id": "batch_001"
}
```

## Monitoring, Drift & Database Queries

### Metrics Tracked

- **Model Behavior:** Anomaly rate over time, PSI data drift score.
    
- **System Health:** Inference latency (p50/p95), error counts, feature null rates.
    

### Backtracking & Audit Logs via PostgreSQL

All prediction attempts (including HTTP 422 schema validation failures) are logged to PostgreSQL for auditing:

Bash

```
# Connect to PostgreSQL container
psql -h localhost -p 5432 -U user -d monitoring_db
```

SQL

```
-- Query recent prediction records
SELECT batch_id, status, anomaly_rate, psi_score, inference_time_ms 
FROM prediction_logs 
ORDER BY timestamp DESC 
LIMIT 5;

-- Inspect failed requests or malformed payloads
SELECT batch_id, error_message 
FROM prediction_logs 
WHERE status != 'SUCCESS';
```

## Summary of Completed Development Phases

- [x] **Phase 1–3:** Data batching, MLflow experiment tracking, and model evaluation.
    
- [x] **Phase 4:** FastAPI serving endpoint with custom validation exception handling.
    
- [x] **Phase 5:** Metric export via Prometheus & visualization in Grafana.
    
- [x] **Phase 6:** Batch simulation & drift detection using Population Stability Index (PSI).
    
- [x] **Phase 7:** Basic CI/CD pipeline automation with GitHub Actions.
    
- [x] **Phase 8:** Postgres database integration for persistent audit logs and batch tracking.


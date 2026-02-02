# End-to-End Anomaly Detection Pipeline (MLOps)

This project implements a anomaly detection system for credit card fraud, focusing on the **MLOps lifecycle**: experiment tracking, model registration, and containerized deployment.

## Project Goals

- **Reproducibility:** Containerize and deploy on a server.
    
- **Experiment Tracking:** Use MLflow to compare models (Isolation Forest vs. Autoencoders).
    
- **Production Simulation:** Simulate real-world data drift by splitting datasets into sequential  batches.
    
- **Model Serving:** Deploy the best model using FastAPI and Docker.
    

---

## Architecture & Project Structure

The project follows a modular design to separate data concerns from training and serving logic:

* `src/prepare_data.py`: splits data using time-based splits (reference-val-untouched) 
	* The dataset used is Credit Card Fraud Detection from Kaggle
	* Raw data is located in `data/raw/`
* `src/training.py`: trains, evaluates, and logs the models.
* `src/utils/register_model.py`: registers the best model based on the given metric.
* `src/utils/export_model.py`: loads and shifts the best model registered in mlflow.db to serve folder

---

## Getting Started

### 1. Environment Setup

This project uses dev container with VSCode for development (.devcontainer folder).


### 2. Data Preparation & Training

The pipeline splits the raw dataset into a **Reference** set (for training) and **10 sequential batches** to simulate streaming production data.

```
# Prepare data
python src/prepare_data.py

# Run training (logs to MLflow)
python src/training.py

# Register model 
python src/utils/register_model.py

# Shift best model for deployment
python src/utils/export_model.py

# Test locally
python -m uvicorn serve.app:app --port 8000
```

### 3. Model Serving (Docker)

In the project folder,
```
# Build the production image
docker build -t anomaly-detection:v1 -f serve/Dockerfile.prod .

# Run the container
docker run -p 8000:8000 anomaly-detection:v1
```


## API Usage

The model accepts a list of features and returns the prediction (`1` for anomaly, `0` for normal) along with its anomaly score and inference metadata.

**Example Request:**
```
curl -X POST http://IP_ADDRESS:8000/predict \
 -H "Content-Type: application/json" \
 -d '{ "features": [
		[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
		 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
		  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
		] 
	 }'
# 29 features in total
```

**Example Response:**

JSON

```
{
  "predictions": [1],
  "anomaly_scores": [1.0],
  "model_version": "baked-in",
  "inference_time_ms": 9.39
}

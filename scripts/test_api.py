"""Test the FastAPI serving application."""
import requests
import json

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    response = requests.get(f"{API_URL}/health")
    print(f"Health Check: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_predict():
    """Test prediction endpoint."""
    # Create dummy features (adjust to your feature count)
    payload = {
        "features": [
            [0.1] * 30,  # Normal transaction
            [5.0] * 30   # Anomalous transaction
        ]
    }
    
    response = requests.post(f"{API_URL}/predict", json=payload)
    print(f"\nPrediction: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_model_info():
    """Test model info endpoint."""
    response = requests.get(f"{API_URL}/model-info")
    print(f"\nModel Info: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_health()
    test_model_info()
    test_predict()
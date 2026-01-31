"""Test the FastAPI serving application."""
import requests
import json
import sys

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Health check passed")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Unexpected status code: {response.status_code}")
            print(f"Response text: {response.text}")
    except requests.exceptions.ConnectionError:
        print("Cannot connect to API. Is the server running?")
        print("   Start with: python -m uvicorn serve.app:app --reload --port 8000")
        sys.exit(1)
    except requests.exceptions.JSONDecodeError as e:
        print(f"Invalid JSON response: {e}")
        print(f"Response text: {response.text}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def test_root():
    """Test root endpoint."""
    print("\n" + "="*60)
    print("Testing Root Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Root endpoint working")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def test_model_info():
    """Test model info endpoint."""
    print("\n" + "="*60)
    print("Testing Model Info Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=5)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Model info retrieved")
            print(json.dumps(response.json(), indent=2))
        elif response.status_code == 503:
            print("Model not loaded (503)")
            print(response.json())
        else:
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def test_predict():
    """Test prediction endpoint."""
    print("\n" + "="*60)
    print("Testing Prediction Endpoint")
    print("="*60)
    
    # Adjust feature count based on your model
    # Check by looking at your training data
    n_features = 29  # Default, adjust if needed
    
    payload = {
        "features": [
            [0.1] * n_features,  # Normal transaction
            [5.0] * n_features   # Anomalous transaction
        ]
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict", 
            json=payload,
            timeout=10
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Prediction successful")
            result = response.json()
            print(json.dumps(result, indent=2))
            
            # Validate response
            assert "predictions" in result
            assert "anomaly_scores" in result
            assert "model_version" in result
            assert "inference_time_ms" in result
            print("\nAll expected fields present")
            
        elif response.status_code == 503:
            print("Model not loaded (503)")
            print(response.json())
        elif response.status_code == 422:
            print(" error (422)")
            print("This might mean wrong number of features")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("FastAPI Fraud Detection API - Test Suite")
    print("="*60)
    print(f"Testing API at: {API_URL}")
    
    test_root()
    test_health()
    test_model_info()
    test_predict()
    
    print("\n" + "="*60)
    print("Test Suite Complete")
    print("="*60)

if __name__ == "__main__":
    main()